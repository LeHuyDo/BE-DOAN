import logging
import os
import sys
import random
from datetime import datetime
sys.path.append(os.getcwd())

import numpy as np
import torch
from torch.cuda.amp import GradScaler

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

from eva_clip import create_model_and_transforms, trace_model, get_tokenizer

from training.data import get_data
from training.logger import setup_logging
from training.params import parse_args
from training.scheduler import warmup_cosine_lr
from training.train import train_one_epoch, evaluate, extract_features
from training.optim import create_optimizer

def random_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def main(args):
    args, _ = parse_args(args)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = True
        
    args.model = args.model.replace('/', '-')

    if args.name is None:
        args.name = '-'.join([
            datetime.now().strftime("%Y_%m_%d-%H_%M_%S"),
            f"model_{args.model}",
            f"lr_{args.lr}",
            f"b_{args.batch_size}",
            f"j_{args.workers}",
            f"p_{args.precision}",
        ])
    else:
        args.name = '-'.join([
            args.name,
            datetime.now().strftime("%Y_%m_%d-%H")
        ])

    args.distributed = False  # Disable distributed training
    args.local_rank = 0
    args.rank = 0
    args.world_size = 1

    log_base_path = os.path.join(args.logs, args.name)
    os.makedirs(log_base_path, exist_ok=True)
    log_filename = 'out.log'
    args.log_path = os.path.join(log_base_path, log_filename)

    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    if args.tensorboard:
        args.tensorboard_path = os.path.join(args.logs, args.name, "tensorboard")
        args.checkpoint_path = os.path.join(args.logs, args.name, "checkpoints")
        os.makedirs(args.tensorboard_path, exist_ok=True)
        os.makedirs(args.checkpoint_path, exist_ok=True)

    random_seed(args.seed)

    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_clip=args.force_custom_clip,
        force_patch_dropout=args.force_patch_dropout,
        pretrained_image=args.pretrained_image,
        pretrained_text=args.pretrained_text,
    )

    model.to(device)
    model_without_ddp = model

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    logging.info(f'number of params: {sum(p.numel() for p in model.parameters())}')

    optimizer = None
    scaler = None
    if args.train_data or args.train_data_list or args.dataset_type == "synthetic":
        scaler = GradScaler() if args.precision == "amp" else None
        optimizer = create_optimizer(args, model_without_ddp)

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location='cpu')
            if 'epoch' in checkpoint:
                start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["state_dict"])
                if optimizer:
                    optimizer.load_state_dict(checkpoint["optimizer"])
                if scaler and 'scaler' in checkpoint:
                    scaler.load_state_dict(checkpoint['scaler'])
            else:
                model.load_state_dict(checkpoint)

    data = get_data(args, (preprocess_train, preprocess_val), epoch=start_epoch, tokenizer=get_tokenizer(args.model))

    scheduler = None
    if 'train' in data and optimizer:
        total_steps = data["train"].dataloader.num_batches * args.epochs
        scheduler = warmup_cosine_lr(optimizer, args, total_steps)

    writer = None
    if args.tensorboard:
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb:
        wandb.init(project=args.wandb_project_name, name=args.name, config=vars(args))
        if args.debug:
            wandb.watch(model, log='all')

    if args.extract_features:
        with torch.no_grad():
            extract_features(model, data, args, device)
        return
        
    if 'train' not in data:
        evaluate(model, data, start_epoch, args, writer)
        return

    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, writer)

        if 'val' in data:
            evaluate(model, data, epoch + 1, args, writer)

        if args.checkpoint_path and (epoch + 1) % args.save_frequency == 0:
            checkpoint_dict = {
                "epoch": epoch + 1,
                "state_dict": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler else None,
            }
            torch.save(checkpoint_dict, os.path.join(args.checkpoint_path, f"epoch_{epoch + 1}.pt"))

    if args.wandb:
        wandb.finish()

if __name__ == "__main__":
    main(sys.argv[1:])
