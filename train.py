import json
import logging
import math
import os
import time
import gc

import numpy as np
import torch
import torch.nn as nn
from torch._six import inf
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None

from eva_clip import ClipLoss, get_cast_dtype, get_tokenizer
from .utils import save_file

class AverageMeter(object):
    """Tính và lưu trữ giá trị trung bình và giá trị hiện tại"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model

def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len parameters == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm.to(dtype=torch.float32)

def train_one_epoch(model, data, epoch, optimizer, scaler, scheduler, args, tb_writer=None):
    device = torch.device(args.device)
    autocast = torch.cuda.amp.autocast  # Sử dụng AMP của PyTorch để huấn luyện với độ chính xác hỗn hợp
    cast_dtype = torch.float16 nếu args.precision == 'fp16' ngược lại torch.float32

    model.train()
    loss_fn = ClipLoss(local_loss=args.local_loss)

    dataloader = data['train'].dataloader
    num_batches_per_epoch = len(dataloader)
    loss_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    for i, batch in enumerate(dataloader):
        step = num_batches_per_epoch * epoch + i

        images, texts = batch
        # Chuyển dữ liệu sang GPU
        images = images.to(device, dtype=cast_dtype)
        texts = texts.to(device)

        # Đo thời gian tải dữ liệu
        data_time_m.update(time.time() - end)

        optimizer.zero_grad()

        with autocast():  # Bối cảnh AMP cho huấn luyện với độ chính xác hỗn hợp
            image_features, text_features, logit_scale = model(images, texts)
            total_loss, acc = loss_fn(image_features, text_features, logit_scale)

        # Lan truyền ngược
        if scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        # Tùy chọn kẹp giá trị logit scale (cụ thể cho huấn luyện CLIP)
        unwrap_model(model).logit_scale.clamp_(0, math.log(100))

        # Đo thời gian thực hiện batch
        batch_time_m.update(time.time() - end)
        end = time.time()

        # Ghi nhật ký sau mỗi n bước
        if i % args.log_every_n_steps == 0:
            batch_size = images.size(0)
            logging.info(f"Epoch: {epoch} [{i}/{num_batches_per_epoch}] "
                         f"Loss: {total_loss.item():.4f} "
                         f"i2t_acc: {acc['i2t'].item() * 100:.2f} "
                         f"t2i_acc: {acc['t2i'].item() * 100:.2f}")

            # Đặt lại bộ đếm sau khi ghi nhật ký
            batch_time_m.reset()
            data_time_m.reset()

    return loss_m.avg

def evaluate(model, data, epoch, args):
    device = torch.device(args.device)
    model.eval()
    metrics = {}

    if 'val' in data:
        dataloader = data['val'].dataloader
        cumulative_loss = 0.0
        num_samples = 0

        with torch.no_grad():
            for batch in dataloader:
                images, texts = batch
                images = images.to(device, dtype=cast_dtype)
                texts = texts.to(device)

                image_features, text_features, logit_scale = model(images, texts)
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                batch_size = images.shape[0]
                labels = torch.arange(batch_size, device=device).long()
                total_loss = (F.cross_entropy(logits_per_image, labels) + 
                              F.cross_entropy(logits_per_text, labels)) / 2

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size

        loss = cumulative_loss / num_samples
        metrics["val_loss"] = loss.item()

    logging.info(f"Eval Epoch: {epoch}, Val Loss: {metrics['val_loss']:.4f}")
    return metrics
