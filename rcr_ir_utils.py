from __future__ import annotations

import copy
import math
import os
import random
from datetime import datetime
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def seed_everything(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_runtime(allow_tf32: bool = True, cudnn_benchmark: bool = True) -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(allow_tf32)
        torch.backends.cudnn.allow_tf32 = bool(allow_tf32)
    torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
    torch.set_float32_matmul_precision("high")


def make_run_dir(root: str, name: str) -> str:
    stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = os.path.join(root, f"{name}_{stamp}")
    os.makedirs(os.path.join(path, "checkpoints"), exist_ok=True)
    return path


def haar_dwt2d(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = x.shape[-2:]
    if h < 2 or w < 2:
        z = torch.zeros_like(x)
        return x, z, z, z
    if h % 2 or w % 2:
        x = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect")
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 - x01 + x10 - x11)
    hl = 0.5 * (x00 + x01 - x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)
    return ll, lh, hl, hh


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = float(eps)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


class GradientLoss(nn.Module):
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def build_reliability_targets(
    degraded: torch.Tensor,
    target: torch.Tensor,
    sizes: Sequence[Tuple[int, int]],
    alpha: float = 2.0,
    beta: float = 0.5,
) -> List[torch.Tensor]:
    cur_deg, cur_tgt = degraded, target
    out: List[torch.Tensor] = []
    for size in sizes:
        ll_d, lh_d, hl_d, hh_d = haar_dwt2d(cur_deg)
        ll_t, lh_t, hl_t, hh_t = haar_dwt2d(cur_tgt)
        diff_low = (ll_d - ll_t).abs().mean(dim=1, keepdim=True)
        diff_high = (
            (lh_d - lh_t).abs().mean(dim=1, keepdim=True)
            + (hl_d - hl_t).abs().mean(dim=1, keepdim=True)
            + (hh_d - hh_t).abs().mean(dim=1, keepdim=True)
        )
        diff = diff_low + float(beta) * diff_high
        diff = diff / diff.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        rel = torch.exp(-float(alpha) * diff)
        out.append(F.interpolate(rel, size=size, mode="bilinear", align_corners=False))
        cur_deg, cur_tgt = ll_d, ll_t
    return out


class ReliabilityLoss(nn.Module):
    def __init__(self, alpha: float = 2.0, beta: float = 0.5):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(
        self,
        pred_maps: Sequence[torch.Tensor],
        degraded: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        if not pred_maps:
            return target.new_tensor(0.0)
        targets = build_reliability_targets(
            degraded,
            target,
            [m.shape[-2:] for m in pred_maps],
            alpha=self.alpha,
            beta=self.beta,
        )
        return sum(F.l1_loss(p, t.detach()) for p, t in zip(pred_maps, targets)) / len(pred_maps)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.module = copy.deepcopy(model).eval()
        self.decay = float(decay)
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema = self.module.state_dict()
        src = model.state_dict()
        for key, value in ema.items():
            source = src[key].detach()
            if torch.is_floating_point(value):
                value.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
            else:
                value.copy_(source)


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def pad_to_multiple(x: torch.Tensor, mult: int = 16) -> Tuple[torch.Tensor, Tuple[int, int]]:
    _, _, h, w = x.shape
    pad_h = (mult - h % mult) % mult
    pad_w = (mult - w % mult) % mult
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="reflect"), (pad_h, pad_w)


def unpad_tensor(x: torch.Tensor, pads: Tuple[int, int]) -> torch.Tensor:
    pad_h, pad_w = pads
    if pad_h:
        x = x[:, :, :-pad_h, :]
    if pad_w:
        x = x[:, :, :, :-pad_w]
    return x


def match_spatial(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    h = min(pred.shape[-2], target.shape[-2])
    w = min(pred.shape[-1], target.shape[-1])
    return pred[..., :h, :w], target[..., :h, :w]


def compute_psnr_ssim(pred: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, int]:
    """PromptIR/AnyIR-compatible RGB metrics on [0, 1] images, averaged per image."""
    assert pred.shape == target.shape, (pred.shape, target.shape)
    pred_np = np.clip(pred.detach().float().cpu().numpy(), 0.0, 1.0).transpose(0, 2, 3, 1)
    tgt_np = np.clip(target.detach().float().cpu().numpy(), 0.0, 1.0).transpose(0, 2, 3, 1)

    psnr = 0.0
    ssim = 0.0
    for p, t in zip(pred_np, tgt_np):
        psnr += peak_signal_noise_ratio(t, p, data_range=1.0)
        try:
            ssim += structural_similarity(t, p, data_range=1.0, channel_axis=2)
        except TypeError:
            ssim += structural_similarity(t, p, data_range=1.0, multichannel=True)
    n = pred_np.shape[0]
    return psnr / n, ssim / n, n


def save_tensor_image(path: str, x: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arr = x.detach().float().cpu().clamp(0, 1)
    if arr.ndim == 4:
        arr = arr[0]
    arr = (arr.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(arr).save(path)


def autocast_context(device: torch.device, precision: str):
    enabled = device.type == "cuda" and precision in {"fp16", "bf16"}
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def cosine_lr(step: int, total_steps: int, warmup_steps: int) -> float:
    total_steps = max(int(total_steps), 1)
    warmup_steps = max(min(int(warmup_steps), total_steps), 0)
    step = max(int(step), 0)
    if warmup_steps and step < warmup_steps:
        ratio = float(step + 1) / float(warmup_steps)
        return 0.5 + 0.5 * ratio
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return max(0.5 * (1.0 + math.cos(math.pi * progress)), 1e-6)


def _strip_prefix(state: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if state and all(k.startswith(prefix) for k in state):
        return {k[len(prefix):]: v for k, v in state.items()}
    return state


def extract_state_dict(checkpoint, state: str = "auto") -> Dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        return checkpoint

    if state == "ema" and "ema_state" in checkpoint:
        out = checkpoint["ema_state"]
    elif state == "raw" and "model_state" in checkpoint:
        out = checkpoint["model_state"]
    elif state == "auto":
        if "ema_state" in checkpoint:
            out = checkpoint["ema_state"]
        elif "model_state" in checkpoint:
            out = checkpoint["model_state"]
        elif "state_dict" in checkpoint:
            out = checkpoint["state_dict"]
        else:
            out = checkpoint
    elif "state_dict" in checkpoint:
        out = checkpoint["state_dict"]
    else:
        out = checkpoint

    out = _strip_prefix(out, "module.")
    out = _strip_prefix(out, "_orig_mod.")
    out = _strip_prefix(out, "net.")
    return out


def load_model_weights(model: nn.Module, ckpt_path: str, state: str = "auto", strict: bool = True) -> None:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    weights = extract_state_dict(checkpoint, state=state)
    incompatible = model.load_state_dict(weights, strict=strict)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(f"Loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"Missing keys ({len(missing)}): {missing[:10]}")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {unexpected[:10]}")
