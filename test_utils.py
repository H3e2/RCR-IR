from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from data_processing_rcr_ir import IRBenchmarks
from net.rcr_ir import RCRIR, RCRIRConfig
from rcr_ir_utils import (
    AverageMeter,
    compute_psnr_ssim,
    load_model_weights,
    match_spatial,
    pad_to_multiple,
    save_tensor_image,
    unpad_tensor,
)


def collate_eval(batch):
    metas = [item[0] for item in batch]
    degraded = torch.stack([item[1] for item in batch], dim=0)
    clean = torch.stack([item[2] for item in batch], dim=0)
    return metas, degraded, clean


def build_model(
    ckpt_path: str,
    device: torch.device,
    state: str = "auto",
    base_dim: int = 64,
) -> RCRIR:
    model = RCRIR(RCRIRConfig(base_dim=base_dim)).to(device)
    load_model_weights(model, ckpt_path, state=state, strict=True)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params / 1e6:.3f} M)")
    return model


def build_ir_dataset(data_file_dir: str, benchmark: str):
    args = SimpleNamespace(
        data_file_dir=data_file_dir,
        de_type=[
            "denoise_15", "denoise_25", "denoise_50",
            "dehaze", "derain", "deblur", "synllie",
        ],
        benchmarks=[benchmark],
    )
    return IRBenchmarks(args)


@torch.inference_mode()
def evaluate_dataset(
    model: RCRIR,
    dataset,
    device: torch.device,
    save_dir: str = "",
) -> Tuple[float, float, int]:
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_eval,
    )
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    for metas, degraded, clean in loader:
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        padded, pads = pad_to_multiple(degraded, 16)
        restored = model(padded)
        restored = unpad_tensor(restored, pads).clamp(0, 1)
        restored, clean = match_spatial(restored, clean)

        psnr, ssim, n = compute_psnr_ssim(restored, clean)
        psnr_meter.update(psnr, n)
        ssim_meter.update(ssim, n)

        if save_dir:
            src_path = metas[0][0]
            filename = os.path.basename(src_path)
            if not os.path.splitext(filename)[1]:
                filename += ".png"
            save_tensor_image(os.path.join(save_dir, filename), restored)

    return psnr_meter.avg, ssim_meter.avg, psnr_meter.count


def print_summary(results: Dict[str, Dict[str, float]]) -> Tuple[float, float]:
    avg_psnr = sum(v["psnr"] for v in results.values()) / len(results)
    avg_ssim = sum(v["ssim"] for v in results.values()) / len(results)

    print("\n" + "=" * 72)
    for name, values in results.items():
        print(
            f"{name:16s}  PSNR={values['psnr']:.4f}  "
            f"SSIM={values['ssim']:.4f}  N={int(values['samples'])}"
        )
    print("-" * 72)
    print(f"Average           PSNR={avg_psnr:.4f}  SSIM={avg_ssim:.4f}")
    print("=" * 72)
    return avg_psnr, avg_ssim
