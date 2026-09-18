from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_processing_rcr_ir import CDD11
from rcr_ir_utils import AverageMeter, compute_psnr_ssim, match_spatial, pad_to_multiple, save_tensor_image, unpad_tensor
from test_utils import build_model, collate_eval


def parse_args():
    p = argparse.ArgumentParser(description="RCR-IR CDD-11 evaluation")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--data_file_dir", type=str, default="")
    p.add_argument("--subset", choices=["all", "single", "double", "triple"], default="all")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--state", choices=["auto", "ema", "raw"], default="auto")
    p.add_argument("--output_dir", type=str, default="results_rcr_ir/cdd11")
    p.add_argument("--save_results", action="store_true")
    return p.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    np.random.seed(0)
    torch.manual_seed(0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args.ckpt_path, device, state=args.state)

    ds_args = SimpleNamespace(data_file_dir=args.data_file_dir, patch_size=128)
    dataset = CDD11(ds_args, split="test", subset=args.subset)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_eval,
    )

    meters = defaultdict(lambda: {"psnr": AverageMeter(), "ssim": AverageMeter()})
    global_psnr = AverageMeter()
    global_ssim = AverageMeter()

    for metas, degraded, clean in loader:
        src_path, category = metas[0]
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        padded, pads = pad_to_multiple(degraded, 16)
        restored = model(padded)
        restored = unpad_tensor(restored, pads).clamp(0, 1)
        restored, clean = match_spatial(restored, clean)

        psnr, ssim, n = compute_psnr_ssim(restored, clean)
        meters[category]["psnr"].update(psnr, n)
        meters[category]["ssim"].update(ssim, n)
        global_psnr.update(psnr, n)
        global_ssim.update(ssim, n)

        if args.save_results:
            save_dir = os.path.join(args.output_dir, args.subset, category)
            save_tensor_image(os.path.join(save_dir, os.path.basename(src_path)), restored)

    results = {}
    print("\n" + "=" * 78)
    for category in sorted(meters):
        p = meters[category]["psnr"].avg
        s = meters[category]["ssim"].avg
        n = meters[category]["psnr"].count
        results[category] = {"psnr": p, "ssim": s, "samples": n}
        print(f"{category:18s} PSNR={p:.4f} SSIM={s:.4f} N={n}")

    # Paper-style CDD-11 averages: single / double / triple / overall.
    grouped = {}
    for level_name, level_count in [("single", 1), ("double", 2), ("triple", 3)]:
        selected = [
            v for k, v in results.items()
            if k.count("_") + 1 == level_count
        ]
        if selected:
            grouped[level_name] = {
                "psnr": float(np.mean([v["psnr"] for v in selected])),
                "ssim": float(np.mean([v["ssim"] for v in selected])),
            }

    category_avg_psnr = float(np.mean([v["psnr"] for v in results.values()]))
    category_avg_ssim = float(np.mean([v["ssim"] for v in results.values()]))
    print("-" * 78)
    for level_name in ["single", "double", "triple"]:
        if level_name in grouped:
            v = grouped[level_name]
            print(f"{level_name.title() + ' Average':18s} PSNR={v['psnr']:.4f} SSIM={v['ssim']:.4f}")
    print(f"Overall Average    PSNR={category_avg_psnr:.4f} SSIM={category_avg_ssim:.4f}")
    print(f"Image Average      PSNR={global_psnr.avg:.4f} SSIM={global_ssim.avg:.4f}")
    print("=" * 78)

    out_dir = os.path.join(args.output_dir, args.subset)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "categories": results,
                "group_average": grouped,
                "overall_average": {"psnr": category_avg_psnr, "ssim": category_avg_ssim},
                "image_average": {"psnr": global_psnr.avg, "ssim": global_ssim.avg},
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
