from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from test_utils import build_ir_dataset, build_model, evaluate_dataset, print_summary


def parse_args():
    p = argparse.ArgumentParser(description="RCR-IR five-task evaluation")
    p.add_argument("--ckpt_path", type=str, required=True)
    p.add_argument("--data_file_dir", type=str, default="")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--state", choices=["auto", "ema", "raw"], default="auto")
    p.add_argument("--output_dir", type=str, default="results_rcr_ir/5task")
    p.add_argument("--save_results", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = build_model(args.ckpt_path, device, state=args.state)

    benchmarks = ["lolv1", "gopro", "derain", "dehaze", "denoise_25"]
    results = {}

    for name in benchmarks:
        if name == "denoise_25":
            np.random.seed(0)
        dataset = build_ir_dataset(args.data_file_dir, name)
        save_dir = os.path.join(args.output_dir, name) if args.save_results else ""
        psnr, ssim, samples = evaluate_dataset(model, dataset, device, save_dir)
        results[name] = {"psnr": psnr, "ssim": ssim, "samples": samples}
        print(f"[{name}] PSNR={psnr:.4f} SSIM={ssim:.4f} N={samples}")

    avg_psnr, avg_ssim = print_summary(results)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"benchmarks": results, "average": {"psnr": avg_psnr, "ssim": avg_ssim}},
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
