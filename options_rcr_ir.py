from __future__ import annotations

import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    value = str(v).lower()
    if value in {"yes", "true", "t", "y", "1"}:
        return True
    if value in {"no", "false", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RCR-IR training options")

    # Data: default is the five-task all-in-one setting.
    p.add_argument("--data_file_dir", type=str, default="/home/user4/cc/data")
    p.add_argument("--trainset", type=str, default="standard")
    p.add_argument(
        "--de_type", nargs="+",
        default=[
            "denoise_15", "denoise_25", "denoise_50",
            "dehaze", "derain", "deblur", "synllie",
        ],
    )
    p.add_argument(
        "--benchmarks", nargs="+",
        default=["lolv1", "gopro", "derain", "dehaze", "denoise_25"],
    )
    p.add_argument("--patch_size", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--persistent_workers", type=str2bool, default=True)
    p.add_argument("--prefetch_factor", type=int, default=4)

    # Dataset balancing used by the five-task training setup.
    p.add_argument("--moce_repeat_synllie", type=int, default=148)
    p.add_argument("--moce_repeat_deblur", type=int, default=34)
    p.add_argument("--moce_repeat_derain", type=int, default=40)
    p.add_argument("--moce_repeat_dehaze", type=int, default=1)
    p.add_argument("--moce_repeat_denoise", type=int, default=14)

    # Runtime.
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--allow_tf32", type=str2bool, default=True)
    p.add_argument("--cudnn_benchmark", type=str2bool, default=True)

    # Optimisation.
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=10000)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--accum_grad", type=int, default=1)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--grad_loss_weight", type=float, default=0.10)
    p.add_argument("--rel_loss_weight", type=float, default=0.05)
    p.add_argument("--rel_alpha", type=float, default=2.0)
    p.add_argument("--rel_beta", type=float, default=0.5)

    # Model.
    p.add_argument("--base_dim", type=int, default=64)
    p.add_argument("--dim_multipliers", nargs="+", type=int, default=[1, 2, 3, 4])
    p.add_argument("--enc_blocks", nargs="+", type=int, default=[2, 2, 4, 2])
    p.add_argument("--dec_blocks", nargs="+", type=int, default=[2, 2, 2, 2])
    p.add_argument("--z_dim", type=int, default=32)
    p.add_argument("--assb_blocks", type=int, default=2)
    p.add_argument("--attention_heads", type=int, default=4)
    p.add_argument("--ffn_expansion", type=float, default=2.0)

    # Checkpoints / logging.
    p.add_argument("--log_root", type=str, default="log_rcr_ir")
    p.add_argument("--run_name", type=str, default="rcr_ir_5task")
    p.add_argument("--resume_ckpt", type=str, default="")
    p.add_argument("--pretrained_ckpt", type=str, default="")
    p.add_argument("--pretrained_state", choices=["auto", "ema", "raw"], default="auto")
    p.add_argument("--val_every_n_epochs", type=int, default=5)
    p.add_argument("--print_every_steps", type=int, default=100)

    # Distributed.
    p.add_argument("--dist_backend", type=str, default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument("--local-rank", dest="local_rank", type=int, default=-1)

    return p


def train_options():
    return build_parser().parse_args()
