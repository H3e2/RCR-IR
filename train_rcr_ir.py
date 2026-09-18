from __future__ import annotations

import json
import math
import os
from datetime import timedelta
from typing import List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data_processing_rcr_ir import AIOTrainDataset, CDD11, IRBenchmarks
from net.rcr_ir import RCRIR, RCRIRConfig
from options_rcr_ir import train_options
from rcr_ir_utils import (
    AverageMeter,
    CharbonnierLoss,
    GradientLoss,
    ModelEMA,
    ReliabilityLoss,
    autocast_context,
    compute_psnr_ssim,
    configure_runtime,
    cosine_lr,
    count_parameters,
    extract_state_dict,
    make_run_dir,
    match_spatial,
    pad_to_multiple,
    seed_everything,
    unpad_tensor,
)


def dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if dist_ready() else 0


def world_size() -> int:
    return dist.get_world_size() if dist_ready() else 1


def is_main() -> bool:
    return rank() == 0


def barrier() -> None:
    if dist_ready():
        dist.barrier()


def init_distributed(opt) -> torch.device:
    local_rank = int(os.environ.get("LOCAL_RANK", opt.local_rank))
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    opt.local_rank = local_rank

    if ws > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requires CUDA.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend=opt.dist_backend,
            init_method="env://",
            timeout=timedelta(minutes=60),
        )
        return torch.device(f"cuda:{local_rank}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def collate_batch(batch):
    metas = [item[0] for item in batch]
    degraded = torch.stack([item[1] for item in batch], dim=0)
    clean = torch.stack([item[2] for item in batch], dim=0)
    return metas, degraded, clean


def build_model(opt) -> RCRIR:
    cfg = RCRIRConfig(
        base_dim=opt.base_dim,
        dim_multipliers=tuple(opt.dim_multipliers),
        enc_blocks=tuple(opt.enc_blocks),
        dec_blocks=tuple(opt.dec_blocks),
        z_dim=opt.z_dim,
        assb_blocks=opt.assb_blocks,
        attention_heads=opt.attention_heads,
        ffn_expansion=opt.ffn_expansion,
    )
    return RCRIR(cfg)


def build_train_dataset(opt):
    if opt.trainset.startswith("CDD11_"):
        subset = opt.trainset.split("_", 1)[1]
        return CDD11(opt, split="train", subset=subset)
    return AIOTrainDataset(opt)


def build_train_loader(opt, dataset):
    sampler = None
    if world_size() > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size(),
            rank=rank(),
            shuffle=True,
            drop_last=True,
        )

    kwargs = dict(
        batch_size=opt.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    if opt.num_workers > 0:
        kwargs["persistent_workers"] = bool(opt.persistent_workers)
        kwargs["prefetch_factor"] = int(opt.prefetch_factor)
    return DataLoader(dataset, **kwargs)


def single_benchmark_dataset(opt, name: str):
    class Args:
        pass

    args = Args()
    args.data_file_dir = opt.data_file_dir
    args.de_type = opt.de_type
    args.benchmarks = [name]
    return IRBenchmarks(args)


@torch.inference_mode()
def evaluate_benchmark(model, dataset, device, precision) -> Tuple[float, float, int]:
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_batch,
    )
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    model.eval()

    for _, degraded, clean in loader:
        degraded = degraded.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)
        padded, pads = pad_to_multiple(degraded, 16)
        with autocast_context(device, precision):
            restored = model(padded)
        restored = unpad_tensor(restored, pads).clamp(0, 1)
        restored, clean = match_spatial(restored, clean)
        psnr, ssim, n = compute_psnr_ssim(restored, clean)
        psnr_meter.update(psnr, n)
        ssim_meter.update(ssim, n)

    return psnr_meter.avg, ssim_meter.avg, psnr_meter.count


def validate(model, opt, device) -> Tuple[float, float, dict]:
    results = {}
    psnr_values: List[float] = []
    ssim_values: List[float] = []

    for name in opt.benchmarks:
        if name.startswith("denoise_"):
            np.random.seed(0)
        dataset = single_benchmark_dataset(opt, name)
        psnr, ssim, samples = evaluate_benchmark(model, dataset, device, "fp32")
        results[name] = {"psnr": psnr, "ssim": ssim, "samples": samples}
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        print(f"[val] {name}: PSNR={psnr:.4f} SSIM={ssim:.4f} N={samples}")

    avg_psnr = float(np.mean(psnr_values))
    avg_ssim = float(np.mean(ssim_values))
    print(f"[val] Average: PSNR={avg_psnr:.4f} SSIM={avg_ssim:.4f}")
    return avg_psnr, avg_ssim, results


def save_checkpoint(path, model, ema, optimizer, epoch, global_step, best_psnr, opt) -> None:
    if not is_main():
        return
    payload = {
        "model_state": unwrap(model).state_dict(),
        "ema_state": ema.module.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_avg_psnr": float(best_psnr),
        "options": vars(opt),
    }
    torch.save(payload, path)


def load_training_state(model, ema, optimizer, opt):
    start_epoch = 0
    global_step = 0
    best_psnr = -float("inf")

    if opt.resume_ckpt:
        ckpt = torch.load(opt.resume_ckpt, map_location="cpu")
        unwrap(model).load_state_dict(extract_state_dict(ckpt, "raw"), strict=True)
        if isinstance(ckpt, dict) and "ema_state" in ckpt:
            ema.module.load_state_dict(ckpt["ema_state"], strict=True)
        else:
            ema.module.load_state_dict(unwrap(model).state_dict(), strict=True)
        if isinstance(ckpt, dict) and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_psnr = float(ckpt.get("best_avg_psnr", best_psnr))
        print(f"Resumed from {opt.resume_ckpt} at epoch {start_epoch + 1}")
        return start_epoch, global_step, best_psnr

    if opt.pretrained_ckpt:
        ckpt = torch.load(opt.pretrained_ckpt, map_location="cpu")
        state = extract_state_dict(ckpt, opt.pretrained_state)
        unwrap(model).load_state_dict(state, strict=True)
        ema.module.load_state_dict(unwrap(model).state_dict(), strict=True)
        print(f"Loaded pretrained weights from {opt.pretrained_ckpt}")

    return start_epoch, global_step, best_psnr


def main() -> None:
    opt = train_options()
    device = init_distributed(opt)
    configure_runtime(opt.allow_tf32, opt.cudnn_benchmark)
    seed_everything(opt.seed + rank())

    if is_main():
        run_dir = make_run_dir(opt.log_root, opt.run_name)
        with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(opt), f, indent=2, ensure_ascii=False)
    else:
        run_dir = ""

    if dist_ready():
        box = [run_dir]
        dist.broadcast_object_list(box, src=0)
        run_dir = box[0]

    dataset = build_train_dataset(opt)
    loader = build_train_loader(opt, dataset)

    base_model = build_model(opt).to(device)
    ema = ModelEMA(base_model, decay=opt.ema_decay)
    model = base_model
    if world_size() > 1:
        model = DDP(base_model, device_ids=[opt.local_rank], output_device=opt.local_rank)

    if is_main():
        n_params = count_parameters(unwrap(model))
        print(f"Model parameters: {n_params:,} ({n_params / 1e6:.3f} M)")
        print(f"Training samples: {len(dataset):,}")
        print(f"Steps per epoch: {len(loader):,}")
        print(f"Run directory: {run_dir}")

    rec_loss = CharbonnierLoss()
    grad_loss = GradientLoss()
    rel_loss = ReliabilityLoss(alpha=opt.rel_alpha, beta=opt.rel_beta)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=opt.lr,
        weight_decay=opt.weight_decay,
    )

    start_epoch, global_step, best_psnr = load_training_state(
        model, ema, optimizer, opt
    )

    total_optimizer_steps = max(
        1,
        opt.epochs * math.ceil(len(loader) / max(opt.accum_grad, 1)),
    )
    optimizer_step = global_step // max(opt.accum_grad, 1)
    use_scaler = device.type == "cuda" and opt.precision == "fp16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    for epoch in range(start_epoch, opt.epochs):
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)

        model.train()
        loss_meter = AverageMeter()
        rec_meter = AverageMeter()
        grad_meter = AverageMeter()
        rel_meter = AverageMeter()
        optimizer.zero_grad(set_to_none=True)

        for step, (_, degraded, clean) in enumerate(loader, start=1):
            degraded = degraded.to(device, non_blocking=True)
            clean = clean.to(device, non_blocking=True)
            padded_deg, pads = pad_to_multiple(degraded, 16)
            padded_clean, _ = pad_to_multiple(clean, 16)

            with autocast_context(device, opt.precision):
                pred_padded, reliability = model(
                    padded_deg,
                    return_reliability=True,
                )
                pred = unpad_tensor(pred_padded, pads)
                pred, clean_match = match_spatial(pred, clean)
                rec = rec_loss(pred, clean_match)
                grad = grad_loss(pred, clean_match)
                rel = rel_loss(reliability, padded_deg, padded_clean)
                loss = rec + opt.grad_loss_weight * grad + opt.rel_loss_weight * rel
                scaled_loss = loss / max(opt.accum_grad, 1)

            if use_scaler:
                scaler.scale(scaled_loss).backward()
            else:
                scaled_loss.backward()

            if step % opt.accum_grad == 0 or step == len(loader):
                if opt.grad_clip > 0:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)

                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                optimizer_step += 1
                lr_scale = cosine_lr(optimizer_step, total_optimizer_steps, opt.warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = opt.lr * lr_scale
                ema.update(unwrap(model))

            global_step += 1
            bs = degraded.shape[0]
            loss_meter.update(loss.item(), bs)
            rec_meter.update(rec.item(), bs)
            grad_meter.update(grad.item(), bs)
            rel_meter.update(rel.item(), bs)

            if is_main() and step % opt.print_every_steps == 0:
                print(
                    f"[train] epoch={epoch + 1}/{opt.epochs} "
                    f"step={step}/{len(loader)} "
                    f"loss={loss_meter.avg:.4f} "
                    f"rec={rec_meter.avg:.4f} "
                    f"grad={grad_meter.avg:.4f} "
                    f"rel={rel_meter.avg:.4f} "
                    f"lr={optimizer.param_groups[0]['lr']:.6e}"
                )

        barrier()

        if is_main() and (
            (epoch + 1) % opt.val_every_n_epochs == 0
            or (epoch + 1) == opt.epochs
        ):
            avg_psnr, _, metrics = validate(ema.module, opt, device)
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                save_checkpoint(
                    os.path.join(run_dir, "checkpoints", "best.ckpt"),
                    model, ema, optimizer, epoch, global_step, best_psnr, opt,
                )
                with open(os.path.join(run_dir, "best_metrics.json"), "w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                print(f"Saved new best checkpoint: PSNR={best_psnr:.4f}")

        if is_main():
            save_checkpoint(
                os.path.join(run_dir, "checkpoints", "last.ckpt"),
                model, ema, optimizer, epoch, global_step, best_psnr, opt,
            )

        barrier()

    if dist_ready():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
