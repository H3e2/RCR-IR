from __future__ import annotations

import glob
import os
import random
from collections import defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def crop_img(img: np.ndarray, base: int = 16) -> np.ndarray:
    h, w = img.shape[:2]
    crop_h = h % base
    crop_w = w % base
    return img[
        crop_h // 2:h - crop_h + crop_h // 2,
        crop_w // 2:w - crop_w + crop_w // 2,
    ]


def random_augmentation(*imgs: np.ndarray) -> Tuple[np.ndarray, ...]:
    mode = random.randint(1, 7)
    ops = [
        lambda x: x,
        np.flipud,
        np.rot90,
        lambda x: np.flipud(np.rot90(x)),
        lambda x: np.rot90(x, k=2),
        lambda x: np.flipud(np.rot90(x, k=2)),
        lambda x: np.rot90(x, k=3),
        lambda x: np.flipud(np.rot90(x, k=3)),
    ]
    return tuple(np.ascontiguousarray(ops[mode](img)) for img in imgs)


def _glob_multi_ext(folder: str, extensions: Sequence[str]) -> List[str]:
    out: List[str] = []
    for ext in extensions:
        out.extend(glob.glob(os.path.join(folder, f"*.{ext}")))
        out.extend(glob.glob(os.path.join(folder, f"*.{ext.upper()}")))
    return sorted(set(out))


def _join_rel(root: str, rel: str) -> str:
    return os.path.join(root, *rel.split("/"))


def _first_nonempty_glob(
    root: str,
    rel_folders: Sequence[str],
    extensions: Sequence[str],
) -> Tuple[List[str], str]:
    last_folder = _join_rel(root, rel_folders[-1]) if rel_folders else root
    for rel in rel_folders:
        folder = _join_rel(root, rel)
        files = _glob_multi_ext(folder, extensions)
        if files:
            return files, folder
        last_folder = folder
    return [], last_folder


def _collect_glob_candidates(
    root: str,
    rel_folders: Sequence[str],
    extensions: Sequence[str],
) -> Tuple[List[str], str]:
    files: List[str] = []
    used: List[str] = []
    for rel in rel_folders:
        folder = _join_rel(root, rel)
        found = _glob_multi_ext(folder, extensions)
        if found:
            files.extend(found)
            used.append(folder)
    source = ", ".join(used) if used else (
        _join_rel(root, rel_folders[0]) if rel_folders else root
    )
    return sorted(set(files)), source


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def np_to_tensor(img: np.ndarray) -> torch.Tensor:
    arr = np.ascontiguousarray(img)
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div(255.0)


def _pad_reflect_to_min_size(img: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    pad_h = max(0, min_h - h)
    pad_w = max(0, min_w - w)
    if pad_h == 0 and pad_w == 0:
        return img
    mode = "reflect" if h >= 2 and w >= 2 else "edge"
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode=mode)


def random_crop_pair(lr: np.ndarray, hr: np.ndarray, patch_size: int):
    h = min(lr.shape[0], hr.shape[0])
    w = min(lr.shape[1], hr.shape[1])
    lr = _pad_reflect_to_min_size(lr[:h, :w], patch_size, patch_size)
    hr = _pad_reflect_to_min_size(hr[:h, :w], patch_size, patch_size)
    h, w = lr.shape[:2]
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return (
        lr[top:top + patch_size, left:left + patch_size],
        hr[top:top + patch_size, left:left + patch_size],
    )


def _sorted_pairs_from_same_names(lr_paths, hr_paths):
    hr_map = {os.path.basename(p): p for p in sorted(hr_paths)}
    return [
        (lr, hr_map[os.path.basename(lr)])
        for lr in sorted(lr_paths)
        if os.path.basename(lr) in hr_map
    ]


class OnlineSyntheticDegrader:
    @staticmethod
    def _gaussian_noise(clean, sigma):
        noise = np.random.randn(*clean.shape)
        return np.clip(clean + noise * sigma, 0, 255).astype(np.uint8)

    def single_degrade(self, clean, name):
        if name == "denoise_15":
            return self._gaussian_noise(clean, 15.0)
        if name == "denoise_25":
            return self._gaussian_noise(clean, 25.0)
        if name == "denoise_50":
            return self._gaussian_noise(clean, 50.0)
        raise NotImplementedError(f"Online degradation '{name}' not implemented.")


class CDD11(Dataset):
    def __init__(self, args, split="train", subset="all"):
        self.args = args
        self.split = split
        self.subset = subset
        self.patch_size = args.patch_size if split == "train" else 64
        self.items: List[Dict] = []
        self._init_items()

    def _root(self) -> str:
        for name in ["cdd11", "CDD11"]:
            root = os.path.join(self.args.data_file_dir, name, self.split)
            if os.path.isdir(root):
                return root
        return os.path.join(self.args.data_file_dir, "cdd11", self.split)

    def _init_items(self):
        root = self._root()
        clean_dir = None
        for name in ["clear", "HR"]:
            candidate = os.path.join(root, name)
            if os.path.isdir(candidate):
                clean_dir = candidate
                break
        if clean_dir is None:
            return
        clean_names = {
            os.path.basename(p): p
            for p in sorted(glob.glob(os.path.join(clean_dir, "*.png")))
        }
        for folder in sorted(glob.glob(os.path.join(root, "*/"))):
            fn = os.path.basename(folder.rstrip("/"))
            count = fn.count("_") + 1
            if fn in {"clear", "HR"}:
                continue
            if self.subset == "single" and count != 1:
                continue
            if self.subset == "double" and count != 2:
                continue
            if self.subset == "triple" and count != 3:
                continue
            if self.subset not in ["single", "double", "triple", "all"] and fn != self.subset:
                continue
            for dp in sorted(glob.glob(os.path.join(folder, "*.png"))):
                name = os.path.basename(dp)
                if name in clean_names:
                    self.items.append({
                        "clean": clean_names[name],
                        "degraded": dp,
                        "de_type": fn,
                    })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        lr = _load_rgb(item["degraded"])
        hr = _load_rgb(item["clean"])
        if self.split == "train":
            lr, hr = random_crop_pair(lr, hr, self.patch_size)
            lr, hr = random_augmentation(lr, hr)
        return [item["degraded"], item["de_type"]], np_to_tensor(lr), np_to_tensor(hr)


class AIOTrainDataset(Dataset):
    """MoCE-IR style merged training dataset with fixed per-task repeats."""

    def __init__(self, args):
        self.args = args
        self.patch_size = int(args.patch_size)
        self.de_type = list(args.de_type)
        self.synthetic = OnlineSyntheticDegrader()
        self.items: List[Dict] = []
        self.task_counts: Dict[str, int] = defaultdict(int)
        self._init_items()
        self._print_summary()

    def _repeat_count(self, de_name: str) -> int:
        if de_name.startswith("denoise"):
            return max(1, int(getattr(self.args, "moce_repeat_denoise", 3)))
        return max(1, int(getattr(self.args, f"moce_repeat_{de_name}", 1)))

    def _add_item(self, lr, hr, de_name):
        self.items.append({"lr": lr, "hr": hr, "de_type": de_name})
        self.task_counts[de_name] += 1

    def _append_pairs(self, pairs, de_name, repeat: int):
        for _ in range(repeat):
            for lr, hr in pairs:
                self._add_item(lr, hr, de_name)
        print(f"Registered samples : {len(pairs)}")
        if repeat > 1:
            print(f"Repeated Dataset length : {len(pairs) * repeat}")

    def _append_clean_list(self, clean_paths, de_name, repeat: int):
        for _ in range(repeat):
            for path in clean_paths:
                self._add_item(path, path, de_name)
        print(f"Registered samples : {len(clean_paths)}")
        if repeat > 1:
            print(f"Repeated Dataset length : {len(clean_paths) * repeat}")

    def _resolve_dehaze_hr_train(self, lr_path, hr_dir):
        base = os.path.basename(lr_path)
        name = base.split("_")[0]
        suffix = os.path.splitext(base)[1]
        for ext in [suffix, ".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]:
            candidate = os.path.join(hr_dir, name + ext)
            if os.path.exists(candidate):
                return candidate
        return None

    def _init_synllie(self):
        root = self.args.data_file_dir
        lrs, lr_src = _first_nonempty_glob(
            root,
            ["llie/LOLv1/Train/input", "LOL/train/LR"],
            ["png", "jpg", "jpeg", "bmp"],
        )
        hrs, hr_src = _first_nonempty_glob(
            root,
            ["llie/LOLv1/Train/target", "LOL/train/HR"],
            ["png", "jpg", "jpeg", "bmp"],
        )
        pairs = _sorted_pairs_from_same_names(lrs, hrs)
        print(f"SynLLIE LR source : {lr_src}")
        print(f"SynLLIE HR source : {hr_src}")
        print(f"Total SynLLIE training pairs : {len(pairs)}")
        self._append_pairs(pairs, "synllie", self._repeat_count("synllie"))

    def _init_deblur(self):
        root = self.args.data_file_dir
        lrs, lr_src = _first_nonempty_glob(
            root,
            ["deblurring/GoPro/crop/train/input_crops", "GoPro/train/LR"],
            ["png", "jpg", "jpeg", "bmp"],
        )
        hrs, hr_src = _first_nonempty_glob(
            root,
            ["deblurring/GoPro/crop/train/target_crops", "GoPro/train/HR"],
            ["png", "jpg", "jpeg", "bmp"],
        )
        pairs = _sorted_pairs_from_same_names(lrs, hrs)
        print(f"Deblur LR source : {lr_src}")
        print(f"Deblur HR source : {hr_src}")
        print(f"Total Deblur training pairs : {len(pairs)}")
        self._append_pairs(pairs, "deblur", self._repeat_count("deblur"))

    def _init_derain(self):
        root = self.args.data_file_dir
        lrs, lr_src = _first_nonempty_glob(
            root,
            ["deraining/RainTrainL/rainy", "Rain100L/train/LR"],
            ["png", "jpg", "jpeg", "bmp"],
        )
        hrs, hr_src = _first_nonempty_glob(
            root,
            ["deraining/RainTrainL/gt", "Rain100L/train/HR"],
            ["png", "jpg", "jpeg", "bmp"],
        )
        pairs = _sorted_pairs_from_same_names(lrs, hrs)
        print(f"Derain LR source : {lr_src}")
        print(f"Derain HR source : {hr_src}")
        print(f"Total Derain training pairs : {len(pairs)}")
        self._append_pairs(pairs, "derain", self._repeat_count("derain"))

    def _init_dehaze(self):
        root = self.args.data_file_dir
        lrs, lr_src = _collect_glob_candidates(
            root,
            [
                "dehazing/RESIDE/part1",
                "dehazing/RESIDE/part2",
                "dehazing/RESIDE/part3",
                "dehazing/RESIDE/part4",
            ],
            ["jpg", "jpeg", "png", "bmp"],
        )
        if lrs:
            hr_dir = _join_rel(root, "dehazing/RESIDE/clear")
        else:
            lrs, lr_src = _first_nonempty_glob(
                root,
                ["SOST/train/LR", "SOTS/train/LR"],
                ["jpg", "jpeg", "png", "bmp"],
            )
            hr_dir = os.path.join(os.path.dirname(lr_src), "HR")
        pairs = [
            (lr, hr)
            for lr in sorted(lrs)
            for hr in [self._resolve_dehaze_hr_train(lr, hr_dir)]
            if hr
        ]
        print(f"Dehaze LR source : {lr_src}")
        print(f"Dehaze HR source : {hr_dir}")
        print(f"Total Dehaze training pairs : {len(pairs)}")
        self._append_pairs(pairs, "dehaze", self._repeat_count("dehaze"))

    def _init_denoise(self, de_name):
        root = self.args.data_file_dir
        clean, clean_src = _collect_glob_candidates(
            root,
            ["denoising/WaterlooED", "denoising/BSD400"],
            ["bmp", "jpg", "jpeg", "png"],
        )
        if not clean:
            clean, clean_src = _first_nonempty_glob(
                root,
                ["Denoise/train"],
                ["bmp", "jpg", "jpeg", "png"],
            )
        print(f"Denoise source : {clean_src}")
        print(f"Total Denoise Ids : {len(clean)}")
        self._append_clean_list(clean, de_name, self._repeat_count(de_name))

    def _init_items(self):
        init_map = {
            "synllie": self._init_synllie,
            "deblur": self._init_deblur,
            "derain": self._init_derain,
            "dehaze": self._init_dehaze,
            "denoise_15": lambda: self._init_denoise("denoise_15"),
            "denoise_25": lambda: self._init_denoise("denoise_25"),
            "denoise_50": lambda: self._init_denoise("denoise_50"),
        }
        for name in self.de_type:
            if name not in init_map:
                raise ValueError(f"Unsupported degradation type: {name}")
            init_map[name]()

    def _print_summary(self):
        print("Data loading style : moce_ir")
        print(f"Total merged training samples : {len(self.items)}")
        batch_size = int(getattr(self.args, "batch_size", 1))
        print(
            f"Estimated steps/epoch (drop_last, bs={batch_size}) : "
            f"{len(self.items) // max(batch_size, 1)}"
        )
        for name in self.de_type:
            print(f"Task samples [{name}] : {self.task_counts.get(name, 0)}")

    def set_patch_size(self, patch_size: int) -> None:
        self.patch_size = int(max(16, patch_size))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        de_name = item["de_type"]
        hr = crop_img(_load_rgb(item["hr"]), base=16)
        if de_name.startswith("denoise"):
            hr = _pad_reflect_to_min_size(hr, self.patch_size, self.patch_size)
            _, hr = random_crop_pair(hr, hr, self.patch_size)
            hr = random_augmentation(hr)[0]
            lr = self.synthetic.single_degrade(hr, de_name)
        else:
            lr = crop_img(_load_rgb(item["lr"]), base=16)
            lr, hr = random_crop_pair(lr, hr, self.patch_size)
            lr, hr = random_augmentation(lr, hr)
        return [item["lr"], de_name], np_to_tensor(lr), np_to_tensor(hr)


class IRBenchmarks(Dataset):
    def __init__(self, args):
        self.args = args
        self.benchmarks = list(args.benchmarks)
        self.synthetic = OnlineSyntheticDegrader()
        self.samples: List[Dict] = []
        self._init_items()

    def _resolve_dehaze_hr_test(self, hazy_name):
        folder = os.path.dirname(os.path.dirname(hazy_name))
        name = os.path.basename(hazy_name).split("_")[0]
        for gt_folder in ["HR", "gt"]:
            gt_dir = os.path.join(folder, gt_folder)
            if not os.path.isdir(gt_dir):
                continue
            for ext in [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"]:
                candidate = os.path.join(gt_dir, name + ext)
                if os.path.exists(candidate):
                    return candidate
        return os.path.join(folder, "HR", name + ".png")

    def _add_pairs(self, lr_paths, hr_paths, de_name):
        for lr, hr in _sorted_pairs_from_same_names(lr_paths, hr_paths):
            self.samples.append({"lr": lr, "hr": hr, "de_type": de_name})

    def _init_items(self):
        root = self.args.data_file_dir.rstrip("/")
        if "lolv1" in self.benchmarks:
            lrs, lr_src = _first_nonempty_glob(
                root,
                ["llie/LOLv1/Test/input", "LOL/test/LR"],
                ["png", "jpg", "jpeg", "bmp"],
            )
            hrs, hr_src = _first_nonempty_glob(
                root,
                ["llie/LOLv1/Test/target", "LOL/test/HR"],
                ["png", "jpg", "jpeg", "bmp"],
            )
            print(f"Total LLIE testing pairs : {len(_sorted_pairs_from_same_names(lrs, hrs))} ({lr_src} -> {hr_src})")
            self._add_pairs(lrs, hrs, "synllie")
        if "gopro" in self.benchmarks:
            lrs, lr_src = _first_nonempty_glob(
                root,
                ["deblurring/GoPro/test/input", "GoPro/test/LR"],
                ["png", "jpg", "jpeg", "bmp"],
            )
            hrs, hr_src = _first_nonempty_glob(
                root,
                ["deblurring/GoPro/test/target", "GoPro/test/HR"],
                ["png", "jpg", "jpeg", "bmp"],
            )
            print(f"Total Deblur testing pairs : {len(_sorted_pairs_from_same_names(lrs, hrs))} ({lr_src} -> {hr_src})")
            self._add_pairs(lrs, hrs, "deblur")
        if "derain" in self.benchmarks:
            lrs, lr_src = _first_nonempty_glob(
                root,
                ["deraining/Rain100L/rainy", "Rain100L/test/LR"],
                ["png", "jpg", "jpeg", "bmp"],
            )
            hrs, hr_src = _first_nonempty_glob(
                root,
                ["deraining/Rain100L/gt", "Rain100L/test/HR"],
                ["png", "jpg", "jpeg", "bmp"],
            )
            print(f"Total Derain testing pairs : {len(_sorted_pairs_from_same_names(lrs, hrs))} ({lr_src} -> {hr_src})")
            self._add_pairs(lrs, hrs, "derain")
        if "dehaze" in self.benchmarks:
            lrs, lr_src = _first_nonempty_glob(
                root,
                ["dehazing/SOTS/outdoor/hazy", "SOST/test/LR", "SOTS/test/LR"],
                ["jpg", "jpeg", "png", "bmp"],
            )
            print(f"Total Dehazing testing inputs : {len(lrs)} ({lr_src})")
            for lr in sorted(lrs):
                self.samples.append({
                    "lr": lr,
                    "hr": self._resolve_dehaze_hr_test(lr),
                    "de_type": "dehaze",
                })
        clean_test, clean_src = _first_nonempty_glob(
            root,
            ["denoising/cBSD68/original_png", "Denoise/test/HR"],
            ["png", "jpg", "jpeg", "bmp"],
        )
        for name in ["denoise_15", "denoise_25", "denoise_50"]:
            if name in self.benchmarks:
                print(f"Total Denoise testing pairs ({name}) : {len(clean_test)} ({clean_src})")
                for path in clean_test:
                    self.samples.append({"lr": path, "hr": path, "de_type": name})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        hr = crop_img(_load_rgb(item["hr"]), base=16)
        if item["de_type"].startswith("denoise"):
            lr = self.synthetic.single_degrade(hr, item["de_type"])
        else:
            lr = crop_img(_load_rgb(item["lr"]), base=16)
            h = min(lr.shape[0], hr.shape[0])
            w = min(lr.shape[1], hr.shape[1])
            lr, hr = lr[:h, :w], hr[:h, :w]
        return [item["lr"], item["de_type"]], np_to_tensor(lr), np_to_tensor(hr)

