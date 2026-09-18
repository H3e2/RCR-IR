# RCR-IR

Official PyTorch implementation of **RCR-IR: Preserve or Correct: Reliability-Calibrated Residual Routing for Blind All-in-One Image Restoration**.

RCR-IR formulates encoder-to-decoder skip reuse as a **preserve-or-correct** routing problem: reliable responses are preserved through an identity path, while only the uncertain complement is adaptively corrected.

## Model Architecture

![model](/figures/model.png)

## Environment

```bash
pip install -r requirements.txt
```

## Dataset

Please prepare the corresponding datasets and specify the dataset root with:

```bash
--data_file_dir /path/to/dataset
```

## Training

### Single GPU

```bash
python train_rcr_ir.py \
  --data_file_dir /path/to/dataset \
  --run_name rcr_ir
```

### Multi-GPU

For example, training with four GPUs:

```bash
torchrun --standalone --nproc_per_node=4 train_rcr_ir.py \
  --data_file_dir /path/to/dataset \
  --batch_size 8 \
  --run_name rcr_ir
```

## Evaluation

### Three-Degradation Protocol

This protocol evaluates SOTS Outdoor, Rain100L, and CBSD68 with $\sigma=15,25,50$.

```bash
python test_3task.py \
  --ckpt_path /path/to/model.ckpt \
  --data_file_dir /path/to/dataset \
  --device cuda:0
```

### Five-Degradation Protocol

This protocol evaluates SOTS Outdoor, Rain100L, CBSD68 with $\sigma=25$, GoPro, and LOLv1.

```bash
python test_5task.py \
  --ckpt_path /path/to/model.ckpt \
  --data_file_dir /path/to/dataset \
  --device cuda:0
```

### CDD-11 Protocol

Evaluate all 11 degradation categories:

```bash
python test_cdd11.py \
  --ckpt_path /path/to/model.ckpt \
  --data_file_dir /path/to/dataset \
  --subset all \
  --device cuda:0
```

Evaluate single, double, or triple degradation subsets:

```bash
python test_cdd11.py \
  --ckpt_path /path/to/model.ckpt \
  --data_file_dir /path/to/dataset \
  --subset single \
  --device cuda:0
python test_cdd11.py \
  --ckpt_path /path/to/model.ckpt \
  --data_file_dir /path/to/dataset \
  --subset double \
  --device cuda:0
python test_cdd11.py \
  --ckpt_path /path/to/model.ckpt \
  --data_file_dir /path/to/dataset \
  --subset triple \
  --device cuda:0
```

For full training checkpoints, EMA weights can be selected with:

```bash
--state ema
```

## Pre-trained Models

Pre-trained models will be released here:

| Protocol           | Checkpoint                                                   |
| ------------------ | ------------------------------------------------------------ |
| Three degradations | [Download](https://drive.google.com/file/d/1K0GczFwB3ql7tVZ-ZeI1BU0jflUSl4K1/view?usp=sharing) |
| Five degradations  | [Download](https://drive.google.com/file/d/1AtP-MqCb1afqkDSYEpX1Y7ksc5xxbP5U/view?usp=sharing) |
| CDD-11             | [Download](https://drive.google.com/file/d/10FrUn2Je4aFioff7urnQxidkGEZnvHng/view?usp=drive_link) |
