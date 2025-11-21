# MiDaS LoRA Fine-tuning

Fine-tune MiDaS depth estimation models using LoRA (Low-Rank Adaptation).

## Quick Start

### 1. Train

```bash
bash finetune/run_train.sh
```

Or with custom parameters:

```bash
python finetune/train_lora.py \
    --model_path weights/dpt_large_384.pt \
    --model_type dpt_large_384 \
    --train_csv input/nyu2_train.csv \
    --val_split 0.1 \
    --batch_size 8 \
    --epochs 20 \
    --lora_lr 1e-4 \
    --lora_rank 8 \
    --lora_alpha 16 \
    --output_dir finetune/checkpoints
```

### 2. Inference

```bash
bash finetune/run_inference.sh
```

Or manually:

```bash
python run.py \
    -i input/nyu2_test_colors \
    -o output/lora_finetuned \
    -m weights/dpt_large_384.pt \
    -t dpt_large_384 \
    --lora_weights finetune/checkpoints/best_lora.pth \
    --lora_rank 8 \
    --lora_alpha 16 \
    --model_name lora_finetuned
```

### 3. Evaluate

```bash
cd eval && python deep_evaluate.py
```

This evaluates all models in `output/` and shows comparison metrics.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model_path` | `weights/dpt_large_384.pt` | Pretrained model path |
| `--model_type` | `dpt_large_384` | Model type |
| `--lora_rank` | 8 | LoRA rank (smaller = fewer parameters) |
| `--lora_alpha` | 16 | LoRA scaling factor |
| `--lora_lr` | 1e-4 | Learning rate |
| `--batch_size` | 8 | Batch size |
| `--epochs` | 20 | Training epochs |
| `--val_split` | 0.1 | Validation split ratio |

## Dataset Format

CSV file (no header):
```
image_path,depth_path
input/nyu2_train/1.jpg,input/nyu2_train/1.png
input/nyu2_train/2.jpg,input/nyu2_train/2.png
```

## Evaluation Metrics

| Metric | Description | Better |
|--------|-------------|--------|
| d1, d2, d3 | Threshold accuracy | Higher |
| abs_rel | Absolute relative error | Lower |
| rmse | Root mean squared error | Lower |
| silog | Scale-invariant log error | Lower |

## Notes

- `lora_rank` and `lora_alpha` must match between training and inference
- Checkpoints saved to `finetune/checkpoints/`
- Training logs saved to `finetune/checkpoints/training_*.log`
- Best model saved as `best_lora.pth` based on validation loss

## Troubleshooting

**Out of Memory**: Reduce `--batch_size` or `--lora_rank`

**Poor Performance**: Try lower `--lora_lr` (5e-5) or use earlier checkpoint (epoch 5 or 10)

## File Structure

```
finetune/
├── train_lora.py      # Training script
├── dataset.py         # Dataset loader
├── loss.py            # Loss functions
├── lora.py            # LoRA implementation
├── run_train.sh       # Training launcher
└── run_inference.sh   # Inference launcher
```
