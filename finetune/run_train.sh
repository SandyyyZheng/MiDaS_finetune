#!/bin/bash
# MiDaS LoRA 微调训练脚本

# 设置路径
MODEL_PATH="weights/dpt_large_384.pt"
MODEL_TYPE="dpt_large_384"
TRAIN_CSV="input/nyu2_train.csv"
VAL_SPLIT=0.1  # 10% of training data for validation
OUTPUT_DIR="finetune/checkpoints"

# Training parameters (following MiDaS paper)
BATCH_SIZE=16
EPOCHS=20
LORA_LR=1e-4  # LoRA parameters (randomly initialized layers)
LR_STEP=10    # Halve learning rate every 10 epochs

# LoRA 参数
LORA_RANK=8
LORA_ALPHA=16

# 系统参数
NUM_WORKERS=4

echo "================================"
echo "MiDaS LoRA Fine-tuning"
echo "================================"
echo "Model: $MODEL_TYPE"
echo "Training CSV: $TRAIN_CSV"
echo "Validation Split: $VAL_SPLIT"
echo "LoRA Rank: $LORA_RANK"
echo "LoRA Alpha: $LORA_ALPHA"
echo "================================"
echo ""

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 运行训练
python finetune/train_lora.py \
    --model_path $MODEL_PATH \
    --model_type $MODEL_TYPE \
    --train_csv $TRAIN_CSV \
    --val_split $VAL_SPLIT \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --lora_lr $LORA_LR \
    --lr_step $LR_STEP \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --num_workers $NUM_WORKERS \
    --output_dir $OUTPUT_DIR \
    --save_interval 5

echo ""
echo "Training completed!"
echo "Checkpoints saved in: $OUTPUT_DIR"
echo "Check log file in: $OUTPUT_DIR/training_*.log"
