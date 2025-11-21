#!/bin/bash
# MiDaS LoRA 推理脚本 - 使用 run.py 生成预测结果

# 设置路径
MODEL_PATH="weights/dpt_large_384.pt"
MODEL_TYPE="dpt_large_384"
LORA_WEIGHTS="finetune/checkpoints/best_lora.pth"

# 输入输出
INPUT_DIR="input/nyu2_test_colors"  # 测试图像目录
OUTPUT_DIR="output/lora_finetuned"

# LoRA 参数 (必须与训练时一致)
LORA_RANK=8
LORA_ALPHA=16

echo "================================"
echo "MiDaS LoRA 推理 - 生成预测"
echo "================================"
echo "模型: $MODEL_TYPE"
echo "LoRA 权重: $LORA_WEIGHTS"
echo "输入目录: $INPUT_DIR"
echo "输出目录: $OUTPUT_DIR"
echo "================================"
echo ""

# 检查 LoRA 权重是否存在
if [ ! -f "$LORA_WEIGHTS" ]; then
    echo "错误: LoRA 权重文件不存在: $LORA_WEIGHTS"
    echo "请先运行训练脚本: ./finetune/run_train.sh"
    exit 1
fi

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "错误: 输入目录不存在: $INPUT_DIR"
    echo "请确保测试图像在该目录下"
    exit 1
fi

# 使用 run.py 进行推理
python run.py \
    -i $INPUT_DIR \
    -o $OUTPUT_DIR \
    -m $MODEL_PATH \
    -t $MODEL_TYPE \
    --lora_weights $LORA_WEIGHTS \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --model_name lora_finetuned

echo ""
echo "预测完成!"
echo "结果保存在: $OUTPUT_DIR"
echo ""
echo "下一步: 评估模型性能"
echo "  cd eval && python deep_evaluate.py"
