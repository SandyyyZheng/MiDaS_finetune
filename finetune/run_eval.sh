#!/bin/bash
# MiDaS 完整评估流程: 推理 + 评估

echo "================================"
echo "MiDaS LoRA 完整评估流程"
echo "================================"
echo ""

# 步骤 1: 生成预测
echo "步骤 1: 生成预测结果"
echo "--------------------------------"
bash finetune/run_inference.sh

if [ $? -ne 0 ]; then
    echo "推理失败，退出"
    exit 1
fi

echo ""
echo "步骤 2: 评估性能指标"
echo "--------------------------------"

# 步骤 2: 评估
cd eval && python deep_evaluate.py

echo ""
echo "================================"
echo "完整评估流程结束"
echo "================================"
