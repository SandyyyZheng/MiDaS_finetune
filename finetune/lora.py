"""LoRA (Low-Rank Adaptation) 实现"""
import torch
import torch.nn as nn
import math


class LoRALayer(nn.Module):
    """LoRA层：对线性层进行低秩分解适配"""

    def __init__(self, in_features, out_features, rank=4, alpha=1.0):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            rank: LoRA秩（越小参数越少）
            alpha: 缩放因子
        """
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA矩阵 A 和 B
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """前向传播：返回 LoRA 调整项"""
        # x @ A^T @ B^T
        return (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def inject_lora_to_linear(model, rank=4, alpha=1.0, target_modules=None):
    """
    向模型的线性层注入 LoRA

    Args:
        model: 要修改的模型
        rank: LoRA秩
        alpha: 缩放因子
        target_modules: 目标模块名称列表，如 ['q_proj', 'v_proj']
                       None表示所有线性层
    """
    lora_params = []

    # 获取模型所在设备
    device = next(model.parameters()).device

    for name, module in model.named_modules():
        # 判断是否是目标模块
        if isinstance(module, nn.Linear):
            if target_modules is None or any(target in name for target in target_modules):
                # 获取原始权重
                in_features = module.in_features
                out_features = module.out_features

                # 创建 LoRA 层并移动到模型所在设备
                lora = LoRALayer(in_features, out_features, rank, alpha).to(device)

                # 冻结原始权重
                module.weight.requires_grad = False
                if module.bias is not None:
                    module.bias.requires_grad = False

                # 保存 LoRA 层到模块
                module.lora = lora

                # 修改 forward 方法
                original_forward = module.forward

                def make_lora_forward(orig_forward, lora_layer):
                    def forward(x):
                        # 原始输出 + LoRA调整
                        return orig_forward(x) + lora_layer(x)
                    return forward

                module.forward = make_lora_forward(original_forward, lora)

                # 收集可训练参数
                lora_params.extend(lora.parameters())

    return lora_params


def mark_only_lora_as_trainable(model):
    """标记只有 LoRA 参数可训练"""
    # 冻结所有参数
    for param in model.parameters():
        param.requires_grad = False

    # 解冻 LoRA 参数
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            for param in module.lora.parameters():
                param.requires_grad = True


def get_lora_parameters(model):
    """获取所有 LoRA 参数"""
    lora_params = []
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_params.extend(module.lora.parameters())
    return lora_params


def save_lora_weights(model, save_path):
    """仅保存 LoRA 权重"""
    lora_state_dict = {}
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):
            lora_state_dict[name] = module.lora.state_dict()

    torch.save(lora_state_dict, save_path)
    print(f"LoRA权重已保存到: {save_path}")


def load_lora_weights(model, load_path):
    """加载 LoRA 权重"""
    lora_state_dict = torch.load(load_path)
    for name, module in model.named_modules():
        if name in lora_state_dict and hasattr(module, 'lora'):
            module.lora.load_state_dict(lora_state_dict[name])

    print(f"LoRA权重已从 {load_path} 加载")
