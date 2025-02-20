import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 加载模型
model = torch.load('/sdc1/songcl/mono3D/Mono3DVG/outputs/mono3dvg/checkpoint_best.pth')

# 计算参数量
param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'模型参数量: {param_count}')