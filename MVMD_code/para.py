import torch
from thop import profile
from torchsummary import summary
from networks.MVMD_network import MVMD_Network

# 初始化模型并移动到 GPU 上
model = MVMD_Network().cuda()

# 打印模型参数数量
print("模型参数数量：")
summary(model, input_size=[(3, 416, 416), (3, 416, 416), (3, 416, 416)])

# 输入数据样例并移动到 GPU 上
input_rgb1 = torch.randn(1, 3, 416, 416).cuda()
input_rgb2 = torch.randn(1, 3, 416, 416).cuda()
input_rgb3 = torch.randn(1, 3, 416, 416).cuda()

# 计算 FLOPs 和参数数量
flops, params = profile(model, inputs=(input_rgb1, input_rgb2, input_rgb3))

# 打印结果
print(f"FLOPs: {flops}")
print(f"参数数量: {params}")

# 计算模型内存使用量
total_params = sum(p.numel() for p in model.parameters())
total_memory = total_params * 4 / (1024 ** 2)  # 假设每个参数占用 4 字节，转换为 MB

print(f"模型内存使用量: {total_memory:.2f} MB")

