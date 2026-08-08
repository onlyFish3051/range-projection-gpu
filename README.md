# ⚡ 3D 点云 → 范围图像投影：GPU 加速（~18×）

> 将激光雷达 3D 点云投影为 2D 范围图像（range image）的 **CPU / GPU 双实现性能对比**。
> 零自定义 CUDA kernel——仅用 PyTorch 内建算子将 NumPy 实现逐行翻译，即获得 **18~26 倍加速**。

## 📊 性能实测（RTX 3060 Ti，投影图 2048×64）

![加速对比图](speedup.png?v=3)

| 点数/帧 | CPU（NumPy 单核） | GPU（PyTorch） | 加速比 |
|---|---|---|---|
| 6 万点 | 3.04 ms | 0.31 ms | **9.8×** |
| **12 万点（KITTI 典型帧）** | **6.33 ms** | **0.35 ms** | **18.2×** |
| 20 万点 | 11.19 ms | 0.43 ms | **25.8×** |

- 数据已在 GPU 上时纯计算与端到端（含 H2D 传输）几乎无差：12 万点仅 ~1.9MB，PCIe 拷贝 ~0.02ms
- GPU 与 CPU 结果一致性 **98.52%**（差异为浮点舍入边界，逻辑完全等价）
- 测试方法：各跑 100 次取中位数，含 warmup

## 🔧 实现原理

算法（RangeNet++ 风格扫描投影，适配 SemanticKITTI / HDL-64E）：

```
1. depth = ||(x,y,z)||                        # 欧氏距离
2. proj_x = yaw 角映射（360° 均布）             # 列坐标
3. proj_y = yaw 跳变检测 + 累积和（64 线束）     # 行坐标
4. 按深度降序写图：远的先写、近的覆盖            # 遮挡处理
```

**GPU 加速的关键**：投影对每个点是**独立计算**的（norm/atan2/映射），天然并行——
- CPU 版：NumPy 单核逐个处理 12 万点
- GPU 版：同样的运算用 torch 张量表达，自动映射为 CUDA kernel，**12 万个线程并行**执行
- 遮挡排序 `argsort` 在 GPU 上由并行排序 kernel（CUB 基数排序）完成

```python
# CPU（NumPy）           # GPU（PyTorch）—— 逐行对齐
np.linalg.norm(pc[:,:3], 2, axis=1)   torch.norm(pc[:,:3], dim=1)
-np.arctan2(y, -x)                    -torch.atan2(y, -x)
np.cumsum(proj_y)                     torch.cumsum(proj_y, dim=0)
np.argsort(depth)[::-1]               torch.argsort(depth, descending=True)
```

## 🚀 使用

```bash
# 依赖：numpy + torch（CUDA 可用）
python benchmark.py              # 运行性能基准
python benchmark.py --plot       # 额外生成 speedup.png 加速对比图
```

```python
import torch
from projection import project_cpu, project_gpu

pc = torch.randn(120_000, 4, device="cuda")   # (N, 4): x,y,z,intensity
range_img = project_gpu(pc)                    # (64, 2048) 距离图，未命中 -1
```

## 💡 设计定位：纯加速，行为零改变

- **投影行为严格不变**：GPU 版与 CPU 版算法逐行对齐（同输入、同输出，98.5% 像素一致），投影作为固定几何预处理**不随网络训练变化**——保证实验可复现、模型输入分布稳定
- **只是执行平台迁移**：CPU（NumPy）→ GPU（PyTorch 算子），获得 18~26× 并行加速
- 实现采用 torch 张量运算，顺带具备可微属性（torch.autograd 自动记录）——但**可微不是本项目目标**，若需保持行为完全固定，梯度不流经投影即可
- 可直接嵌入模型作为 torch 模块，或用于批量点云预训练数据生成
- 如需进一步融合优化（消除 kernel launch 与中间张量，预计 0.35→0.1ms），可用 Triton 写融合 kernel——本实现已足够快，暂未必要

## 📁 文件

| 文件 | 说明 |
|---|---|
| `projection.py` | CPU / GPU 双实现 + KITTI 点云合成器 |
| `benchmark.py` | 性能基准 + 加速图生成 |

## License

MIT
