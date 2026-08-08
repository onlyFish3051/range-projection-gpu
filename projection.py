"""3D 点云 → 2D 范围图像（range image）投影：CPU（NumPy）与 GPU（PyTorch）双实现。

算法（RangeNet++ 风格扫描投影，适配 SemanticKITTI / HDL-64E）：
  1. 每个点计算欧氏距离 depth
  2. yaw 角映射到列坐标 proj_x（360° 均布）
  3. 检测 yaw 跳变识别线束切换，累积和得到行坐标 proj_y（对应 64 线束）
  4. 按深度降序写图：远的先写、近的覆盖 → 正确解决遮挡

CPU 版：NumPy 逐点向量化（单核执行）
GPU 版：同样的运算用 PyTorch 张量表达，自动映射为 CUDA kernel 并行执行
        ——零自定义算子，全部使用 torch 内建算子（norm/atan2/cumsum/argsort）
"""
import numpy as np
import torch


def make_kitti_frame(n_points: int, seed: int = 42) -> np.ndarray:
    """合成一帧 HDL-64 点云：azimuth 均布 360°，elevation 按 64 线束分布，带 intensity。"""
    rng = np.random.default_rng(seed)
    az = np.linspace(-np.pi, np.pi, n_points)
    elev = np.linspace(np.radians(3.0), np.radians(-25.0), n_points)
    dist = 3.0 + rng.random(n_points) * 60.0  # 3~63m
    x = dist * np.cos(elev) * np.cos(az)
    y = dist * np.cos(elev) * np.sin(az)
    z = dist * np.sin(elev)
    intensity = rng.random(n_points)
    return np.stack([x, y, z, intensity], axis=1).astype(np.float32)


def project_cpu(pointcloud: np.ndarray, proj_w: int = 2048, proj_h: int = 64) -> np.ndarray:
    """CPU（NumPy）投影，返回 (proj_h, proj_w) 的距离图，未命中像素为 -1。"""
    depth = np.linalg.norm(pointcloud[:, :3], 2, axis=1)
    yaw = -np.arctan2(pointcloud[:, 1], -pointcloud[:, 0])
    proj_x = 0.5 * (yaw / np.pi + 1.0)
    # 线束切换检测：proj_x 从 ~1.0 跳回 ~0.0 处
    new_raw = np.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8))[0] + 1
    proj_y = np.zeros_like(proj_x)
    proj_y[new_raw] = 1
    proj_y = np.cumsum(proj_y)
    proj_x = proj_x * proj_w - 0.001
    px = np.maximum(np.minimum(proj_w - 1, np.floor(proj_x)), 0).astype(np.int32)
    py = np.maximum(np.minimum(proj_h - 1, np.floor(proj_y)), 0).astype(np.int32)
    # 遮挡：按深度降序，远的先写、近的覆盖
    order = np.argsort(depth)[::-1]
    proj_range = np.full((proj_h, proj_w), -1, dtype=np.float32)
    proj_range[py[order], px[order]] = depth[order]
    return proj_range


def project_gpu(pointcloud: torch.Tensor, proj_w: int = 2048, proj_h: int = 64) -> torch.Tensor:
    """GPU（PyTorch）投影：与 CPU 版逐行对齐，返回距离图（在 GPU 上）。"""
    x, y, z = pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2]
    depth = torch.norm(pointcloud[:, :3], dim=1)
    yaw = -torch.atan2(y, -x)
    proj_x = 0.5 * (yaw / np.pi + 1.0)
    new_raw = torch.nonzero((proj_x[1:] < 0.2) * (proj_x[:-1] > 0.8)).squeeze(1) + 1
    proj_y = torch.zeros_like(proj_x)
    proj_y[new_raw] = 1.0
    proj_y = torch.cumsum(proj_y, dim=0)
    proj_x = proj_x * proj_w - 0.001
    px = torch.clamp(proj_x.floor(), 0, proj_w - 1).long()
    py = torch.clamp(proj_y.floor(), 0, proj_h - 1).long()
    order = torch.argsort(depth, descending=True)
    flat_idx = py[order] * proj_w + px[order]
    proj_range = torch.full((proj_h, proj_w), -1.0, device=pointcloud.device)
    proj_range.view(-1)[flat_idx] = depth[order]
    return proj_range
