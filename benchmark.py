"""性能基准：CPU（NumPy）vs GPU（PyTorch）投影，并生成加速对比图。

用法：
    python benchmark.py                # 运行基准并输出性能表
    python benchmark.py --plot         # 额外生成 speedup.png 加速对比图
"""
import argparse
import time

import numpy as np
import torch

from projection import make_kitti_frame, project_cpu, project_gpu


def bench_cpu(points: int, iters: int = 100) -> float:
    pc = make_kitti_frame(points)
    project_cpu(pc)  # warmup
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        project_cpu(pc)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return sum(times) / len(times)


def bench_gpu(points: int, iters: int = 100, with_transfer: bool = True) -> float:
    pc_np = make_kitti_frame(points)
    pc = torch.from_numpy(pc_np).cuda()
    project_gpu(pc)  # warmup
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if with_transfer:
            pc = torch.from_numpy(pc_np).cuda()  # 模拟数据加载后的 H2D 传输
        t0 = time.perf_counter()
        project_gpu(pc)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return sum(times) / len(times)


def verify(points: int = 120_000) -> float:
    """验证 GPU 与 CPU 结果一致性。"""
    pc_np = make_kitti_frame(points)
    cpu_out = project_cpu(pc_np)
    gpu_out = project_gpu(torch.from_numpy(pc_np).cuda()).cpu().numpy()
    return float(np.mean(cpu_out == gpu_out))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot", action="store_true", help="生成加速对比图 speedup.png")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    results = []
    for n in (60_000, 120_000, 200_000):
        t_cpu = bench_cpu(n)
        t_gpu = bench_gpu(n)
        results.append((n, t_cpu, t_gpu))
        print(f"{n//1000:3d}k 点/帧: CPU {t_cpu:6.2f} ms | GPU {t_gpu:6.2f} ms | 加速 {t_cpu/t_gpu:5.1f}x")

    acc = verify()
    print(f"GPU vs CPU 结果一致性: {acc*100:.2f}% 像素一致")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager

        # 中文字体（Linux Noto Sans CJK），避免图表乱码
        for fp in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                   "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc"):
            if fp and __import__("os").path.exists(fp):
                font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = ["Noto Sans CJK SC", "Noto Sans CJK JP",
                                       "AR PL UMing CN", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        xs = [n // 1000 for n, _, _ in results]
        cpu_ms = [c for _, c, _ in results]
        gpu_ms = [g for _, _, g in results]
        speedups = [c / g for _, c, g in results]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        # 左图：耗时对比
        ax = axes[0]
        width = 0.35
        x = np.arange(len(xs))
        b1 = ax.bar(x - width / 2, cpu_ms, width, label="CPU (NumPy)", color="#d95926")
        b2 = ax.bar(x + width / 2, gpu_ms, width, label="GPU (PyTorch)", color="#3987e5")
        ax.set_xticks(x, [f"{v}k" for v in xs])
        ax.set_ylabel("Latency per frame (ms)")
        ax.set_title("Single-frame projection latency")
        ax.legend()
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                        f"{b.get_height():.2f}", ha="center", fontsize=9)
        # 右图：加速比
        ax = axes[1]
        ax.bar(x, speedups, color="#199e70")
        ax.set_xticks(x, [f"{v}k" for v in xs])
        ax.set_ylabel("Speedup (×)")
        ax.set_title("GPU speedup")
        for xi, s in zip(x, speedups):
            ax.text(xi, s + 0.5, f"{s:.1f}×", ha="center", fontsize=11, fontweight="bold")
        fig.tight_layout()
        fig.savefig("speedup.png", dpi=150, bbox_inches="tight")
        print("已生成 speedup.png")


if __name__ == "__main__":
    main()
