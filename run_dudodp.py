"""DuDoDp-MAR 批量推理与评估 — 独立程序入口

适配 SynDeepLesion h5 数据格式（与 nmar_astra 一致）。

步骤：
1. 调用官方 sample_dudodp.py 生成去伪影图像
2. 加载结果，计算 PSNR/SSIM
3. 生成可视化
"""

import os
import csv
import subprocess
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from config_local import (
    DUDODP_PROJECT_DIR, TEST_GEO_DIR, DIR_LIST_FILE,
    DUDODP_IMAGE_DIR, DUDODP_FIGURE_DIR, DUDODP_METRICS_FILE,
    SOFT_WINDOW, WIDE_WINDOW, NUM_METAL_MASKS,
)


def get_sample_dirs():
    """获取测试样本目录列表"""
    if os.path.exists(DIR_LIST_FILE):
        with open(DIR_LIST_FILE) as f:
            lines = [l.strip() for l in f if l.strip()]
        return sorted({os.path.dirname(l) for l in lines})
    dirs = []
    for root, _, files in os.walk(TEST_GEO_DIR):
        if "gt.h5" in files:
            dirs.append(os.path.relpath(root, TEST_GEO_DIR))
    return sorted(dirs)


def step1_run_inference():
    """调用官方推理脚本"""
    print("=" * 50)
    print("Step 1: 运行 DuDoDp-MAR 推理")
    print("注意：每张图约 29 秒")
    print("=" * 50)

    cmd = [
        "python", "mar.py", "-c", "config/MAR.yaml",
    ]
    dudodp_dir = os.path.join(DUDODP_PROJECT_DIR, "DuDoDp-MAR")
    subprocess.run(cmd, cwd=dudodp_dir)


def step2_evaluate():
    """加载推理结果，计算指标，生成可视化"""
    print("=" * 50)
    print("Step 2: 评估 DuDoDp-MAR 结果")
    print("=" * 50)

    os.makedirs(DUDODP_FIGURE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(DUDODP_METRICS_FILE), exist_ok=True)

    csv_file = open(DUDODP_METRICS_FILE, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        'sample_dir', 'mask_idx',
        'MA_PSNR', 'MA_SSIM',
        'DuDoDp_PSNR', 'DuDoDp_SSIM',
    ])

    sample_dirs = get_sample_dirs()
    # 只取第1张图像的前10个mask结果（对应 mar.py 的 num_test_image=1）
    sample_dirs = sample_dirs[:1]
    total = len(sample_dirs) * NUM_METAL_MASKS
    count = 0
    sums = {k: {'psnr': 0, 'ssim': 0} for k in ['MA', 'DuDoDp']}

    for img_idx, sample_dir in enumerate(sample_dirs):
        full_dir = os.path.join(TEST_GEO_DIR, sample_dir)

        # 加载 GT
        with h5py.File(os.path.join(full_dir, "gt.h5"), 'r') as f:
            gt = f['image'][()]

        for mask_idx in range(NUM_METAL_MASKS):
            count += 1
            sid = f"{sample_dir.replace('/', '_')}_{mask_idx}"

            # 加载 MA
            with h5py.File(os.path.join(full_dir, f"{mask_idx}.h5"), 'r') as f:
                ma = f['ma_CT'][()]

            # 加载 DuDoDp 结果 — mar.py 输出为 results/DuDoDp-MAR/{img_idx:03d}_{mask_idx:03d}.png (16-bit)
            dudodp_result_dir = os.path.join(DUDODP_PROJECT_DIR, "DuDoDp-MAR", "results", "DuDoDp-MAR")
            dudodp_path = os.path.join(dudodp_result_dir, f"{img_idx:03d}_{mask_idx:03d}.png")
            if not os.path.exists(dudodp_path):
                print(f"[{count}/{total}] {sample_dir} mask={mask_idx} — 跳过（结果不存在: {dudodp_path}）")
                continue

            import imageio.v2 as imageio
            from skimage.transform import resize as sk_resize
            dudodp_result = imageio.imread(dudodp_path).astype(np.float64) / 65535.
            if dudodp_result.shape != gt.shape:
                dudodp_result = sk_resize(dudodp_result, gt.shape, order=1, anti_aliasing=True)
            dr = gt.max() - gt.min()

            p_ma = psnr(gt, ma, data_range=dr)
            s_ma = ssim(gt, ma, data_range=dr)
            p_dd = psnr(gt, dudodp_result, data_range=dr)
            s_dd = ssim(gt, dudodp_result, data_range=dr)

            writer.writerow([
                sample_dir, mask_idx,
                f"{p_ma:.2f}", f"{s_ma:.4f}",
                f"{p_dd:.2f}", f"{s_dd:.4f}",
            ])

            sums['MA']['psnr'] += p_ma;  sums['MA']['ssim'] += s_ma
            sums['DuDoDp']['psnr'] += p_dd; sums['DuDoDp']['ssim'] += s_dd

            # 可视化: GT / MA / DuDoDp
            fig, axes = plt.subplots(2, 3, figsize=(12, 8))
            images = [gt, ma, dudodp_result]
            titles = ['Ground Truth', 'Metal Artifact', 'DuDoDp-MAR']
            for row, (window, wname) in enumerate([(SOFT_WINDOW, 'Soft'), (WIDE_WINDOW, 'Wide')]):
                for col, (img, title) in enumerate(zip(images, titles)):
                    axes[row, col].imshow(img, cmap='gray', vmin=window[0], vmax=window[1])
                    if row == 0:
                        axes[row, col].set_title(title, fontsize=11)
                    axes[row, col].axis('off')
                axes[row, 0].set_ylabel(wname, fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(DUDODP_FIGURE_DIR, f"{sid}.png"),
                        dpi=150, bbox_inches='tight')
            plt.close()

            print(f"[{count}/{total}] {sample_dir} mask={mask_idx}  "
                  f"MA: {p_ma:.2f}dB  DuDoDp: {p_dd:.2f}dB")

    csv_file.close()

    # 汇总
    if count > 0:
        print(f"\n{'='*60}")
        print(f"=== DuDoDp-MAR 平均指标 ({count} 样本) ===")
        for name in ['MA', 'DuDoDp']:
            print(f"  {name:10s}: PSNR={sums[name]['psnr']/count:.2f}  "
                  f"SSIM={sums[name]['ssim']/count:.4f}")
    print(f"\n指标: {DUDODP_METRICS_FILE}")
    print(f"可视化: {DUDODP_FIGURE_DIR}")


if __name__ == "__main__":
    step1_run_inference()
    step2_evaluate()
