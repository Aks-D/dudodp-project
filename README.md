# DuDoDp-MAR 复现与评估项目

基于 [DuDoDp-MAR](https://github.com/DeepXuan/DuDoDp-MAR)（Dual-Domain Diffusion-based Purification for Metal Artifact Reduction）的复现项目，针对 SynDeepLesion 数据集进行了本地适配、批量推理与定量评估。

## 与原仓库的区别

| 对比项 | 原仓库 (DeepXuan/DuDoDp-MAR) | 本项目 |
|--------|-------------------------------|--------|
| **定位** | 论文官方代码，提供模型与推理脚本 | 复现验证项目，增加了完整的评估流程 |
| **数据格式** | 需手动准备数据并配置路径 | 适配 SynDeepLesion `.h5` 数据格式，自动解析目录结构 |
| **评估指标** | 仅输出推理图像，无定量评估 | 自动计算 PSNR / SSIM，输出 CSV 指标文件 |
| **可视化** | 无 | 自动生成 GT / MA / DuDoDp 三栏对比图（Soft / Wide 双窗位） |
| **运行方式** | 需手动分步调用 `sample.py`、`mar.py` 等 | 一键运行 `run_dudodp.py`，串联推理 + 评估 + 可视化 |
| **环境配置** | 需自行安装依赖 | 提供 `setup_dudodp.sh` 一键配置脚本 |
| **路径管理** | 硬编码在 YAML 配置中 | 集中管理于 `config_local.py`，便于迁移 |

## 项目结构

```
dudodp_project/
├── DuDoDp-MAR/              # 原仓库（submodule）
├── guided-diffusion/        # OpenAI guided-diffusion（submodule）
├── run_dudodp.py            # 主程序：推理 + 评估 + 可视化
├── config_local.py          # 本地路径与参数配置
├── setup_dudodp.sh          # 环境一键配置脚本
└── results_10images_fixed/  # 实验结果
    ├── metrics.csv          # 定量指标
    ├── figures/             # 可视化对比图
    └── inference/           # 推理输出图像
```

## 初步实验结果

在 SynDeepLesion 测试集上评估 10 张图像（每张 10 个金属掩模，共 100 个样本）：

| 方法 | PSNR (dB) | SSIM |
|------|-----------|------|
| Metal Artifact (MA) | 12.54 | 0.8982 |
| **DuDoDp-MAR** | **31.22** | **0.9413** |

DuDoDp-MAR 相比含金属伪影的输入图像：
- **PSNR 提升约 18.68 dB**
- **SSIM 从 0.90 提升至 0.94**

结果表明 DuDoDp-MAR 能够有效去除 CT 图像中的金属伪影，在双域（sinogram + image）扩散净化框架下取得了显著的图像质量改善。

## 快速开始

### 1. 环境配置

```bash
bash setup_dudodp.sh
```

### 2. 准备数据与模型

- 下载预训练模型放到 `DuDoDp-MAR/Patch-diffusion-pretrained/model150000.pt`
- 将 SynDeepLesion 数据链接到 `DuDoDp-MAR/test_data/SynDeepLesion`

### 3. 运行

```bash
python run_dudodp.py
```

输出结果保存在 `outputs/` 目录下。

## 参考

- 原始论文：Hu et al., "Dual-Domain Diffusion-based Purification for Metal Artifact Reduction"
- 原始代码：https://github.com/DeepXuan/DuDoDp-MAR
