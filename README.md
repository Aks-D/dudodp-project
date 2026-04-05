# DuDoDp-MAR 复现与评估项目

基于 [DuDoDp-MAR](https://github.com/DeepXuan/DuDoDp-MAR)（Dual-Domain Diffusion-based Purification for Metal Artifact Reduction）的复现项目，针对 SynDeepLesion 数据集进行了批量推理与定量评估。

## 与原仓库的区别

| 对比项 | 原仓库 (DeepXuan/DuDoDp-MAR) | 本项目 |
|--------|-------------------------------|--------|
| **投影算子** | `torch-radon` FanBeam + `odl` 几何构建 | 相同（已恢复为 torch-radon） |
| **评估指标** | 仅输出推理图像，无定量评估 | 自动计算 PSNR / SSIM，输出 CSV 指标文件 |
| **可视化** | 无 | 自动生成 GT / MA / DuDoDp 三栏对比图（Soft / Wide 双窗位） |
| **运行方式** | 需手动调用 `mar.py` | 一键运行 `run_dudodp.py`，串联推理 + 评估 + 可视化 |
| **路径管理** | 硬编码为原作者本地路径 | 修改为相对路径，集中管理于 `config_local.py` |
| **环境配置** | 需自行安装各依赖 | 提供 `setup_dudodp.sh` 一键安装（conda 环境下） |
| **测试数据** | 默认仅 1 张图像 | 扩展到 10 张图像（可配置至全部 200 张） |

### 关键修改清单

相对于原仓库 `DeepXuan/DuDoDp-MAR`，本项目仅修改了以下文件：

| 文件 | 修改内容 |
|------|---------|
| `DuDoDp-MAR/config/MAR.yaml` | `data_dir` 和 `model_path` 改为相对路径 |
| `DuDoDp-MAR/test_data/SynDeepLesion/test_640geo_dir.txt` | 从 1 张扩展到 10 张测试图像 |

**核心推理代码 `mar.py`、`syndeeplesion_data.py` 与原仓库完全一致**，使用 `torch-radon` FanBeam 算子。

### 新增文件

| 文件 | 说明 |
|------|------|
| `run_dudodp.py` | 评估主程序：调用 `mar.py` 推理 → 计算 PSNR/SSIM → 生成可视化 |
| `config_local.py` | 本地路径与参数集中配置 |
| `setup_dudodp.sh` | 云服务器一键环境安装脚本 |

## 项目结构

```
dudodp_project/
├── DuDoDp-MAR/              # 原仓库（submodule）
│   ├── mar.py               # 推理主程序（torch-radon，与原仓库一致）
│   ├── geometry/
│   │   ├── build_gemotry.py  # 几何参数构建（odl，与原仓库一致）
│   │   └── syndeeplesion_data.py  # 数据加载（odl ray_trafo，与原仓库一致）
│   ├── patch_diffusion/      # 扩散模型代码（与原仓库一致）
│   └── config/MAR.yaml       # 配置文件（路径已修改为相对路径）
├── guided-diffusion/        # OpenAI guided-diffusion（submodule）
├── run_dudodp.py            # 评估主程序：推理 + 指标计算 + 可视化
├── config_local.py          # 本地路径与参数配置
├── setup_dudodp.sh          # 云服务器一键环境安装脚本
└── results_10images_fixed/  # 早期实验结果（astra-toolbox 版，仅供参考）
    ├── metrics.csv          # 定量指标
    ├── figures/             # 可视化对比图
    └── inference/           # 推理输出图像
```

## 早期实验结果（astra-toolbox 版，仅供参考）

> 以下结果使用 astra-toolbox 替代 torch-radon 得到，与原论文存在偏差。当前代码已恢复为 torch-radon，需在 Linux 云服务器上重新运行。

在 SynDeepLesion 测试集上评估 10 张图像（每张 10 个金属掩模，共 100 个样本）：

| 方法 | PSNR (dB) | SSIM |
|------|-----------|------|
| Metal Artifact (MA) | 12.54 | 0.8982 |
| **DuDoDp-MAR (astra)** | **31.22** | **0.9413** |

PSNR 提升约 18.68 dB，SSIM 从 0.90 提升至 0.94。与原论文报告的 ~33-34 dB 存在差异，主要原因：
1. 投影算子不同（astra-toolbox vs torch-radon）
2. 测试样本数量不同（10 vs 200）

## 环境配置（Linux 云���务器）

> 需要 **Linux + NVIDIA GPU + CUDA 11.x**，推荐 AutoDL / Featurize 等云平台。

| 依赖 | 版本 | 说明 |
|------|------|------|
| OS | Ubuntu 20.04 | torch-radon 仅支持 Linux |
| Python | 3.9 | 与原仓库一致 |
| PyTorch | 1.13.1 + CUDA 11.7 | GPU 必需 |
| torch-radon | 2.0.0 | 扇束 Radon 变换算子，需 CUDA 编译 |
| odl | 1.0.0.dev0 | CT 几何参数构建 |
| astra-toolbox | - | odl 后端 |
| guided-diffusion | - | OpenAI 扩散模型框架 |

### 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/Aks-D/dudodp-project.git
cd dudodp-project
git submodule update --init --recursive

# 2. 创建 conda 环境（在 PyCharm 远程开发中配置为项目解释器）
conda create -n dudodp python=3.9 -y
conda activate dudodp

# 3. 一键安装所有依赖
bash setup_dudodp.sh

# 4. 准备预训练模型
mkdir -p DuDoDp-MAR/Patch-diffusion-pretrained/
# 从 Google Drive 下载 model150000.pt 放到上述目录
# 链接: https://drive.google.com/file/d/1pXsLIzQq_PBs52oZ5Sl5sXyGZj7tRdet/view

# 5. ���备 SynDeepLesion 测试数据
ln -s /path/to/SynDeepLesion DuDoDp-MAR/test_data/SynDeepLesion

# 6. 运行推理与评估
python run_dudodp.py
```

## 参考

- 原始论文：Hu et al., "Unsupervised CT Metal Artifact Reduction by Plugging Diffusion Priors in Dual Domains"
- 原始代码：https://github.com/DeepXuan/DuDoDp-MAR
- torch-radon：https://github.com/matteo-ronchetti/torch-radon
