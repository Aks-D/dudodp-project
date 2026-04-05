# DuDoDp-MAR 复现与评估项目

基于 [DuDoDp-MAR](https://github.com/DeepXuan/DuDoDp-MAR)（Dual-Domain Diffusion-based Purification for Metal Artifact Reduction）的复现项目，针对 SynDeepLesion 数据集进行了本地适配、批量推理与定量评估。

## 与原仓库的区别

| 对比项 | 原仓库 (DeepXuan/DuDoDp-MAR) | 本项目（当前版本） |
|--------|-------------------------------|--------|
| **投影算子** | `torch-radon` FanBeam（GPU 原生，PyTorch 集成） | `astra-toolbox` 扇束投影（GPU，通过 numpy 中转） |
| **投影流程** | Tensor 全程在 GPU，无数据搬运 | 每次 fp/bp 需 GPU→CPU→GPU，有额外开销 |
| **评估指标** | 仅输出推理图像，无定量评估 | 自动计算 PSNR / SSIM，输出 CSV 指标文件 |
| **可视化** | 无 | 自动生成 GT / MA / DuDoDp 三栏对比图（Soft / Wide 双窗位） |
| **运行方式** | 需手动调用 `mar.py` | 一键运行 `run_dudodp.py`，串联推理 + 评估 + 可视化 |
| **路径管理** | 硬编码在 YAML 配置中 | 集中管理于 `config_local.py`，便于迁移 |
| **环境依赖** | torch-radon（仅 Linux + CUDA） | astra-toolbox（跨平台，conda 安装） |

### 投影算子差异说明

原仓库使用 [`torch-radon`](https://github.com/matteo-ronchetti/torch-radon) 的 `FanBeam` 类实现扇束投影与反投影，数据全程以 PyTorch Tensor 留在 GPU 上。本项目因 `torch-radon` 仅支持 Linux + CUDA 编译，替换为 `astra-toolbox` 实现等效的扇束几何投影。主要差异：

1. **数据搬运开销**：astra 版每次 `fp()`/`bp()` 需将 Tensor 转为 numpy 送 CPU，投影后再转回 GPU Tensor
2. **投影核实现细节**：两者在射线离散化、插值方式上存在微小差异，导致重建结果略有不同
3. **滤波器实现**：astra FBP 使用内置 Hamming 滤波；torch-radon 使用 `filter_sinogram()` + `backward()`

这些差异导致本项目结果与原论文存在一定偏差（见下方实验结果）。后续计划在 Linux 云服务器上使用 `torch-radon` 重新运行以获得与原论文一致的结果。

## 项目结构

```
dudodp_project/
├── DuDoDp-MAR/              # 原仓库（submodule，已修改 mar.py 等使用 astra）
│   ├── mar.py               # 推理主程序（当前：astra 版）
│   ├── geometry/
│   │   └── syndeeplesion_data.py  # 数据加载（当前：astra 正投影）
│   └── patch_diffusion/
│       └── image_datasets.py      # 训练数据集（保留 torch-radon import，推理不使用）
├── guided-diffusion/        # OpenAI guided-diffusion（submodule）
├── run_dudodp.py            # 评估主程序：推理 + 指标计算 + 可视化
├── config_local.py          # 本地路径与参数配置
├── setup_dudodp.sh          # 环境一键配置脚本（torch-radon 版）
└── results_10images_fixed/  # 实验结果（astra 版）
    ├── metrics.csv          # 定量指标
    ├── figures/             # 可视化对比图
    └── inference/           # 推理输出图像
```

## 初步实验结果（astra-toolbox 版）

在 SynDeepLesion 测试集上评估 10 张图像（每张 10 个金属掩模，共 100 个样本）：

| 方法 | PSNR (dB) | SSIM |
|------|-----------|------|
| Metal Artifact (MA) | 12.54 | 0.8982 |
| **DuDoDp-MAR (astra)** | **31.22** | **0.9413** |

DuDoDp-MAR 相比含金属伪影的输入图像：
- **PSNR 提升约 18.68 dB**
- **SSIM 从 0.90 提升至 0.94**

> **注意：** 以上结果使用 astra-toolbox 替代 torch-radon，与原论文结果存在一定差异。原论文在完整测试集（200 张图像）上报告的 PSNR 约 33-34 dB。差异来源包括：投影算子实现不同、测试样本数量不同（10 vs 200）。后续将在云服务器上使用 torch-radon 复现原始结果。

## 环境配置

### 方案一：torch-radon 版（推荐，与原论文一致）

> 需要 **Linux + NVIDIA GPU + CUDA**，建议使用云服务器（AutoDL、Featurize 等）。

| 依赖 | 版本 | 说明 |
|------|------|------|
| OS | Ubuntu 20.04 | torch-radon 仅支持 Linux |
| Python | 3.9 | 与原仓库一致 |
| PyTorch | 1.13.1 + CUDA 11.7 | GPU 必需 |
| torch-radon | 2.0.0 | 扇束 Radon 变换算子，需 CUDA 编译 |
| guided-diffusion | - | OpenAI 扩散模型框架 |

```bash
git clone https://github.com/Aks-D/dudodp-project.git
cd dudodp-project
git submodule update --init --recursive
bash setup_dudodp.sh
```

> 使用 torch-radon 版时，需要将 `mar.py` 和 `syndeeplesion_data.py` 恢复为原仓库的 torch-radon 实现。

### 方案二：astra-toolbox 版（当前版本，跨平台）

适用于 Windows/Linux，无需编译 torch-radon：

```bash
conda create -n dudodp python=3.9 -y
conda activate dudodp
pip install torch torchvision
conda install -c astra-toolbox astra-toolbox -y
pip install h5py imageio scikit-image matplotlib pyyaml tqdm blobfile mpi4py
```

### 准备数据与模型

```bash
# 下载预训练模型
# 链接: https://drive.google.com/file/d/1pXsLIzQq_PBs52oZ5Sl5sXyGZj7tRdet/view
mkdir -p DuDoDp-MAR/Patch-diffusion-pretrained/
# mv model150000.pt DuDoDp-MAR/Patch-diffusion-pretrained/

# 链接 SynDeepLesion 测试数据
ln -s /path/to/SynDeepLesion DuDoDp-MAR/test_data/SynDeepLesion
```

### 运行

```bash
conda activate dudodp
python run_dudodp.py
```

## TODO

- [ ] 在 Linux 云服务器上配置 torch-radon 环境
- [ ] 使用 torch-radon 版恢复原始 `mar.py`，重新运行推理
- [ ] 在完整测试集（200 张图像）上评估并对比原论文结果

## 参考

- 原始论文：Hu et al., "Unsupervised CT Metal Artifact Reduction by Plugging Diffusion Priors in Dual Domains"
- 原始代码：https://github.com/DeepXuan/DuDoDp-MAR
- torch-radon：https://github.com/matteo-ronchetti/torch-radon
