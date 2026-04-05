#!/bin/bash
# DuDoDp-MAR 一键环境配置脚本
# 前置条件: 已通过 conda 创建并激活 Python 3.9 环境
#   conda create -n dudodp python=3.9 -y
#   conda activate dudodp
# 要求: Linux + NVIDIA GPU + CUDA 11.x
# 用法: bash setup_dudodp.sh

set -e

echo "=============================================="
echo "  DuDoDp-MAR 环境配置"
echo "=============================================="

# 检查是否在 conda 环境中
if [ -z "$CONDA_PREFIX" ]; then
    echo "[ERROR] 请先激活 conda 环境："
    echo "  conda activate dudodp"
    exit 1
fi

echo "当前 conda 环境: $CONDA_PREFIX"
echo "Python 版本: $(python --version)"

echo ""
echo "=== 1/6 安装 PyTorch 1.13.1 + CUDA 11.7 ==="
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

echo ""
echo "=== 2/6 安装 torch-radon ==="
# torch-radon 需要 CUDA toolkit，如果 pip 安装失败则从源码编译
pip install torch-radon==2.0.0 || {
    echo "[INFO] pip 安装 torch-radon 失败，尝试从源码编译..."
    if [ ! -d "torch-radon" ]; then
        git clone https://github.com/matteo-ronchetti/torch-radon.git
        cd torch-radon
        git checkout c1d3af21a64f4c97e74d37472dccda2294b65ae2
    else
        cd torch-radon
    fi
    python setup.py install
    cd ..
}

echo ""
echo "=== 3/6 安装 odl (用于几何参数构建) ==="
pip install https://github.com/odlgroup/odl/archive/master.zip

echo ""
echo "=== 4/6 安装 astra-toolbox (odl 后端) ==="
conda install -c astra-toolbox astra-toolbox -y

echo ""
echo "=== 5/6 安装其他 Python 依赖 ==="
pip install argparse blobfile h5py imageio mpi4py numpy Pillow pyyaml tqdm
pip install scikit-image matplotlib

echo ""
echo "=== 6/6 安装 guided-diffusion ==="
if [ -d "guided-diffusion" ]; then
    pip install -e guided-diffusion/
else
    echo "[WARN] guided-diffusion 目录不存在，跳过"
fi

echo ""
echo "=============================================="
echo "  环境配置完成！"
echo "=============================================="
echo ""
echo "验证安装："
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
from torch_radon.radon import FanBeam
print('  torch-radon: OK')
import odl
print('  odl: OK')
import astra
print('  astra-toolbox: OK')
print()
print('All dependencies installed successfully!')
"
echo ""
echo "下一步："
echo "  1. 下载预训练模型到 DuDoDp-MAR/Patch-diffusion-pretrained/model150000.pt"
echo "     链接: https://drive.google.com/file/d/1pXsLIzQq_PBs52oZ5Sl5sXyGZj7tRdet/view"
echo "  2. 准备 SynDeepLesion 测试数据:"
echo "     ln -s /path/to/SynDeepLesion DuDoDp-MAR/test_data/SynDeepLesion"
echo "  3. 运行: python run_dudodp.py"
echo ""
