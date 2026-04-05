#!/bin/bash
# DuDoDp-MAR 环境配置脚本
# 用法: bash setup_dudodp.sh

set -e

echo "=== 1. 创建conda环境 ==="
conda create -n dudodp python=3.10 -y
conda activate dudodp

echo "=== 2. 安装PyTorch (需要GPU版本) ==="
pip install torch torchvision

echo "=== 3. 安装astra-toolbox ==="
conda install -c astra-toolbox astra-toolbox -y

echo "=== 4. 安装其他依赖 ==="
pip install guided-diffusion pyyaml tqdm matplotlib scikit-image

echo "=== 5. 克隆DuDoDp-MAR仓库 ==="
if [ ! -d "DuDoDp-MAR" ]; then
    git clone https://github.com/DeepXuan/DuDoDp-MAR.git DuDoDp-MAR
else
    echo "DuDoDp-MAR 仓库已存在，跳过克隆"
fi

echo "=== 6. 提示 ==="
echo ""
echo "请手动完成以下步骤："
echo "  1. 从README中的Google Drive链接下载预训练模型"
echo "     放到 DuDoDp-MAR/Patch-diffusion-pretrained/model150000.pt"
echo ""
echo "  2. 将SynDeepLesion数据链接到仓库期望路径："
echo "     ln -s /absolute/path/to/SynDeepLesion DuDoDp-MAR/test_data/SynDeepLesion"
echo ""
echo "  3. 确认/修改 DuDoDp-MAR/config/sample_dudodp.yaml 中的路径配置"
echo ""
echo "配置完成后运行: python run_dudodp.py"
