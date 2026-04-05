"""DuDoDp项目本地配置 - 适配实际数据格式"""

import os

# === DuDoDp项目根目录 ===
DUDODP_PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# === 数据路径 ===
DATA_ROOT = os.path.join(DUDODP_PROJECT_DIR, "DuDoDp-MAR", "test_data", "SynDeepLesion")
TEST_GEO_DIR = os.path.join(DATA_ROOT, "test_640geo")
DIR_LIST_FILE = os.path.join(DATA_ROOT, "test_640geo_dir.txt")
GT_KEY = "image"          # gt.h5中的GT图像键名

# === DuDoDp输出路径 ===
DUDODP_IMAGE_DIR = os.path.join(DUDODP_PROJECT_DIR, "outputs", "images")
DUDODP_FIGURE_DIR = os.path.join(DUDODP_PROJECT_DIR, "outputs", "figures")
DUDODP_METRICS_FILE = os.path.join(DUDODP_PROJECT_DIR, "outputs", "metrics.csv")

# === 可视化窗位（衰减系数单位，与程序A一致） ===
SOFT_WINDOW = (0.15, 0.25)
WIDE_WINDOW = (0.0, 0.45)

# === 参数 ===
IMAGE_SIZE = 416
NUM_METAL_MASKS = 10
