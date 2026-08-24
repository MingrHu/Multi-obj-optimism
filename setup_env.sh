#!/usr/bin/env bash
# =============================================================================
# mobo 环境一键安装脚本（Linux / macOS，CPU-only）
#
# 步骤：
#   1. 检测操作系统与 python3
#   2. 创建并激活虚拟环境 .venv
#   3. 升级 pip
#   4. 从 PyTorch 官方 CPU 源安装 torch（避免拉取 GPU 版及大量 CUDA 包）
#   5. 以可编辑模式安装本包及开发依赖：pip install -e ".[dev]"
#   6. 可选：--with-gui 追加安装 PySide6
#   7. 自检导入
#
# 用法：
#   bash setup_env.sh              # 安装核心 + 开发依赖
#   bash setup_env.sh --with-gui   # 额外安装 GUI 依赖
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- 可配置项 ----
VENV_DIR="$SCRIPT_DIR/.venv"
TORCH_VERSION="2.10.0"
TORCH_CPU_INDEX="https://download.pytorch.org/whl/cpu"
# PyPI 镜像（留空则使用默认源）。国内可设为 https://mirrors.aliyun.com/pypi/simple/
PIP_INDEX="${PIP_INDEX:-}"

WITH_GUI=0
for arg in "$@"; do
    case "$arg" in
        --with-gui) WITH_GUI=1 ;;
        *) echo "未知参数：$arg"; exit 1 ;;
    esac
done

# ---- 1. 检测 OS 与 python3 ----
OS="$(uname -s)"
case "$OS" in
    Linux*)  echo "==> 检测到 Linux" ;;
    Darwin*) echo "==> 检测到 macOS" ;;
    *)       echo "警告：未识别的系统 $OS，将按类 Unix 处理" ;;
esac

if ! command -v python3 >/dev/null 2>&1; then
    echo "错误：未找到 python3，请先安装 Python 3.11+。"
    exit 1
fi
PYTHON="python3"
echo "==> 使用 $($PYTHON --version)"
"$PYTHON" -c "import sys; assert (3, 11) <= sys.version_info[:2] < (3, 13), '需要 Python 3.11/3.12'"

# ---- 2. 创建虚拟环境 ----
if [ ! -d "$VENV_DIR" ]; then
    echo "==> 创建虚拟环境 $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
else
    echo "==> 虚拟环境 $VENV_DIR 已存在，复用"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# 组装可选的 pip 镜像参数
PIP_ARGS=()
if [ -n "$PIP_INDEX" ]; then
    PIP_ARGS+=(-i "$PIP_INDEX")
fi

# ---- 3. 升级 pip ----
echo "==> 升级 pip"
python -m pip install -U pip setuptools wheel "${PIP_ARGS[@]}"

# ---- 4. 安装 CPU 版 torch ----
echo "==> 安装 CPU 版 torch==${TORCH_VERSION}"
python -m pip install "torch==${TORCH_VERSION}" --index-url "$TORCH_CPU_INDEX"

# ---- 5. 安装本包 + 开发依赖 ----
echo "==> 安装锁定的运行时、开发依赖并以可编辑模式安装 mobo"
python -m pip install -r "$SCRIPT_DIR/requirements-dev.txt" "${PIP_ARGS[@]}"
python -m pip install -e "$SCRIPT_DIR" --no-deps

# ---- 6. 可选 GUI ----
if [ "$WITH_GUI" -eq 1 ]; then
    echo "==> 安装 GUI 依赖 (PySide6)"
    python -m pip install -r "$SCRIPT_DIR/requirements-gui.txt" "${PIP_ARGS[@]}"
fi

# ---- 7. 自检 ----
echo "==> 自检导入"
python -m pip check
python -c "import numpy, pandas, scipy, sklearn, matplotlib, keras, tensorflow, pymoo, gymnasium, stable_baselines3, pyDOE, torch, mobo; print('torch', torch.__version__); print('mobo', mobo.__version__)"

echo ""
echo "✅ 环境安装完成。激活方式：source $VENV_DIR/bin/activate"
echo "   运行测试：pytest -m 'not slow'"
