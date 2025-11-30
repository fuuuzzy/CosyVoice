#!/bin/bash
# CosyVoice GPU 服务器安装脚本 (Linux + CUDA 12.1)
# 严格强制使用 Python 3.10

set -e

echo "================================================"
echo "CosyVoice GPU 版本安装"
echo "================================================"
echo ""

# 检查是否在 Linux 上
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ 错误: 此脚本只能在 Linux 系统上运行"
    echo "   当前系统: $OSTYPE"
    exit 1
fi

# 检查 uv 是否安装
if ! command -v uv &> /dev/null; then
    echo "❌ 错误: 未找到 uv"
    echo "请先安装 uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "uv 版本: $(uv --version)"
echo ""

# 检查 CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ 检测到 NVIDIA GPU："
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
    echo ""
else
    echo "⚠️  警告: 未检测到 nvidia-smi，请确保已安装 CUDA 驱动"
    echo ""
fi

# 选择安装类型
echo "请选择安装类型："
echo "  1) 基础 GPU 版本 (包含 ONNX Runtime GPU + TensorRT)"
echo "  2) 完整 GPU 版本 (基础 + vLLM 加速)"
echo ""
read -p "请输入选项 [1-2]: " choice

echo ""
echo "================================================"
echo "开始安装 (强制 Python 3.10)..."
echo "================================================"
echo ""

# 关键步骤：完全清理并重建环境
echo "步骤 1: 清理旧环境..."
if [ -d ".venv" ]; then
    echo "  删除 .venv 目录..."
    rm -rf .venv
fi
if [ -f "uv.lock" ]; then
    echo "  删除 uv.lock 文件..."
    rm -f uv.lock
fi

echo ""
echo "步骤 2: 创建 Python 3.10 虚拟环境..."
# 明确创建 Python 3.10 虚拟环境
uv venv --python 3.10 .venv

echo ""
echo "步骤 3: 验证 Python 版本..."
source .venv/bin/activate
PYTHON_VERSION=$(python --version)
echo "  当前 Python 版本: $PYTHON_VERSION"

if [[ ! $PYTHON_VERSION =~ "Python 3.10" ]]; then
    echo ""
    echo "❌ 错误: Python 版本不是 3.10"
    echo "   检测到: $PYTHON_VERSION"
    echo "   请手动安装 Python 3.10 或联系管理员"
    exit 1
fi

echo "  ✅ Python 版本正确"
echo ""

case $choice in
    1)
        echo "步骤 4: 安装核心依赖..."
        uv pip install -e .
        
        echo ""
        echo "步骤 5: 安装 GPU 加速组件..."
        uv pip install -r requirements-gpu.txt
        ;;
    2)
        echo "步骤 4: 安装核心依赖..."
        uv pip install -e .
        
        echo ""
        echo "步骤 5: 安装 GPU 加速组件..."
        uv pip install -r requirements-gpu.txt
        
        echo ""
        echo "步骤 6: 安装 vLLM..."
        uv pip install -r requirements-vllm.txt
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "✅ 安装完成！"
echo "================================================"
echo ""

# 最终验证
source .venv/bin/activate
FINAL_PYTHON_VERSION=$(python --version)
echo "最终 Python 版本: $FINAL_PYTHON_VERSION"

if [[ ! $FINAL_PYTHON_VERSION =~ "Python 3.10" ]]; then
    echo ""
    echo "⚠️  警告: Python 版本检查失败"
    exit 1
fi

echo ""
echo "已安装的 GPU 组件："
echo "  ✓ DeepSpeed (分布式训练)"
echo "  ✓ ONNX Runtime GPU (CUDA 加速推理)"
echo "  ✓ TensorRT (高性能推理)"
if [ "$choice" = "2" ]; then
    echo "  ✓ vLLM (大语言模型加速)"
fi

echo ""
echo "使用方法："
echo "  1. 激活环境:"
echo "     source .venv/bin/activate"
echo ""
echo "  2. 下载模型 (首次使用):"
echo "     mkdir -p pretrained_models"
echo "     git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B"
echo ""
echo "  3. 运行示例:"
echo "     python vllm_example.py"
echo "     python voice_clone.py --text \"你好\" --reference audio.wav"
echo "     python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B"
echo ""
