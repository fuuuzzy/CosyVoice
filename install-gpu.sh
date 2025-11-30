#!/bin/bash
# CosyVoice GPU 服务器安装脚本 (Linux + CUDA 12.1)

set -e

echo "================================================"
echo "CosyVoice GPU 版本安装"
echo "================================================"
echo ""
echo "系统要求："
echo "  - Linux 操作系统"
echo "  - NVIDIA GPU (CUDA 12.1)"
echo "  - Python 3.10"
echo ""

# 检查是否在 Linux 上
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "❌ 错误: 此脚本只能在 Linux 系统上运行"
    echo "   当前系统: $OSTYPE"
    exit 1
fi


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

case $choice in
    1)
        echo ""
        echo "安装基础 GPU 版本..."
        echo "1/2 安装核心依赖..."
        uv sync
        echo "2/2 安装 GPU 加速组件..."
        uv pip install -r requirements-gpu.txt
        ;;
    2)
        echo ""
        echo "安装完整 GPU 版本（包含 vLLM）..."
        echo "1/3 安装核心依赖..."
        uv sync
        echo "2/3 安装 GPU 加速组件..."
        uv pip install -r requirements-gpu.txt
        echo "3/3 安装 vLLM..."
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
echo "已安装的 GPU 组件："
echo "  - DeepSpeed (分布式训练)"
echo "  - ONNX Runtime GPU (CUDA 加速推理)"
echo "  - TensorRT (高性能推理)"
if [ "$choice" = "2" ]; then
    echo "  - vLLM (大语言模型加速)"
fi

cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
uv pip install ttsfrd_dependency-0.1-py3-none-any.whl
uv pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl

echo ""
echo "使用方法："
echo "  激活环境: source .venv/bin/activate"
echo "  运行示例: uv run python vllm_example.py"
echo "  启动 WebUI: uv run python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B"
echo ""
