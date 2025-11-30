#!/bin/bash
# CosyVoice 安装脚本 - macOS 版本

set -e

echo "================================================"
echo "CosyVoice macOS 版本安装"
echo "================================================"
echo ""

# 设置 C++ 编译环境变量（用于编译 pyworld）
export CXXFLAGS="-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 -isysroot $(xcrun --show-sdk-path)"

echo "正在安装 CosyVoice 依赖（CPU 版本）..."
uv sync

echo ""
echo "================================================"
echo "✅ 安装完成！"
echo "================================================"
echo ""
echo "已安装组件："
echo "  - 核心依赖包"
echo "  - ONNX Runtime (CPU)"
echo "  - PyTorch (CPU)"
echo ""
echo "使用方法："
echo "  激活环境: source .venv/bin/activate"
echo "  运行示例: uv run python vllm_example.py"
echo "  启动 WebUI: uv run python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B"
echo ""
echo "注意：macOS 版本仅支持 CPU 推理"
echo ""