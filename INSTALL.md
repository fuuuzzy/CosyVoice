# CosyVoice 快速安装指南

本项目已迁移到 **uv** 现代化 Python 项目管理工具。

## 系统要求

- **Python**: 3.10 (严格要求)
- **macOS**: CPU 推理
- **Linux GPU**: CUDA 12.1 + NVIDIA GPU

---

## macOS 安装

```bash
# 安装 uv (如果未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆项目
cd /path/to/CosyVoice

# 删除旧环境（如果存在）
rm -rf .venv

# 安装依赖（会自动使用 Python 3.10）
export CXXFLAGS="-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 -isysroot $(xcrun --show-sdk-path)"
uv sync --python 3.10

# 激活环境
source .venv/bin/activate

# 验证安装
python --version  # 应该显示 Python 3.10.x
```

---

## Linux GPU 服务器安装

### 方法 1: 使用安装脚本（推荐）

```bash
cd /path/to/CosyVoice

# 运行安装脚本
bash install-gpu.sh
```

### 方法 2: 手动安装

```bash
# 确保删除旧环境
rm -rf .venv

# 1. 安装核心依赖（强制 Python 3.10）
uv sync --python 3.10

# 2. 安装 GPU 加速组件
uv pip install -r requirements-gpu.txt

# 3. (可选) 安装 vLLM 加速
uv pip install -r requirements-vllm.txt

# 激活环境
source .venv/bin/activate

# 验证安装
python --version  # 必须是 Python 3.10.x
```

---

## 常见问题

### Q: 为什么 uv sync 使用了 Python 3.12 而不是 3.10？

**原因**: 已存在的 `.venv` 目录可能使用了错误的 Python 版本。

**解决方案**:
```bash
# 删除虚拟环境
rm -rf .venv

# 强制使用 Python 3.10
uv sync --python 3.10
```

### Q: TensorRT 安装失败

**原因**: Python 版本不对（必须是 3.10），或者在 macOS 上尝试安装 GPU 组件。

**解决方案**:
- **Linux**: 确保使用 `uv sync --python 3.10`
- **macOS**: 不要安装 `requirements-gpu.txt`，TensorRT 仅支持 Linux

### Q: pyworld 编译失败 (macOS)

**原因**: 缺少 C++ 编译环境变量。

**解决方案**:
```bash
export CXXFLAGS="-I/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/c++/v1 -isysroot $(xcrun --show-sdk-path)"
uv sync --python 3.10
```

---

## 使用示例

```bash
# 激活环境
source .venv/bin/activate

# 或使用 uv run（自动激活）
uv run python your_script.py

# 语音克隆
uv run python voice_clone.py --text "你好世界" --reference audio.wav

# 启动 WebUI
uv run python webui.py --port 50000 --model_dir pretrained_models/CosyVoice2-0.5B
```

---

## 下载预训练模型

```bash
mkdir -p pretrained_models

# 下载 CosyVoice2-0.5B (推荐)
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B

# 或其他模型
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
git clone https://www.modelscope.cn/iic/CosyVoice-300M-SFT.git pretrained_models/CosyVoice-300M-SFT
```

---

## 文件说明

- `pyproject.toml`: 核心依赖配置（所有平台通用）
- `requirements-gpu.txt`: Linux GPU 加速组件
- `requirements-vllm.txt`: vLLM 加速推理
- `install-gpu.sh`: Linux GPU 自动安装脚本
- `.python-version`: 指定 Python 3.10
- `uv.lock`: 依赖锁定文件

