#!/bin/bash
# CADD-Toolbox 分步环境创建脚本 - CUDA 12版本 (使用mamba加速)

set -e  # 遇到错误立即停止

echo "🚀 开始创建CADD-Toolbox环境 - CUDA 12版本 (使用mamba加速)..."
echo ""

# Step 0: 配置channels (优先配置以加速后续下载)
echo "📡 Step 0: 配置conda channels"
echo "添加清华源镜像..."
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud//pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
echo "添加必要的特殊channels..."
conda config --add channels rapidsai
conda config --add channels nvidia
conda config --add channels pytorch
conda config --set channel_priority flexible
echo "✅ Channels配置完成"
echo ""

# Step 1: 检查系统要求
echo "🔍 Step 1: 检查系统要求"
echo "检查GPU和CUDA驱动..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi | head -3
    echo "✅ NVIDIA GPU检测成功"
else
    echo "⚠️  未检测到NVIDIA GPU或驱动，将安装CPU版本"
    export USE_CPU_ONLY=true
fi

# 检查并安装mamba
echo "🔧 检查mamba安装状态"
if ! command -v mamba &> /dev/null; then
    echo "⚠️  mamba未安装，正在安装mamba..."
    conda install mamba -n base -c conda-forge -y
    echo "✅ mamba安装完成"
else
    echo "✅ mamba已安装"
fi
echo ""

# Step 2: 删除已存在的环境（如果有）
echo "🗑️  Step 2: 检查并删除已存在的CADD-Toolbox环境"
if conda env list | grep -q "CADD-Toolbox"; then
    echo "发现已存在的CADD-Toolbox环境，正在删除..."
    conda env remove -n CADD-Toolbox -y
    echo "✅ 旧环境已删除"
else
    echo "✅ 没有发现旧环境"
fi
echo ""

# Step 3: 创建基础Python环境
echo "📦 Step 3: 创建基础Python环境"
mamba create -n CADD-Toolbox python=3.10 -y

# Step 4: 激活环境
echo "🔧 Step 4: 激活环境"
eval "$(conda shell.bash hook)"
conda activate CADD-Toolbox

# Step 5: 安装统一的CUDA 12支持
echo "⚡ Step 5: 安装统一的CUDA 12支持"
if [ "$USE_CPU_ONLY" != "true" ]; then
    echo "安装CUDA 12.2 toolkit和runtime..."
    mamba install -c nvidia cuda-toolkit=12.2.2 cuda-cudart=12.2.* cuda-version=12.2 -y
    echo "✅ CUDA 12.2环境已配置"
else
    echo "⚠️  跳过CUDA安装（CPU模式）"
fi

# Step 6: 安装核心科学计算库
echo "🧮 Step 6: 安装核心科学计算库"
mamba install scipy scikit-learn -y

# Step 7: 安装PyTorch (GPU或CPU版本)
echo "🔥 Step 7: 安装PyTorch"
if [ "$USE_CPU_ONLY" != "true" ]; then
    echo "安装GPU版本PyTorch (CUDA 12.1)..."
    mamba install -c pytorch -c nvidia pytorch torchvision pytorch-cuda=12.1 -y
    echo "✅ GPU版本PyTorch安装完成"
else
    echo "安装CPU版本PyTorch..."
    mamba install -c pytorch pytorch torchvision cpuonly -y
    echo "✅ CPU版本PyTorch安装完成"
fi

# Step 8: 安装最新版FAISS
echo "🔍 Step 8: 安装最新版FAISS"
if [ "$USE_CPU_ONLY" != "true" ]; then
    echo "安装FAISS-GPU 1.9.0..."
    mamba install faiss-gpu=1.9.0 -y
    echo "✅ FAISS-GPU 1.9.0安装完成"
else
    echo "安装FAISS-CPU..."
    mamba install faiss-cpu -y
    echo "✅ FAISS-CPU安装完成"
fi

# Step 10: 安装RAPIDS核心库
echo "🚀 Step 10: 安装RAPIDS核心库"
if [ "$USE_CPU_ONLY" != "true" ]; then
    echo "尝试安装最新版RAPIDS..."
    mamba install -c rapidsai -c nvidia rapids=25.04 'cuda-version>=12.0,<=12.8' -y
    echo "✅ RAPIDS 25.04安装完成"
else
    echo "⚠️  跳过RAPIDS安装（CPU模式）"
fi

# Step 12: 安装分子计算库
echo "🧬 Step 12: 安装分子计算库"
mamba install rdkit -y
echo "✅ 分子计算库安装完成"

# Step 13: 安装可视化库
echo "📊 Step 13: 安装可视化库"
mamba install matplotlib seaborn plotly altair -y
echo "✅ 可视化库安装完成"

# Step 14: 安装Web框架和数据处理工具
echo "🌐 Step 14: 安装Web框架和数据处理工具"
mamba install streamlit tqdm psutil hdbscan umap-learn watchdog -y
echo "✅ Web框架和数据处理工具安装完成"

# Step 15: 安装文件处理和配置库
echo "📁 Step 15: 安装文件处理和配置库"
mamba install pyarrow h5py pyyaml requests pillow jinja2 -y
echo "✅ 文件处理和配置库安装完成"

# Step 16: 安装纯 Python 包（使用 mamba 而非 pip）
echo "📁 Step 16: 安装纯 Python 包（使用 mamba 而非 pip）"
mamba install loguru=0.7.3 rapidfuzz=3.13.0 safetensors=0.5.3 ml_dtypes=0.5.* nvidia-ml-py=12.575.51 -y
echo "✅ 纯 Python 包安装完成"

# Step 17: 环境验证
echo "🧪 Step 17: 环境验证"
echo "验证Python和基础库..."
python -c "import sys; print(f'Python版本: {sys.version}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"
python -c "import pandas; print(f'Pandas版本: {pandas.__version__}')"
python -c "import rdkit; print(f'RDKit版本: {rdkit.__version__}')"
python -c "import streamlit; print(f'Streamlit版本: {streamlit.__version__}')"

if [ "$USE_CPU_ONLY" != "true" ]; then
    echo ""
    echo "验证GPU支持..."
    echo "检查PyTorch CUDA支持:"
    python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA设备数: {torch.cuda.device_count()}')"
    
    echo "检查FAISS GPU支持:"
    python -c "import faiss; print(f'FAISS版本: {faiss.__version__}'); print(f'FAISS GPU数量: {faiss.get_num_gpus()}')"
    
    echo "检查CuML支持:"
    if python -c "import cuml; print(f'CuML版本: {cuml.__version__}')" 2>/dev/null; then
        echo "✅ CuML正常工作"
    else
        echo "⚠️  CuML可能需要额外配置"
    fi
else
    echo "⚠️  CPU模式，跳过GPU验证"
fi

echo ""
echo "🎉 环境创建完成!"
echo ""
echo "📋 环境信息:"
echo "  - 环境名称: CADD-Toolbox"
echo "  - Python版本: 3.10"
if [ "$USE_CPU_ONLY" != "true" ]; then
    echo "  - CUDA版本: 12.2"
    echo "  - PyTorch: GPU版本 (CUDA 12.1)"
    echo "  - FAISS: GPU版本 1.9.0"
    echo "  - RAPIDS: 最新可用版本"
else
    echo "  - 模式: CPU版本"
fi
echo ""
echo "🧪 下一步:"
echo "  1. 激活环境: conda activate CADD-Toolbox"
echo "  2. 运行测试: python test_environment.py"
echo "  3. 启动应用: streamlit run Home.py"
echo ""
echo "🚀 主要特性:"
echo "  - 统一的CUDA 12.2环境 (GPU模式)"
echo "  - 最新版本的关键库 (FAISS 1.9.0)"
echo "  - 完整的GPU加速支持"
echo "  - 自动依赖解决 (mamba)"
echo "  - 智能CPU/GPU模式切换"
echo ""
if [ "$USE_CPU_ONLY" != "true" ]; then
    echo "💡 GPU优化提示:"
    echo "  - 确保NVIDIA驱动 >= 535.86.10 (CUDA 12.2支持)"
    echo "  - 建议16GB+内存用于大型数据集"
    echo "  - 使用nvidia-smi监控GPU使用情况"
else
    echo "💡 CPU模式提示:"
    echo "  - 所有功能都可用，但性能可能较慢"
    echo "  - 考虑安装NVIDIA驱动以启用GPU加速"
fi 