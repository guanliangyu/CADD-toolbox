#!/bin/bash
# CUDA版本检查脚本

echo "🔍 检查CUDA环境配置..."
echo ""

# 检查nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU信息:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader,nounits
    echo ""
    
    echo "🔧 CUDA Driver版本:"
    nvidia-smi | grep "CUDA Version" | awk '{print $9}'
    echo ""
else
    echo "❌ nvidia-smi 未找到，可能没有NVIDIA GPU或驱动未安装"
    echo ""
fi

# 检查nvcc
if command -v nvcc &> /dev/null; then
    echo "⚡ NVCC版本:"
    nvcc --version | grep "Cuda compilation tools"
    echo ""
else
    echo "⚠️  nvcc 未找到，CUDA开发工具包可能未安装"
    echo ""
fi

# 检查当前conda环境中的CUDA包
if command -v conda &> /dev/null; then
    echo "📦 当前环境中的CUDA相关包:"
    conda list | grep -E "(cuda|cupy|torch)" || echo "  未找到CUDA相关包"
    echo ""
fi

# 给出建议
echo "💡 安装建议:"
echo "  对于CUDA 11.x: conda install cudatoolkit cuda-version=11"
echo "  对于CUDA 12.x: conda install cuda-cudart cuda-version=12"
echo "  自动选择版本: conda install cuda-cudart cudatoolkit"
echo ""
echo "🎯 推荐操作:"
echo "  1. 运行: ./create_env_step_by_step.sh"
echo "  2. 或使用: conda env create -f environment.yml --solver=libmamba" 