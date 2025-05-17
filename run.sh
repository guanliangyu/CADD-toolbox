#!/bin/bash
# 启动脚本 - 用于快速启动分子库代表性子集选择系统

# 检查命令行参数
if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "分子库代表性子集选择系统启动脚本"
    echo "用法:"
    echo "  ./run.sh web          # 启动网页界面"
    echo "  ./run.sh cli [args]   # 运行命令行工具，后面可接其他参数"
    echo "  ./run.sh -h           # 显示帮助信息"
    exit 0
fi

# 如果参数为空，启动网页界面
if [ -z "$1" ]; then
    echo "正在启动网页界面..."
    streamlit run app.py
    exit 0
fi

# 根据参数选择启动模式
case "$1" in
    web)
        echo "正在启动网页界面..."
        streamlit run app.py
        ;;
    cli)
        shift  # 移除第一个参数
        echo "正在运行命令行工具..."
        if [ -z "$1" ]; then
            # 没有额外参数，显示使用方法
            python scripts/run_pipeline.py --help
        else
            # 传递所有剩余参数给脚本
            python scripts/run_pipeline.py "$@"
        fi
        ;;
    *)
        echo "错误: 未知参数 '$1'"
        echo "使用 './run.sh --help' 查看帮助信息"
        exit 1
        ;;
esac 