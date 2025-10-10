# GPU 环境检测报告

最近运行的 `test/check_gpu_support.py` 脚本显示当前环境支持以下 GPU 要求：

- PyTorch 2.5.1（CUDA 可用，设备：NVIDIA TITAN V）
- FAISS 1.9.0（检测到 1 块 GPU）
- cuML 25.04.00
- CuPy 13.5.1

因此，在默认情况下系统已满足实验室所需的 GPU 加速功能。如果后续出现驱动或权限故障导致 CUDA 初始化失败，可按指南重新检查设置。
