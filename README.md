# 分子库代表性子集选择系统

本项目实现了从大型分子库（千万级别）中选取代表性子集（约1%）的算法和流程。系统利用RDKit进行分子处理，融合2D和3D特征进行多样性分析，通过聚类和降维技术确保选取的子集能够合理覆盖整个化学空间。

## 功能特点

- 数据准备与规范化：SMILES处理、3D构象生成、Gasteiger电荷计算
- 多种特征提取：拓扑指纹、3D形状特征、静电特征、理化性质
- 高效聚类算法：支持Butina算法、K-means、MaxMin多样性选择
- 并行计算优化：分批处理设计，支持大规模数据集
- 子集验证工具：覆盖度分析、分布比较、多样性评估
- 交互式界面：Streamlit应用支持参数调整和结果可视化
- GPU加速：支持使用CUDA加速聚类、距离计算和降维操作

## GPU加速特性

系统支持使用GPU加速以下计算密集型操作：

- **K-means聚类**：使用FAISS-GPU或cuML加速，处理速度提升5-20倍
- **距离矩阵计算**：使用FAISS或CuPy加速，特别适合大型分子库
- **PCA降维**：使用cuML加速，处理速度提升3-10倍
- **HDBSCAN聚类**：使用cuML加速密度聚类

这些加速特性对于超大规模分子库（百万级以上）特别有用，可以将小时级的计算缩短到分钟级。

## 使用方法

1. 环境配置：`conda env create -f environment.yml`
2. 激活环境：`conda activate vs_env`
3. 运行单步处理：`python scripts/run_pipeline.py --input [input_file] --output [output_dir] --config [config_file]`
4. 启动交互界面：`streamlit run app.py`
5. 使用GPU加速：`python scripts/run_pipeline.py --input [input_file] --output [output_dir] --use_gpu`

## 文件结构

- `scripts/`: 主要处理脚本
- `utils/`: 辅助功能模块
  - `molecular_utils.py`: 分子处理功能
  - `clustering_utils.py`: 聚类和多样性选择算法
  - `feature_utils.py`: 特征组合和降维
  - `validation_utils.py`: 子集验证工具
  - `gpu_utils.py`: GPU加速工具
- `data/`: 数据存储目录
- `notebooks/`: 实验和示例笔记本
- `tests/`: 单元测试
- `app.py`: Streamlit应用入口

## 配置参数

系统支持通过配置文件（YAML格式）调整各处理阶段的参数，包括：
- 分子过滤条件
- 指纹生成参数
- 3D构象生成设置
- 聚类算法选择与参数
- 并行处理配置
- GPU加速设置

详细参数说明请参考`configs/default_config.yml`和`USAGE.md`

## 性能建议

- 对大型库（>100万分子），建议使用命令行模式
- 启用GPU加速可显著提高聚类和距离计算性能
- 对于特别大的数据集，可调整GPU批处理参数以适应GPU内存限制
- 对于初步筛选，可禁用3D构象生成，这是计算中最耗时的部分

## 依赖库

主要依赖项：
- RDKit: 分子处理和指纹计算
- NumPy/Pandas: 数据处理
- scikit-learn: 机器学习算法
- FAISS-GPU: GPU加速的相似性搜索
- CuPy: GPU加速的数值计算
- cuML: GPU加速的机器学习
- Streamlit: 交互式Web界面 