# 分子库代表性子集选择系统使用指南

本系统提供了从大型分子库中选择代表性子集的功能，可以通过两种方式使用：命令行模式和交互式Web界面。

## 1. 环境设置

首先，创建并激活conda环境：

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate vs_env
```

### GPU加速支持

系统支持使用GPU加速多种计算过程，包括：
- K-means聚类
- 距离矩阵计算
- PCA降维
- HDBSCAN聚类

要启用GPU加速，您需要具备：
- NVIDIA GPU
- 已安装CUDA（与环境中cudatoolkit版本匹配）
- 已安装cuDNN（用于部分深度学习加速库）

系统会自动检测可用GPU和相关库，并在合适时使用GPU加速。

## 2. 命令行模式

命令行模式适合批处理大型数据集和自动化流程。

### 基本用法

```bash
python scripts/run_pipeline.py --input 输入文件.csv --output 输出目录 --config configs/default_config.yml
```

### 参数说明

- `--input`: 输入文件路径，支持CSV格式（包含SMILES列）
- `--output`: 输出目录，用于保存结果文件
- `--config`: 配置文件路径，默认使用configs/default_config.yml
- `--smiles_col`: SMILES列名，默认为"SMILES"
- `--use_gpu`: 启用GPU加速（覆盖配置文件设置）
- `--gpu_id`: 指定使用的GPU设备ID（默认为0）

### 示例

```bash
# 使用测试数据和默认配置
python scripts/run_pipeline.py --input data/test_data.csv --output results

# 自定义SMILES列名
python scripts/run_pipeline.py --input my_data.csv --output results --smiles_col "canonical_smiles"

# 使用自定义配置
python scripts/run_pipeline.py --input data/large_library.csv --output results --config configs/my_config.yml

# 使用GPU加速
python scripts/run_pipeline.py --input data/large_library.csv --output results --use_gpu --gpu_id 0
```

## 3. 交互式Web界面

交互式界面提供了可视化操作和实时反馈，适合探索性分析和参数调整。

### 启动界面

```bash
streamlit run app.py
```

执行上述命令后，Streamlit将启动本地Web服务器，并自动打开浏览器。

### 使用流程

1. **数据加载**: 
   - 上传含有SMILES列的CSV文件
   - 指定SMILES列名（默认为"SMILES"）
   - 设置子集比例（占原始集合的百分比）

2. **参数配置**:
   - 选择聚类方法（Butina、K-means、MaxMin）
   - 配置指纹计算参数
   - 设置是否生成3D构象和计算电荷
   - 调整计算资源参数（批处理大小、并行数）
   - 配置GPU加速选项（如果可用）

3. **数据处理**:
   - 点击"开始处理分子"按钮
   - 系统会读取分子、计算特征，并显示处理结果

4. **聚类与选择**:
   - 点击"开始聚类和选择"按钮
   - 系统根据选择的算法执行聚类并挑选代表分子
   - 显示选择结果预览

5. **验证与下载**:
   - 点击"验证子集质量"按钮评估选择质量
   - 查看覆盖度指标、属性分布和最近邻分析
   - 下载CSV或SDF格式的代表性子集

## 4. 配置文件说明

配置文件使用YAML格式，包含以下主要部分：

### 数据处理配置

```yaml
data:
  filtering:
    enabled: true
    max_mw: 1000         # 最大分子量
    min_mw: 100          # 最小分子量
    max_atoms: 100       # 最大原子数
    remove_salts: true   # 是否移除盐
  
  conformers:
    enabled: true        # 是否生成3D构象
    method: "ETKDG"      # 构象生成方法
    force_field: "MMFF94"  # 力场类型
    
  charges:
    enabled: true        # 是否计算电荷
    method: "gasteiger"  # 电荷计算方法
```

### GPU加速配置

```yaml
gpu:
  enabled: true          # 是否启用GPU加速
  auto_detect: true      # 自动检测GPU可用性
  device_id: 0           # 使用的GPU设备ID
  use_batching: true     # 使用批处理以适应GPU内存
  batch_size: 5000       # GPU批处理大小
  features:
    kmeans: true         # GPU加速K-means聚类
    pca: true            # GPU加速PCA降维
    distances: true      # GPU加速距离矩阵计算
```

### 特征计算配置

```yaml
features:
  fingerprints:
    types: ["morgan"]    # 指纹类型
    morgan_radius: 2     # Morgan半径
    morgan_bits: 1024    # Morgan位数
    
  shape:
    enabled: true        # 是否计算形状特征
    
  properties:
    enabled: true        # 是否计算理化性质
    
  dimensionality_reduction:
    enabled: true        # 是否降维
    method: "pca"        # 降维方法
    n_components: 50     # 组分数量
```

### 聚类配置

```yaml
clustering:
  method: "butina"       # 聚类方法
  
  butina:
    cutoff: 0.4          # Butina相似度阈值
    
  kmeans:
    n_clusters: 100000   # K-means簇数量
    
  maxmin:
    init_method: "random"  # 初始点选择方法
```

## 5. 测试数据

系统提供了小型测试数据集`data/test_data.csv`，包含20个常见药物分子，可用于快速测试和熟悉系统功能。

## 6. 输出文件说明

系统会生成以下输出文件：

- `representative_subset_[时间戳].csv`: CSV格式的代表性子集
- `representative_subset_[时间戳].sdf`: SDF格式的代表性子集（带3D构象）
- `plots/`: 包含验证结果图表的目录
- `coverage_metrics_[时间戳].txt`: 覆盖度评估指标

## 7. 性能注意事项

- 处理大型库（>100万分子）时，请确保足够的内存和磁盘空间
- 3D构象生成非常耗时，对于初步筛选可先禁用此功能
- 对大型库，建议使用命令行模式并调整批处理参数
- 可以使用配置文件中的`batching`参数调整并行计算设置
- 对于特别大的数据集，启用GPU加速可以显著提高性能，特别是在聚类和距离计算阶段
- GPU内存有限，对于超大规模数据集，系统会自动使用批处理方式 