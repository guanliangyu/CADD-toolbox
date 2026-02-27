# CADD-Toolbox - 计算机辅助药物设计工具箱

一个功能强大的计算机辅助药物设计工具箱，提供从大型分子库中选择代表性子集的完整解决方案。系统融合2D和3D分子特征，通过先进的聚类和多样性分析技术，帮助研究人员高效筛选和优化化合物库。

## 🌟 核心功能

### 分子处理与优化
- **数据预处理**：SMILES规范化、分子过滤、重复去除
- **3D构象生成**：支持ETKDG、MMFF力场优化
- **构象动力学优化**：OpenMM分子动力学模拟
- **成药性筛选**：基于Lipinski规则的基础筛选

### 特征计算与描述符
- **2D分子指纹**：Morgan、MACCS、拓扑指纹等
- **3D形状描述符**：RDKit 3D描述符、形状特征
- **理化性质**：分子量、LogP、极性表面积等
- **电荷分布**：Gasteiger电荷计算

### 多样性分析与聚类
- **高效聚类算法**：Butina、K-means、MaxMin多样性选择
- **降维技术**：PCA、UMAP降维可视化
- **密度聚类**：HDBSCAN自适应密度聚类
- **子集验证**：覆盖度分析、分布比较、多样性评估

### GPU加速支持
- **K-means聚类**：使用FAISS-GPU或cuML，处理速度提升5-20倍
- **距离矩阵计算**：FAISS或CuPy加速，适合大型分子库
- **PCA降维**：cuML加速，处理速度提升3-10倍
- **HDBSCAN聚类**：cuML加速密度聚类

### 用户界面
- **Streamlit Web界面**：交互式参数调整和实时可视化
- **后台任务管理**：支持长时间运行的计算任务
- **进度监控**：实时查看处理进度和系统资源使用
- **断点恢复**：支持中断后自动恢复计算

## 🚀 安装指南

### 系统要求

#### 最低要求
- Linux系统（Ubuntu 18.04+推荐）
- Python 3.10+
- 8GB RAM
- 10GB 可用磁盘空间
- 稳定的网络连接

#### 推荐配置
- Ubuntu 20.04 LTS
- Python 3.10
- 16GB+ RAM
- NVIDIA GPU（支持 CUDA 12.x，脚本默认安装 CUDA 12.2 运行时）
- 20GB+ SSD空间
- 100Mbps+ 网络

#### GPU支持要求
- NVIDIA GPU with CUDA 12.x
- CUDA Driver 535+（建议）
- 至少 8GB GPU 内存

### 一键安装（推荐）

CADD-Toolbox 提供了一个经过优化的安装脚本，能够自动检测并安装 mamba，然后分步创建环境，确保最高的成功率。

```bash
# 克隆项目
git clone https://github.com/guanliangyu/CADD-toolbox.git
cd CADD-toolbox

# 给安装脚本执行权限
chmod +x create_env_step_by_step.sh

# 运行安装脚本（自动安装mamba并创建环境）
./create_env_step_by_step.sh

# 激活环境
conda activate CADD-Toolbox

# 启动应用
streamlit run Home.py
```

### 安装过程说明

安装脚本会输出 Step 日志，核心流程包括：

1. 配置 conda channels，并检查 GPU/驱动
2. 检查并安装 mamba（如缺失）
3. 创建并激活 `CADD-Toolbox` 环境
4. 安装 CUDA/PyTorch/FAISS/RAPIDS（GPU 模式）或 CPU 版本依赖
5. 安装 RDKit、Streamlit、可视化与文件处理相关依赖
6. 执行环境自检并打印结果

### 环境验证

#### 基础环境检查
```bash
# 检查 Python 版本
python --version

# 运行核心依赖自检（推荐）
python test_environment.py
```

#### CUDA 环境检查
```bash
# 给脚本执行权限并运行
chmod +x check_cuda_version.sh
./check_cuda_version.sh

# 手动检查 CUDA
nvidia-smi

# 运行GPU特性自检脚本
python test/check_gpu_support.py
```

#### 运行测试
```bash
# 可选：JAX/TensorFlow 兼容性检查（未安装JAX时会提示并跳过）
python test/check_jax.py
```

### 为什么使用 mamba？

| 特性 | conda | mamba |
|------|-------|-------|
| 依赖解析速度 | 慢 | 快5-10倍 |
| 内存使用 | 高 | 低 |
| 并行下载 | 否 | 是 |
| 冲突解决 | 普通 | 更准确 |

### 安装故障排除

#### 问题1：权限错误
```bash
# 确保脚本有执行权限
chmod +x create_env_step_by_step.sh
```

#### 问题2：mamba 安装失败
```bash
# 手动安装 mamba
conda install mamba -n base -c conda-forge -y
```

#### 问题3：RAPIDS 依赖冲突
分步安装脚本已经优化了依赖安装顺序，但如果仍有问题：
```bash
# 删除环境重试
conda env remove -n CADD-Toolbox -y
./create_env_step_by_step.sh
```

#### 问题4：CUDA 版本不兼容
```bash
# 检查 CUDA 版本
nvidia-smi

# 手动安装匹配的 CUDA 版本
conda activate CADD-Toolbox
mamba install cuda-cudart=11.8 -c nvidia  # 对于 CUDA 11.x
mamba install cuda-cudart=12.0 -c nvidia  # 对于 CUDA 12.x
```

#### 问题5：内存不足
```bash
# 关闭其他程序，释放内存
# 或者在内存较大的机器上安装
```

#### 问题6：网络问题
```bash
# 配置 conda 镜像源（中国用户）
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
```

### 环境管理

#### 删除环境
```bash
conda env remove -n CADD-Toolbox
```

#### 导出环境配置
```bash
conda activate CADD-Toolbox
mamba env export > my_environment.yml
```

#### 环境文件说明
- `environment.yml`：主环境（含 GPU 计算栈，面向本地/服务器运行）。
- `environment.ci.yml`：CI 环境（CPU-only，面向 GitHub Actions 稳定构建）。

#### 环境信息查看
```bash
# 查看已安装的包
conda list

# 查看环境信息
conda info

# 查看 GPU 相关包
conda list | grep -E "(cuda|gpu|rapids)"
```

### 代码质量与 CI 一致性

为避免本地与 CI 格式规则漂移，建议在本地固定与 CI 相同版本：

```bash
pip install "ruff==0.15.4" "black==24.2.0"
```

提交前执行：

```bash
ruff check .
black --check --diff .
```

若失败可自动修复后复检：

```bash
ruff check . --fix
black .
ruff check .
black --check --diff .
```

## 📊 使用指南

CADD-Toolbox 提供了两种使用方式：直观的 Web 界面和高效的命令行模式。Web 界面适合交互式探索和学习，命令行模式适合批处理和自动化流程。

### 🌐 Web 界面使用（推荐新手）

#### 启动应用

```bash
# 激活环境
conda activate CADD-Toolbox

# 启动应用
streamlit run Home.py
```

应用启动后会自动打开浏览器，或者访问 `http://localhost:8501`

#### 页面功能详解

##### 1. 数据处理（预处理）
- **功能**：上传分子文件，进行基础数据清理和格式化
- **支持格式**：CSV、SDF 文件
- **主要操作**：
  - SMILES 规范化
  - 分子有效性检查
  - 重复分子去除
  - 数据统计预览

##### 2. 基础成药性筛选
- **功能**：基于 Lipinski 规则等药物化学规则筛选
- **筛选条件**：
  - 分子量（< 500 Da）
  - LogP（< 5）
  - 氢键供体（< 5）
  - 氢键受体（< 10）
  - 极性表面积等
- **自定义规则**：支持调整筛选参数

##### 3. 生成2D描述符
- **功能**：计算分子的2D结构特征
- **描述符类型**：
  - Morgan 指纹
  - MACCS 指纹
  - 拓扑指纹
  - 理化性质描述符
- **输出**：特征矩阵和统计信息

##### 4-1. 生成3D构象
- **功能**：为分子生成3D空间结构
- **参数选项**：
  - 构象生成算法（ETKDG 推荐）
  - 构象数量
  - 能量最小化方法
- **注意**：耗时较长，大数据集建议使用后台模式

##### 4-2. 构象动力学优化
- **功能**：使用 OpenMM 进行分子动力学模拟优化
- **智能后台执行**：
  - 支持长时间运行
  - 断点恢复功能
  - 资源监控
  - 不受页面刷新影响
- **任务管理**：
  - 查看运行日志
  - 监控进度
  - 停止/恢复任务

##### 4-3. 生成3D描述符
- **功能**：计算3D结构相关的分子描述符
- **特征类型**：
  - 3D 形状描述符
  - 静电特征
  - 表面积和体积
  - 几何描述符
- **后台支持**：支持智能后台计算

##### 5. 化合物多样性筛选
- **功能**：自动生成 GPU 加速的多样性筛选脚本并批量运行
- **工作流程**：
  - 根据描述符 CSV 和筛选参数生成 `subset_select_*.py/.sh`
  - 可选在后台执行脚本并实时查看日志、下载结果
- **可调参数**：
  - 子集大小（保留样本数）
  - 距离度量（欧氏/曼哈顿/余弦）
  - 首个分子策略（随机/质心/首行）
  - GPU/FP16 开关（自动检测 GPU 可用性）
- **结果验证**：界面内提供输出下载、日志查看和任务状态管理

##### 6. 结构多样性评估（指纹数据）
- **功能**：针对任意数值化指纹（可含 2D/3D 特征）进行多样性统计、聚类与可视化
- **核心能力**：
  - 流式加载 + float16 精度，支持大规模指纹矩阵
  - k-NN + 随机采样替代完整相似性矩阵，复杂度从 O(N²) 降到 O(N)
  - “采样比例”滑块可将两个数据集按 1%–100% 抽样，控制计算规模
  - 优先使用 FAISS/cuML/cuPy/PyTorch 的 GPU 加速（k-NN、PCA→UMAP、K-means、DBSCAN），自动回退 CPU
  - 支持对筛选前后数据集做覆盖度、分布、聚类质量对比
- **使用场景**：
  - 直接对生成的指纹 CSV（2D/3D）进行多样性评估
  - 对筛选前后数据集进行覆盖度、分布、聚类质量对比
- **模块化实现**：
  - `pages/5_结构多样性评估.py` 负责参数交互与流程编排
  - `utils/structure_diversity_data.py` 负责数据读取、缓存与抽样
  - `utils/structure_diversity_similarity.py` 负责相似性与多样性统计
  - `utils/structure_diversity_analysis.py` 负责降维与聚类计算
  - `utils/structure_diversity_visualization.py` 负责图表与分布对比渲染

#### 使用流程建议

##### 基础流程（推荐新手）
1. **数据处理** → 上传并清理数据
2. **成药性筛选** → 基础过滤
3. **生成2D描述符** → 计算2D特征
4. **多样性筛选（2D）** → 选择代表性子集
5. **多样性评估** → 验证结果质量

##### 完整流程（高级用户）
1. **数据处理** → 上传并清理数据
2. **成药性筛选** → 基础过滤
3. **生成2D描述符** → 计算2D特征
4. **生成3D构象** → 生成3D结构
5. **构象动力学优化** → MD 模拟优化
6. **生成3D描述符** → 计算3D特征
7. **多样性筛选（2D+3D）** → 综合特征选择
8. **结构多样性评估（指纹数据）** → 全面验证

### 💻 命令行模式（高级用户）

#### 基本使用

```bash
# 基础命令
python scripts/run_pipeline.py --input /path/to/molecules.csv --output results/

# 完整参数
python scripts/run_pipeline.py \
  --input /path/to/molecules.sdf \
  --output results/ \
  --config configs/default_config.yml \
  --smiles_col "SMILES" \
  --use_gpu \
  --gpu_id 0
```

> 说明：当输入是 SDF 时，`run_pipeline.py` 会自动从 `ROMol` 生成 `SMILES` 列，可直接使用默认 `--smiles_col`。

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入文件路径 | 必需 |
| `--output` | 输出目录 | 必需 |
| `--config` | 配置文件路径 | `configs/default_config.yml` |
| `--smiles_col` | SMILES 列名 | `"SMILES"` |
| `--use_gpu` | 启用 GPU 加速 | False |
| `--gpu_id` | GPU 设备 ID | 0 |

#### 批处理示例

```bash
# 处理多个文件
for file in data/*.csv; do
  echo "Processing $file"
  python scripts/run_pipeline.py --input "$file" --output "results/$(basename $file .csv)"
done

# 使用自定义配置（先从默认配置复制一份）
cp configs/default_config.yml configs/large_dataset_config.yml
python scripts/run_pipeline.py \
  --input /path/to/large_library.csv \
  --output results/large_lib \
  --config configs/large_dataset_config.yml \
  --use_gpu
```

## ⚙️ 配置文件详解

命令行入口 `scripts/run_pipeline.py` 默认加载 `configs/default_config.yml`。可以复制该文件并按需修改。配置文件使用 YAML 格式，主要包括以下部分：

### 数据处理配置

```yaml
data:
  filtering:
    enabled: true
    max_mw: 1000          # 最大分子量
    min_mw: 100           # 最小分子量
    max_atoms: 100        # 最大原子数
    remove_salts: true    # 移除盐分
    neutralize: true      # 中和电荷
    standardize: true     # 分子标准化

  conformers:
    enabled: true         # 生成3D构象
    method: "ETKDG"       # 构象生成方法
    force_field: "MMFF94" # 力场类型
    max_iters: 200        # 力场优化迭代
    num_conf: 1           # 每个分子的构象数量

  charges:
    enabled: true         # 计算电荷
    method: "gasteiger"   # 电荷计算方法

  batching:
    enabled: true
    batch_size: 10000     # 每批分子数
    n_jobs: -1            # 并行进程数
```

### GPU 加速配置

```yaml
gpu:
  enabled: true          # 启用GPU加速
  auto_detect: true      # 自动检测GPU
  device_id: 0           # GPU设备ID
  use_batching: true     # 使用批处理
  batch_size: 5000       # 批处理大小
  
  features:
    kmeans: true         # GPU加速K-means
    pca: true            # GPU加速PCA
    distances: true      # GPU加速距离计算
    transformers: true   # GPU分子表示学习/转换
```

### 特征计算配置

```yaml
features:
  fingerprints:
    types: ["morgan"]          # 支持 morgan/rdkit/maccs 等
    morgan_radius: 2
    morgan_bits: 1024

  shape:
    enabled: true
    descriptors: ["usr", "moments"]

  electrostatics:
    enabled: true
    descriptors: ["charges_stats", "dipole"]

  properties:
    enabled: true
    descriptors: ["mw", "logp", "tpsa", "hba", "hbd", "rotatable_bonds"]

  dimensionality_reduction:
    enabled: true
    method: "pca"
    n_components: 50
    scaler: "standard"
    variance_ratio: 0.95
```

### 聚类配置

```yaml
clustering:
  method: "butina"       # 聚类方法
  
  butina:
    cutoff: 0.4          # 相似度阈值
    
  kmeans:
    n_clusters: 100000   # 簇数量
    
  maxmin:
    init_method: "random" # 初始化方法
    distance_measure: "combo"
```

## 📊 结果文件说明

### 输出文件结构

```
results/
├── representative_subset_20241201_143022.csv       # CSV结果
├── representative_subset_20241201_143022.sdf       # SDF结果
├── coverage_metrics_20241201_143022.txt            # 覆盖度指标
├── processed_results_20241201_143022.pkl           # 可选：中间结果（save_intermediates=true）
└── plots/
    ├── property_distribution_20241201_143022.png
    └── nearest_neighbor_distribution_20241201_143022.png

# 另：运行目录下会生成日志文件
molecular_subset.log
```

### 结果解读

#### CSV 结果文件
- 包含选中的代表性分子
- 默认保留输入中的原始列与记录顺序子集
- 不会自动附加聚类标签或多样性评分列

#### 覆盖度指标
- `coverage_ratio`：在距离阈值内被覆盖的分子比例
- `mean_distance` / `median_distance` / `max_distance`：全库到子集最近邻距离统计
- `radius_90` / `radius_95`：覆盖 90%/95% 分子的有效半径
- `hybrid_score`：覆盖率与距离表现的综合指标

## 🚀 性能优化建议

### 硬件优化
1. **内存**：建议 16GB+ 用于大型数据集
2. **存储**：使用 SSD 提升 I/O 性能
3. **GPU**：启用 GPU 加速可显著提升性能
4. **CPU**：多核 CPU 有助于并行处理

### 软件优化
1. **分批处理**：对于大型数据集，调整批处理大小
2. **特征选择**：根据需求选择必要的特征类型
3. **3D构象**：对于初步筛选，可先禁用 3D 特征计算
4. **GPU 批处理**：调整 GPU 批处理大小以适应显存

### 处理策略
```yaml
# 大数据集配置示例
data:
  batching:
    enabled: true
    batch_size: 10000

gpu:
  batch_size: 5000       # 根据显存调整

features:
  fingerprints:
    types: ["morgan"]    # 先使用单一指纹类型

clustering:
  method: "butina"       # Butina 通常比 K-means 更快
  cutoff: 0.6            # 较高阈值减少计算量
```

### 大数据集处理建议
- 对于百万级分子库，使用命令行模式
- 启用分块处理模式
- 考虑先进行2D筛选，再计算3D特征
- 使用后台任务模式，避免界面超时

## 📁 项目结构

```
CADD-toolbox/
├── Home.py                    # Streamlit主应用
├── create_env_step_by_step.sh # 环境安装脚本
├── environment.yml            # 主Conda环境（GPU）
├── environment.ci.yml         # CI环境（CPU-only）
├── .github/workflows/ci.yml   # GitHub Actions流程
├── pages/                     # Streamlit页面
│   ├── 0_数据处理.py
│   ├── 1_基础成药性筛选.py
│   ├── 2_生成2D描述符.py
│   ├── 3-1_生成3D构象.py
│   ├── 3-2_构象动力学优化.py
│   ├── 3-3_生成3D描述符.py
│   ├── 4_化合物多样性筛选.py
│   └── 5_结构多样性评估.py
├── utils/                     # 核心工具模块
│   ├── molecular_utils.py     # 分子处理
│   ├── clustering_utils.py    # 聚类算法
│   ├── feature_utils.py       # 特征计算
│   ├── gpu_utils.py           # GPU加速
│   ├── descriptor_generation.py # 2D描述符/指纹生成
│   ├── structure_diversity_data.py        # 结构多样性：数据读取/缓存/抽样
│   ├── structure_diversity_similarity.py  # 结构多样性：相似性/多样性统计
│   ├── structure_diversity_analysis.py    # 结构多样性：降维/聚类分析
│   ├── structure_diversity_visualization.py # 结构多样性：可视化与分布对比
│   ├── background_*.py       # 后台任务管理
│   └── validation_utils.py   # 结果验证
├── configs/                   # 配置文件
├── data/                     # 数据目录
├── scripts/                  # 命令行脚本
└── test/                     # 测试文件
```

## 🔧 故障排除

### 常见问题

#### 1. 内存不足错误
```bash
# 解决方案：减小批处理大小
# 编辑配置文件，调整 batch_size
```

#### 2. GPU 相关错误
```bash
# 检查 GPU 状态
nvidia-smi

# 测试 GPU 库
python -c "import cudf; print('GPU available')"
```

#### 3. 计算时间过长
- 减少构象数量
- 使用较高的相似度阈值
- 启用 GPU 加速
- 使用分批处理

#### 4. 结果质量不佳
- 调整聚类参数
- 尝试不同的特征组合
- 增加构象多样性
- 检查数据质量

## 🔬 算法特性

### 多样性选择算法
- **Butina聚类**：基于Tanimoto相似度的快速聚类
- **K-means**：基于特征向量的无监督聚类
- **MaxMin算法**：最大最小距离多样性选择
- **层次聚类**：支持不同linkage方法

### 特征工程
- **多尺度指纹**：从分子片段到整体结构特征
- **3D形状特征**：考虑分子立体结构信息
- **理化性质组合**：多维度分子属性描述
- **降维优化**：PCA、UMAP等降维技术

## 📚 使用技巧

### 1. 数据准备
- 确保 SMILES 格式正确
- 预先去除明显的错误分子
- 保持一致的分子 ID 格式

### 2. 参数调优
- 从默认参数开始
- 小数据集上测试参数效果
- 根据结果逐步调整

### 3. 结果验证
- 检查覆盖度指标
- 可视化聚类结果
- 比较不同参数的效果

### 4. 大数据集处理
- 使用命令行模式
- 启用后台处理
- 监控系统资源使用

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目采用MIT许可证 - 详见 `LICENSE` 文件

## 🔗 参考资源

- [RDKit官方文档](https://www.rdkit.org/docs/)
- [RAPIDS官方文档](https://rapids.ai/)
- [Streamlit官方文档](https://docs.streamlit.io/)
- [OpenMM官方文档](http://openmm.org/documentation.html)
- [mamba 官方文档](https://mamba.readthedocs.io/)
- [PyTorch 安装指南](https://pytorch.org/get-started/locally/)

## 📞 获取帮助

如果遇到问题：

1. 首先查看本文档的故障排除部分
2. 检查是否有网络连接问题
3. 确认系统满足最低要求
4. 提交 GitHub Issue 并附上详细的错误信息
5. 项目仓库地址：`https://github.com/guanliangyu/CADD-toolbox`

---

**提示**：安装过程可能需要 15-30 分钟，请耐心等待。建议先使用小数据集熟悉工具功能，然后再处理大型数据集。

**CADD-Toolbox** - 让药物设计更简单、更高效！
