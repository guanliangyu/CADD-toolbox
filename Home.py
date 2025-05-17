"""
分子库代表性子集选择系统 - 主页
"""
import os

import sys
import streamlit as st

# 添加项目根目录到路径，确保能导入utils模块
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

# 导入工具模块
from utils.state_utils import initialize_session_state, display_state_sidebar

# 设置页面标题和配置
st.set_page_config(
    page_title="分子库代表性子集选择系统",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化会话状态
initialize_session_state()

# 在侧边栏显示当前状态
display_state_sidebar()

# 主页内容
st.title("分子库代表性子集选择系统 🧪")

st.markdown("""
## 欢迎使用分子库代表性子集选择系统

本应用实现了从大型分子库中提取代表性子集的流程，主要功能包括：
- 分子过滤和标准化
- 2D指纹和理化性质计算
- 3D构象生成和形状特征计算
- 多种聚类算法支持
- 子集代表性验证与评价

### 使用流程

1. **数据处理页面**：上传分子数据、配置参数并处理分子
2. **聚类与选择页面**：对处理后的分子进行聚类并选择代表性分子
3. **验证与下载页面**：验证选择的子集质量并下载结果
4. **可视化分析页面**：对分子和子集进行可视化分析
5. **批处理页面**：批量处理多个数据集

### 快速开始

请点击左侧菜单中的"数据处理"开始使用系统，或者点击下方按钮直接进入相应页面：
""")

# 创建按钮行
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("数据处理", use_container_width=True):
        st.switch_page("pages/1_数据处理.py")
        
with col2:
    if st.button("聚类与选择", use_container_width=True):
        st.switch_page("pages/2_聚类与选择.py")
        
with col3:
    if st.button("验证与下载", use_container_width=True):
        st.switch_page("pages/3_验证与下载.py")

# 创建第二行按钮
col4, col5, _ = st.columns(3)

with col4:
    if st.button("可视化分析", use_container_width=True):
        st.switch_page("pages/4_可视化分析.py")
        
with col5:
    if st.button("批处理", use_container_width=True):
        st.switch_page("pages/5_批处理.py")

# 添加应用程序信息
st.markdown("""
---
### 关于本系统

本系统基于RDKit、Streamlit和scikit-learn等开源工具开发，支持多种分子聚类和选择算法，
能够高效处理大规模分子库，是药物发现和虚拟筛选中的实用工具。

#### 支持的分子表示
- Morgan指纹（ECFP4/FCFP6）
- RDKit拓扑指纹
- MACCS结构键
- 理化性质（分子量、LogP、TPSA等）
- 3D构象特征（可选）

#### 支持的聚类算法
- Butina聚类（基于相似度阈值）
- K-means聚类（基于欧几里得距离）
- MaxMin选择（基于最大最小距离原则）

#### 可视化和验证功能
- 覆盖度分析
- 属性分布对比
- 最近邻分析
- 分子结构可视化
- 聚类结果可视化
""")

# 页面底部信息
st.sidebar.markdown("---")
st.sidebar.info(
    "本应用由VS开发团队开发。"
) 