"""分子库代表性子集选择系统 - 主页"""
from pathlib import Path
import sys
import streamlit as st

# 添加项目根目录到路径，确保能导入utils模块
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

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
- 2D/3D指纹与理化性质计算
- 多样化聚类与代表性选择
- 子集质量验证与可视化分析

### 推荐使用流程

1. **数据预处理**：上传数据并完成基本清洗与统计。
2. **基础成药性筛选**：依据Lipinski等规则快速筛除不合格分子。
3. **描述符生成**：在`生成2D描述符`、`生成3D构象`与`构象动力学优化`页面中构建分子特征，并通过`生成3D描述符`完成特征提取。
4. **代表性筛选与评估**：在`化合物多样性筛选`页面挑选子集，并使用`结构多样性评估（指纹数据）`检验覆盖度和分布。

请使用下方快捷按钮或左侧菜单进入相应页面：
""")

page_links = [
    ("数据预处理", "pages/0_数据处理.py"),
    ("基础成药性筛选", "pages/1_基础成药性筛选.py"),
    ("生成2D描述符", "pages/2_生成2D描述符.py"),
    ("生成3D构象", "pages/3-1_生成3D构象.py"),
    ("构象动力学优化", "pages/3-2_构象动力学优化.py"),
    ("生成3D描述符", "pages/3-3_生成3D描述符.py"),
    ("化合物多样性筛选", "pages/4_化合物多样性筛选.py"),
    ("结构多样性评估", "pages/5_结构多样性评估.py"),
]

for start in range(0, len(page_links), 3):
    cols = st.columns(3)
    for col, (label, target) in zip(cols, page_links[start : start + 3]):
        with col:
            if st.button(label, use_container_width=True):
                st.switch_page(target)

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
