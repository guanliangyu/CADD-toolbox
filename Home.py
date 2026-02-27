"""分子库代表性子集选择系统 - 主页"""

from pathlib import Path
import sys
import os
from datetime import datetime
import importlib.util
import streamlit as st

# 添加项目根目录到路径，确保能导入utils模块
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 导入工具模块
from utils.state_utils import (  # noqa: E402
    initialize_session_state,
    display_state_sidebar,
)

# 设置页面标题和配置
st.set_page_config(
    page_title="分子库代表性子集选择系统",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 初始化会话状态
initialize_session_state()

# 在侧边栏显示当前状态
display_state_sidebar()

DATA_DIR = project_root / "data"
APP_VERSION = os.environ.get("CADD_TOOLBOX_VERSION", "dev")
UPDATED_AT = datetime.fromtimestamp(Path(__file__).stat().st_mtime).strftime("%Y-%m-%d")


def _pkg_available(pkg_name: str) -> bool:
    return importlib.util.find_spec(pkg_name) is not None


def detect_env_status() -> dict:
    """检测环境状态（轻量，不主动导入重型依赖）"""
    status = {
        "torch_installed": _pkg_available("torch"),
        "cuda_available": False,
        "faiss_installed": _pkg_available("faiss"),
        "faiss_gpu": False,
        "cupy_installed": _pkg_available("cupy"),
        "cuml_installed": _pkg_available("cuml"),
    }

    if status["torch_installed"]:
        try:
            import torch  # type: ignore

            status["cuda_available"] = bool(torch.cuda.is_available())
        except Exception:
            status["cuda_available"] = False

    if status["faiss_installed"]:
        try:
            import faiss  # type: ignore

            status["faiss_gpu"] = bool(faiss.get_num_gpus() > 0)
        except Exception:
            status["faiss_gpu"] = False

    return status


def _has_data_file(prefix: str, suffix: str) -> bool:
    if not DATA_DIR.exists():
        return False
    for path in DATA_DIR.rglob(f"*{suffix}"):
        if path.is_file() and path.name.startswith(prefix):
            return True
    return False


def _has_data_file_contains(keyword: str, suffix: str) -> bool:
    if not DATA_DIR.exists():
        return False
    for path in DATA_DIR.rglob(f"*{suffix}"):
        if path.is_file() and keyword in path.name:
            return True
    return False


def build_workflow_status() -> list[tuple[str, bool, str]]:
    """构建流程完成度摘要"""
    full_df = st.session_state.get("full_df")
    preview_df = st.session_state.get("preview_df")
    filtered_df = st.session_state.get("filtered_df")
    druglike_df = st.session_state.get("druglike_df")
    dub_library_df = st.session_state.get("dub_library_df")

    data_loaded = full_df is not None or preview_df is not None
    druglike_done = filtered_df is not None or druglike_df is not None
    desc2d_done = _has_data_file("2d_fingerprint_", ".csv")
    conf_done = _has_data_file_contains("conformer", ".sdf")
    desc3d_done = _has_data_file("descriptors_", ".csv")
    subset_done = _has_data_file("subset_", ".csv")
    eval_done = bool(
        st.session_state.get("metric_cache") or st.session_state.get("fps_cache")
    )
    dub_loaded = dub_library_df is not None

    return [
        (
            "1. 数据处理（预处理）",
            data_loaded,
            "已加载数据" if data_loaded else "未检测到数据加载状态",
        ),
        (
            "2. 基础成药性筛选",
            druglike_done,
            "已完成筛选" if druglike_done else "未检测到筛选结果",
        ),
        (
            "3. 生成2D描述符",
            desc2d_done,
            "检测到 2D 输出文件" if desc2d_done else "未检测到 2D 输出文件",
        ),
        (
            "4. 生成3D构象/构象优化",
            conf_done,
            "检测到 3D 构象文件" if conf_done else "未检测到 3D 构象文件",
        ),
        (
            "5. 生成3D描述符",
            desc3d_done,
            "检测到 3D 描述符文件" if desc3d_done else "未检测到 3D 描述符文件",
        ),
        (
            "6. 化合物多样性筛选",
            subset_done,
            "检测到子集文件" if subset_done else "未检测到子集文件",
        ),
        (
            "7. 结构多样性评估（指纹数据）",
            eval_done,
            "本会话已运行评估" if eval_done else "本会话尚未运行评估",
        ),
        (
            "8. Deubiquitinase Focused Library",
            dub_loaded,
            "已读取 CSV 数据" if dub_loaded else "尚未读取 CSV 数据",
        ),
    ]


# 主页内容
st.title("分子库代表性子集选择系统 🧪")
st.markdown("面向分子库筛选与代表性子集构建的工作台。")

st.markdown("### 快速开始")
csv_count = len(list(DATA_DIR.rglob("*.csv"))) if DATA_DIR.exists() else 0
sdf_count = len(list(DATA_DIR.rglob("*.sdf"))) if DATA_DIR.exists() else 0
st.info(f"当前 `data/` 中检测到 `{csv_count}` 个 CSV、`{sdf_count}` 个 SDF 文件。")
quick_col1, quick_col2 = st.columns(2)
with quick_col1:
    if st.button("从数据处理开始（推荐）", use_container_width=True):
        st.switch_page("pages/0_数据处理.py")
with quick_col2:
    if st.button("直接进入结构多样性评估", use_container_width=True):
        st.switch_page("pages/5_结构多样性评估.py")

st.markdown(
    """
### 推荐流程
1. `数据处理（预处理）`：加载并校验 SMILES 数据。
2. `基础成药性筛选`：应用 Lipinski / PAINS 等规则。
3. `生成2D描述符`、`生成3D构象`、`构象动力学优化`、`生成3D描述符`：构建建模特征。
4. `化合物多样性筛选`：基于最大最小距离贪心算法生成代表性子集。
5. `结构多样性评估（指纹数据）`：评估子集覆盖与分布。
提示：评估页面包含“优化模式/兼容模式”，中大规模数据优先使用优化模式。
"""
)

st.markdown("### 流程完成度")
workflow_status = build_workflow_status()
for step_name, done, detail in workflow_status:
    icon = "✅" if done else "⏳"
    st.markdown(f"- {icon} **{step_name}**：{detail}")

st.markdown("### 页面导航")
page_links = [
    ("数据处理（预处理）", "pages/0_数据处理.py"),
    ("基础成药性筛选", "pages/1_基础成药性筛选.py"),
    ("生成2D描述符", "pages/2_生成2D描述符.py"),
    ("生成3D构象", "pages/3-1_生成3D构象.py"),
    ("构象动力学优化", "pages/3-2_构象动力学优化.py"),
    ("生成3D描述符", "pages/3-3_生成3D描述符.py"),
    ("化合物多样性筛选", "pages/4_化合物多样性筛选.py"),
    ("结构多样性评估（指纹数据）", "pages/5_结构多样性评估.py"),
    ("Deubiquitinase Focused Library", "pages/7_Deubiquitinase_Focused_Library.py"),
    ("常用小工具", "pages/6_常用小工具.py"),
]

for start in range(0, len(page_links), 3):
    cols = st.columns(3)
    for col, (label, target) in zip(cols, page_links[start : start + 3]):
        with col:
            if st.button(label, key=f"go_{target}", use_container_width=True):
                st.switch_page(target)

st.markdown("### 运行环境状态")
env = detect_env_status()
env_col1, env_col2, env_col3, env_col4 = st.columns(4)
with env_col1:
    st.metric(
        "PyTorch/CUDA",
        "可用" if (env["torch_installed"] and env["cuda_available"]) else "不可用",
    )
with env_col2:
    st.metric(
        "FAISS",
        "GPU" if env["faiss_gpu"] else ("CPU" if env["faiss_installed"] else "未安装"),
    )
with env_col3:
    st.metric("CuPy", "已安装" if env["cupy_installed"] else "未安装")
with env_col4:
    st.metric("cuML", "已安装" if env["cuml_installed"] else "未安装")

# 添加应用程序信息
st.markdown(
    """
---
### 关于本系统

本系统基于 RDKit、Streamlit 和 scikit-learn 等开源工具开发，面向大规模分子库筛选场景。

#### 当前页面能力（前端）
- 数据清洗与 SMILES 校验
- 2D/3D 描述符与构象处理
- 多样性筛选（当前默认：最大最小距离贪心）
- 结构多样性评估（k-NN 统计、聚类、分布可视化）
"""
)

# 页面底部信息
st.sidebar.markdown("---")
st.sidebar.info(
    f"版本: {APP_VERSION}\n最近更新: {UPDATED_AT}\n本应用由 VS 开发团队开发。"
)
