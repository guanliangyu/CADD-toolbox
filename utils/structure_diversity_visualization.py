"""
结构多样性评估可视化与理化性质分布工具
"""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import streamlit as st
from scipy.stats import gaussian_kde

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False


DIVERSITY_METRIC_DISPLAY_ORDER = [
    "NN_Mean",
    "NN_Median",
    "NN_Std",
    "NN_Min",
    "NN_Max",
    "NN_Q25",
    "NN_Q75",
    "Pair_Mean",
    "Pair_Median",
    "Pair_Std",
    "Pair_Min",
    "Pair_Max",
    "Shannon_Entropy",
]

SCATTER_POINT_SIZE = 12
DISTRIBUTION_SCATTER_POINT_SIZE = 6
DISTRIBUTION_SCATTER_ALPHA = 0.35
MAX_PHYSCHEM_PLOT_ROWS = 30_000
DEFAULT_RANDOM_SEED = 42

PHYS_CHEM_PROPERTIES = ["MW", "AlogP", "HBD", "HBA", "TPSA", "RB"]
PHYSCHEM_PROPERTY_ALIASES = {
    "MW": ["mw", "molwt", "molecularweight", "molecular_weight", "molweight", "amw", "exactmass"],
    "AlogP": ["alogp", "logp", "clogp", "xlogp", "crippenlogp"],
    "HBD": ["hbd", "hbonddonor", "numhdonors", "hdonors", "hbond_donor"],
    "HBA": ["hba", "hbondacceptor", "numhacceptors", "hacceptors", "hbond_acceptor"],
    "TPSA": ["tpsa", "topologicalpolarsurfacearea", "polarsurfacearea", "psa"],
    "RB": ["rb", "rotatablebonds", "numrotatablebonds", "rotb", "nrotb"],
}
SMILES_COLUMN_ALIASES = [
    "smiles",
    "canonical_smiles",
    "canon_smiles",
    "isomeric_smiles",
    "input_smiles",
    "molecule_smiles",
    "structure_smiles",
]


def render_diversity_metrics_list(dataset_metrics: list[tuple[str, dict[str, float]]]) -> None:
    """以列表形式展示多数据集多样性指标（每行一个数据集）"""
    st.markdown("**多样性指标列表**")

    for dataset_name, metrics in dataset_metrics:
        if not metrics:
            st.markdown(f"- **{dataset_name}**：暂无可用指标")
            continue

        ordered_keys = [key for key in DIVERSITY_METRIC_DISPLAY_ORDER if key in metrics]
        remaining_keys = [key for key in metrics.keys() if key not in DIVERSITY_METRIC_DISPLAY_ORDER]
        all_keys = ordered_keys + remaining_keys

        metric_text = " | ".join([f"{key}={metrics[key]:.4f}" for key in all_keys])
        st.markdown(f"- **{dataset_name}**：{metric_text}")


def plot_nearest_neighbor_distribution(sim_matrix=None, knn_sim=None, title="Nearest Neighbor Distribution"):
    """绘制最近邻分布（支持完整矩阵或k-NN数据）"""
    if sim_matrix is not None:
        np.fill_diagonal(sim_matrix, 0)
        nearest_neighbors = np.max(sim_matrix, axis=1)
    elif knn_sim is not None:
        nearest_neighbors = np.max(knn_sim, axis=1)
    else:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(nearest_neighbors, bins=30, alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Nearest Neighbor Similarity")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Nearest Neighbor Similarity Distribution")
    ax1.grid(True, alpha=0.3)

    sorted_nn = np.sort(nearest_neighbors)
    y = np.arange(1, len(sorted_nn) + 1) / len(sorted_nn)
    ax2.plot(sorted_nn, y, linewidth=2)
    ax2.set_xlabel("Nearest Neighbor Similarity")
    ax2.set_ylabel("Cumulative Probability")
    ax2.set_title("Cumulative Distribution")
    ax2.grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def _normalize_column_name(name: str) -> str:
    """标准化列名，便于跨数据源匹配同义字段"""
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _find_property_column(
    meta_df: pd.DataFrame,
    aliases: list[str],
    prefer_numeric: bool = True,
) -> str | None:
    """在元数据中查找指定性质列（支持同义名匹配，可优先选数值有效列）"""
    if meta_df is None or meta_df.empty:
        return None

    normalized_to_original = {}
    for col in meta_df.columns:
        normalized = _normalize_column_name(col)
        if normalized and normalized not in normalized_to_original:
            normalized_to_original[normalized] = col

    norm_aliases = [_normalize_column_name(alias) for alias in aliases if alias]
    candidates = []

    for alias in norm_aliases:
        if alias in normalized_to_original:
            candidates.append(normalized_to_original[alias])

    for col in meta_df.columns:
        normalized_col = _normalize_column_name(col)
        for alias in norm_aliases:
            if alias and alias in normalized_col:
                candidates.append(col)
                break

    seen = set()
    dedup_candidates = []
    for col in candidates:
        if col not in seen:
            seen.add(col)
            dedup_candidates.append(col)

    if not dedup_candidates:
        return None
    if not prefer_numeric:
        return dedup_candidates[0]

    best_col = dedup_candidates[0]
    best_count = -1
    for col in dedup_candidates:
        numeric_count = int(pd.to_numeric(meta_df[col], errors="coerce").notna().sum())
        if numeric_count > best_count:
            best_count = numeric_count
            best_col = col
    return best_col


def _find_smiles_column(meta_df: pd.DataFrame) -> str | None:
    """自动匹配SMILES列名"""
    if meta_df is None or meta_df.empty:
        return None

    aliases = SMILES_COLUMN_ALIASES + ["smiles"]
    return _find_property_column(meta_df, aliases, prefer_numeric=False)


def _prepare_numeric_series(
    meta_df: pd.DataFrame,
    column_name: str,
    max_points: int = MAX_PHYSCHEM_PLOT_ROWS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.Series:
    """提取并清洗数值列，必要时下采样以控制绘图开销"""
    series = pd.to_numeric(meta_df[column_name], errors="coerce")
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) > max_points:
        series = series.sample(n=max_points, random_state=random_seed)
    return series.astype(float)


def subset_meta_with_indices(
    meta_df: pd.DataFrame | None,
    selected_indices: np.ndarray | None,
) -> pd.DataFrame | None:
    """根据已抽样索引对元数据对齐子集（若可对齐）"""
    if meta_df is None or selected_indices is None:
        return meta_df
    if not isinstance(meta_df, pd.DataFrame) or meta_df.empty:
        return meta_df
    if len(selected_indices) == 0:
        return meta_df.iloc[0:0]

    max_idx = int(np.max(selected_indices))
    if max_idx >= len(meta_df):
        return meta_df

    try:
        return meta_df.iloc[selected_indices].reset_index(drop=True)
    except Exception:
        return meta_df


def _prepare_meta_for_physchem_plot(
    meta_df: pd.DataFrame | None,
    max_rows: int = MAX_PHYSCHEM_PLOT_ROWS,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame | None:
    """限制理化性质绘图使用的元数据规模，避免大数据集计算过慢"""
    if meta_df is None or meta_df.empty:
        return meta_df
    if len(meta_df) <= max_rows:
        return meta_df.reset_index(drop=True).copy()
    sampled = meta_df.sample(n=max_rows, random_state=random_seed).reset_index(drop=True).copy()
    return sampled


def _calculate_physchem_from_smiles(smiles: str) -> dict[str, float]:
    """从SMILES计算标准理化性质，失败时返回NaN"""
    values = {prop: np.nan for prop in PHYS_CHEM_PROPERTIES}
    if not RDKIT_AVAILABLE:
        return values
    if smiles is None:
        return values

    smiles_text = str(smiles).strip()
    if not smiles_text or smiles_text.lower() == "nan":
        return values

    try:
        mol = Chem.MolFromSmiles(smiles_text)
        if mol is None:
            return values
        values["MW"] = float(Descriptors.MolWt(mol))
        values["AlogP"] = float(Descriptors.MolLogP(mol))
        values["HBD"] = float(Lipinski.NumHDonors(mol))
        values["HBA"] = float(Lipinski.NumHAcceptors(mol))
        values["TPSA"] = float(rdMolDescriptors.CalcTPSA(mol))
        values["RB"] = float(Lipinski.NumRotatableBonds(mol))
    except Exception:
        return values
    return values


def _ensure_physchem_columns(
    meta_df: pd.DataFrame,
    dataset_label: str,
) -> tuple[pd.DataFrame, dict]:
    """若缺失目标理化性质列，尝试基于SMILES自动补算"""
    info = {
        "computed": [],
        "unresolved": [],
        "smiles_col": None,
        "reason": None,
    }

    if meta_df is None or meta_df.empty:
        info["unresolved"] = list(PHYS_CHEM_PROPERTIES)
        info["reason"] = "empty_meta"
        return meta_df, info

    working_df = meta_df.copy()
    missing_props = []

    for prop in PHYS_CHEM_PROPERTIES:
        aliases = [prop, f"calc_{prop}", f"{prop}_calc"] + PHYSCHEM_PROPERTY_ALIASES.get(prop, [])
        col = _find_property_column(working_df, aliases, prefer_numeric=True)
        if col is None:
            missing_props.append(prop)
            continue
        valid_numeric = int(pd.to_numeric(working_df[col], errors="coerce").notna().sum())
        if valid_numeric == 0:
            missing_props.append(prop)

    if not missing_props:
        return working_df, info

    if not RDKIT_AVAILABLE:
        info["unresolved"] = missing_props
        info["reason"] = "rdkit_unavailable"
        return working_df, info

    smiles_col = _find_smiles_column(working_df)
    if not smiles_col:
        info["unresolved"] = missing_props
        info["reason"] = "smiles_missing"
        return working_df, info

    info["smiles_col"] = smiles_col
    st.info(f"🧮 {dataset_label} 缺失性质 {', '.join(missing_props)}，正在基于 `{smiles_col}` 自动计算...")

    calc_values = {prop: [] for prop in missing_props}
    for smiles in working_df[smiles_col]:
        calc_row = _calculate_physchem_from_smiles(smiles)
        for prop in missing_props:
            calc_values[prop].append(calc_row.get(prop, np.nan))

    for prop in missing_props:
        calc_col = f"calc_{prop}"
        working_df[calc_col] = calc_values[prop]
        valid_count = int(pd.to_numeric(working_df[calc_col], errors="coerce").notna().sum())
        if valid_count > 0:
            info["computed"].append(prop)
        else:
            info["unresolved"].append(prop)

    return working_df, info


def plot_physchem_distribution_comparison(
    meta_A: pd.DataFrame,
    meta_B: pd.DataFrame,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[plt.Figure | None, dict]:
    """绘制 MW/AlogP/HBD/HBA/TPSA/RB 两库分布对比图"""
    if meta_A is None or meta_B is None or meta_A.empty or meta_B.empty:
        return None, {"matched": {}, "missing": list(PHYS_CHEM_PROPERTIES)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    matched = {}
    missing = []
    discrete_properties = {"HBD", "HBA", "RB"}

    for i, prop in enumerate(PHYS_CHEM_PROPERTIES):
        ax = axes[i]
        aliases = [prop, f"calc_{prop}", f"{prop}_calc"] + PHYSCHEM_PROPERTY_ALIASES.get(prop, [])
        col_A = _find_property_column(meta_A, aliases, prefer_numeric=True)
        col_B = _find_property_column(meta_B, aliases, prefer_numeric=True)

        if not col_A or not col_B:
            missing.append(prop)
            ax.text(0.5, 0.5, f"{prop}\\n缺少可匹配列", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{prop} Distribution")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        values_A = _prepare_numeric_series(meta_A, col_A, random_seed=random_seed)
        values_B = _prepare_numeric_series(meta_B, col_B, random_seed=random_seed)
        if values_A.empty or values_B.empty:
            missing.append(prop)
            ax.text(0.5, 0.5, f"{prop}\\n有效数值不足", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{prop} Distribution")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        matched[prop] = (col_A, col_B)

        if prop in discrete_properties:
            value_min = int(np.floor(min(values_A.min(), values_B.min())))
            value_max = int(np.ceil(max(values_A.max(), values_B.max())))
            if value_max - value_min <= 60:
                bins = np.arange(value_min - 0.5, value_max + 1.5, 1.0)
            else:
                bins = 30
        else:
            bins = 40

        sns.histplot(
            values_A,
            bins=bins,
            stat="count",
            element="bars",
            fill=True,
            alpha=0.25,
            color="blue",
            label="Dataset A",
            ax=ax,
        )
        sns.histplot(
            values_B,
            bins=bins,
            stat="count",
            element="bars",
            fill=True,
            alpha=0.25,
            color="orange",
            label="Dataset B",
            ax=ax,
        )

        ax.set_title(f"{prop} Distribution")
        ax.set_xlabel(prop)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False, fontsize=8)

    plt.tight_layout()
    return fig, {"matched": matched, "missing": missing}


def plot_physchem_distribution_single(
    meta_df: pd.DataFrame,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> tuple[plt.Figure | None, dict]:
    """绘制单库 MW/AlogP/HBD/HBA/TPSA/RB 分布图"""
    if meta_df is None or meta_df.empty:
        return None, {"matched": {}, "missing": list(PHYS_CHEM_PROPERTIES)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.ravel()

    matched = {}
    missing = []
    discrete_properties = {"HBD", "HBA", "RB"}

    for i, prop in enumerate(PHYS_CHEM_PROPERTIES):
        ax = axes[i]
        aliases = [prop, f"calc_{prop}", f"{prop}_calc"] + PHYSCHEM_PROPERTY_ALIASES.get(prop, [])
        col = _find_property_column(meta_df, aliases, prefer_numeric=True)

        if not col:
            missing.append(prop)
            ax.text(0.5, 0.5, f"{prop}\n缺少可匹配列", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{prop} Distribution")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        values = _prepare_numeric_series(meta_df, col, random_seed=random_seed)
        if values.empty:
            missing.append(prop)
            ax.text(0.5, 0.5, f"{prop}\n有效数值不足", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{prop} Distribution")
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        matched[prop] = col

        if prop in discrete_properties:
            value_min = int(np.floor(values.min()))
            value_max = int(np.ceil(values.max()))
            if value_max - value_min <= 60:
                bins = np.arange(value_min - 0.5, value_max + 1.5, 1.0)
            else:
                bins = 30
        else:
            bins = 40

        sns.histplot(
            values,
            bins=bins,
            stat="count",
            element="bars",
            fill=True,
            alpha=0.30,
            color="blue",
            ax=ax,
        )

        ax.set_title(f"{prop} Distribution")
        ax.set_xlabel(prop)
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    return fig, {"matched": matched, "missing": missing}


def render_physchem_distribution_single(
    meta_df: pd.DataFrame | None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """渲染单库 MW/AlogP/HBD/HBA/TPSA/RB 分布区域"""
    st.markdown("#### 🧪 数据集A理化性质分布 (MW/AlogP/HBD/HBA/TPSA/RB)")

    if meta_df is None or meta_df.empty:
        st.info(
            "当前未加载可用元数据，无法绘制理化性质分布。可在“数据读取设置”中选择“采样读取元数据”或“完整读取元数据”。"
        )
        return

    meta_plot = _prepare_meta_for_physchem_plot(meta_df, max_rows=MAX_PHYSCHEM_PLOT_ROWS, random_seed=random_seed)
    meta_ready, calc_info = _ensure_physchem_columns(meta_plot, "数据集A")

    if calc_info.get("computed"):
        st.caption(f"数据集A已自动补算: {', '.join(calc_info['computed'])}")

    if calc_info.get("unresolved"):
        reason = calc_info.get("reason")
        if reason == "smiles_missing":
            st.caption(f"数据集A仍无法补算: {', '.join(calc_info['unresolved'])}（未找到SMILES列）")
        elif reason == "rdkit_unavailable":
            st.caption(f"数据集A仍无法补算: {', '.join(calc_info['unresolved'])}（RDKit不可用）")
        else:
            st.caption(f"数据集A仍无法补算: {', '.join(calc_info['unresolved'])}（SMILES无效或无法解析）")

    fig, summary = plot_physchem_distribution_single(meta_ready, random_seed=random_seed)
    if fig is None:
        st.info("未找到可用的理化性质列，无法绘图。")
        return

    st.pyplot(fig)
    plt.close(fig)

    matched = summary.get("matched", {})
    missing = summary.get("missing", [])

    if matched:
        matched_text = "；".join([f"{prop}: A[{col}]" for prop, col in matched.items()])
        st.caption(f"自动匹配列: {matched_text}")
    if missing:
        st.caption(f"以下性质未成功绘制（列缺失或数值不足）: {', '.join(missing)}")


def render_physchem_distribution_comparison(
    meta_A: pd.DataFrame | None,
    meta_B: pd.DataFrame | None,
    random_seed: int = DEFAULT_RANDOM_SEED,
) -> None:
    """渲染两库 MW/AlogP/HBD/HBA/TPSA/RB 分布对比区域"""
    st.markdown("#### 🧪 两库理化性质分布对比 (MW/AlogP/HBD/HBA/TPSA/RB)")

    if meta_A is None or meta_B is None or meta_A.empty or meta_B.empty:
        st.info(
            "当前未加载可用元数据，无法绘制理化性质分布。可在“数据读取设置”中选择“采样读取元数据”或“完整读取元数据”。"
        )
        return

    meta_A_plot = _prepare_meta_for_physchem_plot(meta_A, max_rows=MAX_PHYSCHEM_PLOT_ROWS, random_seed=random_seed)
    meta_B_plot = _prepare_meta_for_physchem_plot(meta_B, max_rows=MAX_PHYSCHEM_PLOT_ROWS, random_seed=random_seed)

    meta_A_ready, calc_info_A = _ensure_physchem_columns(meta_A_plot, "数据集A")
    meta_B_ready, calc_info_B = _ensure_physchem_columns(meta_B_plot, "数据集B")

    if calc_info_A.get("computed"):
        st.caption(f"数据集A已自动补算: {', '.join(calc_info_A['computed'])}")
    if calc_info_B.get("computed"):
        st.caption(f"数据集B已自动补算: {', '.join(calc_info_B['computed'])}")

    if calc_info_A.get("unresolved"):
        reason_a = calc_info_A.get("reason")
        if reason_a == "smiles_missing":
            st.caption(f"数据集A仍无法补算: {', '.join(calc_info_A['unresolved'])}（未找到SMILES列）")
        elif reason_a == "rdkit_unavailable":
            st.caption(f"数据集A仍无法补算: {', '.join(calc_info_A['unresolved'])}（RDKit不可用）")
        else:
            st.caption(f"数据集A仍无法补算: {', '.join(calc_info_A['unresolved'])}（SMILES无效或无法解析）")
    if calc_info_B.get("unresolved"):
        reason_b = calc_info_B.get("reason")
        if reason_b == "smiles_missing":
            st.caption(f"数据集B仍无法补算: {', '.join(calc_info_B['unresolved'])}（未找到SMILES列）")
        elif reason_b == "rdkit_unavailable":
            st.caption(f"数据集B仍无法补算: {', '.join(calc_info_B['unresolved'])}（RDKit不可用）")
        else:
            st.caption(f"数据集B仍无法补算: {', '.join(calc_info_B['unresolved'])}（SMILES无效或无法解析）")

    fig, summary = plot_physchem_distribution_comparison(meta_A_ready, meta_B_ready, random_seed=random_seed)
    if fig is None:
        st.info("未找到可用的理化性质列，无法绘图。")
        return

    st.pyplot(fig)
    plt.close(fig)

    matched = summary.get("matched", {})
    missing = summary.get("missing", [])

    if matched:
        matched_text = "；".join([f"{prop}: A[{col_a}] vs B[{col_b}]" for prop, (col_a, col_b) in matched.items()])
        st.caption(f"自动匹配列: {matched_text}")
    if missing:
        st.caption(f"以下性质未成功绘制（列缺失或数值不足）: {', '.join(missing)}")


def plot_clustering_results(clustering_results, title="Clustering Results", method="PCA-UMAP"):
    """绘制聚类结果"""
    coords = clustering_results["coords"]
    clusters = clustering_results["clusters"]
    cluster_method = clustering_results.get("cluster_method", "K-means")

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if "UMAP" in method:
        xlabel, ylabel = f"{method} Dimension 1", f"{method} Dimension 2"
    elif "t-SNE" in method:
        xlabel, ylabel = "t-SNE Dimension 1", "t-SNE Dimension 2"
    elif "PCA" in method:
        xlabel, ylabel = "PCA Dimension 1", "PCA Dimension 2"
    else:
        xlabel, ylabel = "Dimension 1", "Dimension 2"

    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=clusters,
        cmap="tab10",
        alpha=0.7,
        s=SCATTER_POINT_SIZE,
    )
    ax.set_title(f"{cluster_method} Clustering")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.colorbar(scatter, ax=ax, label="Cluster ID")

    if cluster_method == "DBSCAN":
        n_noise = int(np.sum(clusters == -1))
        n_clusters_found = len(set(clusters)) - (1 if -1 in clusters else 0)
        ax.text(
            0.02,
            0.02,
            f"簇数: {n_clusters_found} | 噪声: {n_noise}",
            transform=ax.transAxes,
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

    if hasattr(fig, "canvas"):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                fig.canvas.draw_idle()
        except Exception:
            pass

    plt.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig


def plot_distribution_comparison(coords_A, coords_B, metrics=None):
    """绘制分布对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(
        coords_A[:, 0],
        coords_A[:, 1],
        c="blue",
        alpha=DISTRIBUTION_SCATTER_ALPHA,
        s=DISTRIBUTION_SCATTER_POINT_SIZE,
        label="Dataset A",
    )
    ax1.scatter(
        coords_B[:, 0],
        coords_B[:, 1],
        c="orange",
        alpha=DISTRIBUTION_SCATTER_ALPHA,
        s=DISTRIBUTION_SCATTER_POINT_SIZE,
        label="Dataset B",
    )
    ax1.set_title("Structure Distribution Scatter Plot")
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")
    ax1.legend()

    x = np.concatenate([coords_A[:, 0], coords_B[:, 0]])
    y = np.concatenate([coords_A[:, 1], coords_B[:, 1]])

    if len(coords_A) > 1 and len(coords_B) > 1:
        xmin, xmax = x.min() - 1, x.max() + 1
        ymin, ymax = y.min() - 1, y.max() + 1
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])

        values_A = np.vstack([coords_A[:, 0], coords_A[:, 1]])
        values_B = np.vstack([coords_B[:, 0], coords_B[:, 1]])

        def _draw_contour(values: np.ndarray, color: str) -> bool:
            try:
                if np.all(np.std(values, axis=1) < 1e-12):
                    return False

                kernel = gaussian_kde(values)
                density = np.reshape(kernel(positions), xx.shape)
                density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
                if np.max(density) <= 0:
                    return False

                ax2.contour(xx, yy, density, levels=5, colors=color, alpha=0.5)
                return True
            except Exception:
                return False

        plotted_A = _draw_contour(values_A, "blue")
        plotted_B = _draw_contour(values_B, "orange")

        if plotted_A or plotted_B:
            from matplotlib.lines import Line2D

            legend_handles = []
            if plotted_A:
                legend_handles.append(Line2D([0], [0], color="blue", lw=2, label="Dataset A"))
            if plotted_B:
                legend_handles.append(Line2D([0], [0], color="orange", lw=2, label="Dataset B"))
            ax2.legend(handles=legend_handles)
        else:
            ax2.text(
                0.5,
                0.5,
                "Density plot not available\\n(insufficient data)",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
    else:
        ax2.text(
            0.5,
            0.5,
            "Density plot not available\\n(insufficient data)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    ax2.set_title("Structure Density Contour Plot")
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")

    plt.tight_layout()
    return fig


def plot_single_dataset_distribution(coords, dataset_name: str = "Dataset A"):
    """绘制单数据集分布图（散点 + 密度等高线）"""
    if coords is None or len(coords) == 0:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(
        coords[:, 0],
        coords[:, 1],
        c="blue",
        alpha=DISTRIBUTION_SCATTER_ALPHA,
        s=DISTRIBUTION_SCATTER_POINT_SIZE,
    )
    ax1.set_title("Structure Distribution Scatter Plot")
    ax1.set_xlabel("Dimension 1")
    ax1.set_ylabel("Dimension 2")

    if len(coords) > 1:
        x = coords[:, 0]
        y = coords[:, 1]
        xmin, xmax = x.min() - 1, x.max() + 1
        ymin, ymax = y.min() - 1, y.max() + 1
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        values = np.vstack([coords[:, 0], coords[:, 1]])

        plotted = False
        try:
            if not np.all(np.std(values, axis=1) < 1e-12):
                kernel = gaussian_kde(values)
                density = np.reshape(kernel(positions), xx.shape)
                density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
                if np.max(density) > 0:
                    ax2.contour(xx, yy, density, levels=5, colors="blue", alpha=0.5)
                    plotted = True
        except Exception:
            plotted = False

        if not plotted:
            ax2.text(
                0.5,
                0.5,
                "Density plot not available\n(insufficient data)",
                ha="center",
                va="center",
                transform=ax2.transAxes,
            )
    else:
        ax2.text(
            0.5,
            0.5,
            "Density plot not available\n(insufficient data)",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )

    ax2.set_title("Structure Density Contour Plot")
    ax2.set_xlabel("Dimension 1")
    ax2.set_ylabel("Dimension 2")

    plt.tight_layout()
    return fig


def monitor_memory_usage():
    """监控内存使用"""
    process = psutil.Process()
    memory_info = process.memory_info()
    return {
        "rss": memory_info.rss / 1024**2,
        "vms": memory_info.vms / 1024**2,
        "percent": process.memory_percent(),
    }
