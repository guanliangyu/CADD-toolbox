"""
CADD-Toolbox - 常用小工具
"""

import streamlit as st


st.set_page_config(page_title="常用小工具", page_icon="🧮", layout="wide")
st.title("🧮 常用小工具")

st.markdown("""
用于药物研发过程中的常见公式速查与快速计算。
""")


def _safe_div(numerator: float, denominator: float):
    """安全除法，避免除零"""
    if abs(denominator) < 1e-12:
        return None
    return numerator / denominator


st.subheader("1. 常用药物开发计算公式（速查）")
st.markdown(r"""
- 清除率（静脉给药）: `CL = Dose / AUC`
- 消除半衰期: `t_{1/2} = 0.693 / k_{el}`
- 分布容积: `V_d = CL / k_{el}`
- 绝对生物利用度: `F = (AUC_{po} \times Dose_{iv}) / (AUC_{iv} \times Dose_{po})`
- 血浆浓度单位换算（按分子量 MW）:
  - `nmol/L = (ng/mL × 1000) / MW`
  - `ng/mL = (nmol/L × MW) / 1000`
- 由称量质量配制目标浓度时所需体积:
  - `V(\mu L) = mass(mg) × 10^6 / (MW × C_{mM})`
""")
st.caption("提示：MW 单位 g/mol。")


st.subheader("2. 交互式计算")

calc_tab1, calc_tab2, calc_tab3, calc_tab4, calc_tab5, calc_tab6 = st.tabs([
    "浓度单位互转",
    "清除率 CL",
    "半衰期 t1/2",
    "分布容积 Vd",
    "生物利用度 F",
    "配液体积（mg→mM）",
])

with calc_tab1:
    col1, col2 = st.columns(2)
    with col1:
        mw = st.number_input("分子量 MW (g/mol)", min_value=0.000001, value=500.0, step=1.0)
        direction = st.radio(
            "换算方向",
            ["ng/mL → nmol/L", "nmol/L → ng/mL"],
            horizontal=True
        )
        if direction == "ng/mL → nmol/L":
            value = st.number_input("输入浓度 (ng/mL)", min_value=0.0, value=100.0, step=1.0)
        else:
            value = st.number_input("输入浓度 (nmol/L)", min_value=0.0, value=100.0, step=1.0)

    with col2:
        if direction == "ng/mL → nmol/L":
            converted = (value * 1000.0) / mw
            st.metric("换算结果 (nmol/L)", f"{converted:.6g}")
        else:
            converted = (value * mw) / 1000.0
            st.metric("换算结果 (ng/mL)", f"{converted:.6g}")

        st.info(
            f"当前换算关系：1 ng/mL = {1000.0 / mw:.6g} nmol/L，"
            f"1 nmol/L = {mw / 1000.0:.6g} ng/mL"
        )

with calc_tab2:
    col1, col2 = st.columns(2)
    with col1:
        dose = st.number_input("Dose", min_value=0.0, value=10.0, step=0.1, key="cl_dose")
        auc = st.number_input("AUC", min_value=0.0, value=2.0, step=0.1, key="cl_auc")
    with col2:
        cl = _safe_div(dose, auc)
        if cl is None:
            st.error("AUC 不能为 0")
        else:
            st.metric("CL = Dose / AUC", f"{cl:.6g}")
    st.caption("请确保 Dose 与 AUC 的单位匹配（例如 mg/kg 与 mg·h/L）。")

with calc_tab3:
    col1, col2 = st.columns(2)
    with col1:
        kel = st.number_input("消除速率常数 kel (1/h)", min_value=0.0, value=0.1, step=0.01)
    with col2:
        half_life = _safe_div(0.693, kel)
        if half_life is None:
            st.error("kel 不能为 0")
        else:
            st.metric("t1/2 = 0.693 / kel (h)", f"{half_life:.6g}")

with calc_tab4:
    col1, col2 = st.columns(2)
    with col1:
        cl_input = st.number_input("CL", min_value=0.0, value=5.0, step=0.1, key="vd_cl")
        kel_input = st.number_input("kel (1/h)", min_value=0.0, value=0.1, step=0.01, key="vd_kel")
    with col2:
        vd = _safe_div(cl_input, kel_input)
        if vd is None:
            st.error("kel 不能为 0")
        else:
            st.metric("Vd = CL / kel", f"{vd:.6g}")

with calc_tab5:
    col1, col2 = st.columns(2)
    with col1:
        auc_po = st.number_input("AUC_po", min_value=0.0, value=1.0, step=0.1)
        dose_po = st.number_input("Dose_po", min_value=0.000001, value=10.0, step=0.1)
        auc_iv = st.number_input("AUC_iv", min_value=0.000001, value=2.0, step=0.1)
        dose_iv = st.number_input("Dose_iv", min_value=0.000001, value=5.0, step=0.1)
    with col2:
        f_abs = _safe_div(auc_po * dose_iv, auc_iv * dose_po)
        if f_abs is None:
            st.error("AUC_iv × Dose_po 不能为 0")
        else:
            st.metric("绝对生物利用度 F", f"{f_abs * 100:.2f}%")
            if f_abs > 1:
                st.warning("F > 100%，请检查输入单位或实验条件。")

with calc_tab6:
    col1, col2 = st.columns(2)
    with col1:
        prep_mw = st.number_input(
            "分子量 MW (g/mol)",
            min_value=0.000001,
            value=500.0,
            step=1.0,
            key="prep_mw",
        )
        prep_mass_mg = st.number_input(
            "称量质量 (mg)",
            min_value=0.0,
            value=1.0,
            step=0.1,
            key="prep_mass_mg",
        )
        prep_conc_mm = st.number_input(
            "目标浓度 (mM)",
            min_value=0.000001,
            value=10.0,
            step=0.5,
            key="prep_conc_mm",
        )

    with col2:
        volume_ul = _safe_div(prep_mass_mg * 1_000_000.0, prep_mw * prep_conc_mm)
        if volume_ul is None:
            st.error("MW × 目标浓度 不能为 0")
        else:
            st.metric("需加入溶剂体积 (uL)", f"{volume_ul:.6g}")
            st.metric("需加入溶剂体积 (mL)", f"{(volume_ul / 1000.0):.6g}")
            st.caption(
                "假设全部样品溶解且忽略溶质体积贡献；"
                "计算关系：n(mmol)=mass(mg)/MW，V(L)=n/C。"
            )

st.markdown("---")
st.caption("该页面用于快速估算，不替代正式 PK/PD 建模与统计分析。")
