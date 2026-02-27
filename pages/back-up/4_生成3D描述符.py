"""
CADD-Toolbox - 3D分子描述符生成页面
使用Mordred库计算3D描述符，支持多构象聚合
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import multiprocessing

# RDKit imports
from rdkit import Chem

# Mordred imports
try:
    from mordred import Calculator, descriptors

    MORDRED_AVAILABLE = True
except ImportError:
    MORDRED_AVAILABLE = False

st.set_page_config(page_title="生成3D描述符", layout="wide")
st.title("🧬 生成3D分子描述符")

# 初始化会话状态
if "background_descriptor_tasks" not in st.session_state:
    st.session_state.background_descriptor_tasks = []


st.markdown(
    """
从优化后的SDF文件计算分子的3D描述符。支持多构象聚合策略，输出CSV格式结果。

💊 **Mordred描述符库**: 1800+ 种2D/3D分子描述符  
🔄 **多构象聚合**: 平均值、最大值、最小值、标准差  
📊 **批量处理**: 支持大规模分子库处理  
📝 **CSV输出**: 便于后续机器学习分析  
"""
)

# 检查Mordred可用性
if not MORDRED_AVAILABLE:
    st.error("❌ 未安装Mordred库！请运行：pip install mordred")
    st.code("pip install mordred", language="bash")
    st.stop()
else:
    st.success("✅ Mordred库已就绪")

# 导入后台描述符计算工具
try:
    from utils.background_descriptor import (
        run_background_descriptor_calculation,
        check_background_descriptor_status,
    )

    BACKGROUND_DESCRIPTOR_AVAILABLE = True
except ImportError:
    BACKGROUND_DESCRIPTOR_AVAILABLE = False
    st.warning("⚠️ 后台描述符计算工具不可用，使用传统多进程模式")

# 数据目录设置
DATA_DIR = "data"


def list_data_folders():
    """列出data目录下的所有文件夹"""
    if not os.path.exists(DATA_DIR):
        return []
    return [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]


def list_sdf_files_in_folder(folder_name):
    """列出指定文件夹中的所有SDF文件"""
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        return []
    return [f for f in os.listdir(folder_path) if f.endswith(".sdf")]


def get_file_info(file_path):
    """获取文件基本信息"""
    if not os.path.exists(file_path):
        return None

    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    mod_time = os.path.getmtime(file_path)
    mod_time_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")

    return {"size_mb": file_size, "modified": mod_time_str}


def count_molecules_in_sdf(file_path, max_count=10000):
    """快速统计SDF文件中的分子数量"""
    try:
        count = 0
        supplier = Chem.ForwardSDMolSupplier(file_path, removeHs=False, sanitize=False)

        for mol in supplier:
            if mol is not None:
                count += 1
            if count >= max_count:
                return f"{max_count}+"

        return count
    except Exception as e:
        return f"错误: {str(e)}"


def create_mordred_calculator(include_3d=True):
    """创建Mordred描述符计算器"""
    calc = Calculator(descriptors, ignore_3D=not include_3d)
    return calc


def create_batch_files(molecules_by_id, batch_size, output_dir, base_filename):
    """
    将大化合物库拆分成多个批次文件

    Args:
        molecules_by_id: 按分子ID分组的分子字典
        batch_size: 每批的分子数量
        output_dir: 输出目录
        base_filename: 基础文件名

    Returns:
        list: 批次文件路径列表
    """
    batch_files = []
    mol_ids = list(molecules_by_id.keys())

    st.info(
        f"🔄 开始创建 {(len(mol_ids) + batch_size - 1) // batch_size} 个批次文件..."
    )

    for batch_idx in range(0, len(mol_ids), batch_size):
        batch_mol_ids = mol_ids[batch_idx : batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        batch_filename = f"{base_filename}_batch_{batch_num:03d}.sdf"
        batch_filepath = os.path.join(output_dir, batch_filename)

        # 创建批次SDF文件
        writer = Chem.SDWriter(batch_filepath)
        mol_count = 0
        conformer_count = 0

        for mol_id in batch_mol_ids:
            mol_info = molecules_by_id[mol_id]
            mol = mol_info["mol"]

            # 恢复分子属性
            for prop_name, prop_value in mol_info["props"].items():
                mol.SetProp(prop_name, prop_value)

            # 写入每个构象
            for conf_id in range(mol.GetNumConformers()):
                mol_copy = Chem.Mol(mol)
                # 只保留当前构象
                mol_copy.RemoveAllConformers()
                conf = mol.GetConformer(conf_id)
                new_conf = Chem.Conformer(mol_copy.GetNumAtoms())
                for i in range(mol_copy.GetNumAtoms()):
                    new_conf.SetAtomPosition(i, conf.GetAtomPosition(i))
                mol_copy.AddConformer(new_conf, assignId=True)

                # 恢复属性
                for prop_name, prop_value in mol_info["props"].items():
                    mol_copy.SetProp(prop_name, prop_value)

                writer.write(mol_copy)
                conformer_count += 1

            mol_count += 1

        writer.close()
        batch_files.append(batch_filepath)

        # 记录批次信息
        st.info(
            f"📦 创建批次 {batch_num}: {batch_filename} ({mol_count} 个分子, {conformer_count} 个构象)"
        )

    st.success(f"✅ 成功创建 {len(batch_files)} 个批次文件")
    return batch_files


def merge_descriptor_results(batch_result_files, final_output_path):
    """
    合并多个批次的描述符结果文件

    Args:
        batch_result_files: 批次结果文件列表
        final_output_path: 最终合并输出文件路径

    Returns:
        tuple: (成功标志, 总分子数, 错误信息)
    """
    try:
        st.info("🔄 开始合并批次结果...")
        combined_dfs = []
        total_molecules = 0

        for batch_file in batch_result_files:
            if os.path.exists(batch_file):
                df = pd.read_csv(batch_file)
                combined_dfs.append(df)
                total_molecules += len(df)
                st.info(
                    f"✅ 批次结果: {os.path.basename(batch_file)} ({len(df)} 个分子)"
                )
            else:
                st.warning(f"⚠️ 批次结果文件不存在: {batch_file}")

        if combined_dfs:
            # 合并所有DataFrame
            st.info("🔄 合并所有批次数据...")
            final_df = pd.concat(combined_dfs, ignore_index=True)

            # 保存最终结果
            st.info(f"💾 保存最终结果到: {final_output_path}")
            final_df.to_csv(final_output_path, index=False)

            file_size = os.path.getsize(final_output_path) / (1024 * 1024)
            st.success(
                f"✅ 合并完成! 总计 {total_molecules} 个分子, 文件大小: {file_size:.1f} MB"
            )

            return True, total_molecules, None
        else:
            return False, 0, "没有找到有效的批次结果文件"

    except Exception as e:
        return False, 0, str(e)


def process_batch_intelligently(batch_files, output_dir, config):
    """
    智能批次处理 - 使用智能后台执行模式

    Args:
        batch_files: 批次文件列表
        output_dir: 输出目录
        config: 配置参数

    Returns:
        list: 批次任务信息列表
    """
    batch_tasks = []

    st.info(f"🚀 准备启动 {len(batch_files)} 个批次的智能后台计算...")

    for i, batch_file in enumerate(batch_files):
        batch_num = i + 1
        st.info(
            f"🔄 启动批次 {batch_num}/{len(batch_files)}: {os.path.basename(batch_file)}"
        )

        try:
            st.info(
                f"🔄 正在启动批次 {batch_num}，输入文件: {os.path.basename(batch_file)}"
            )

            # 调用智能后台描述符计算工具
            with st.spinner(f"启动批次 {batch_num}..."):
                script_file, script_name = run_background_descriptor_calculation(
                    input_file=os.path.abspath(batch_file),
                    output_dir=output_dir,
                    processing_limit=float("inf"),  # 处理批次中的所有分子
                    num_workers=config["num_workers"],
                    include_3d=config["include_3d"],
                    aggregation_method=config["aggregation_method"],
                    include_smiles=config["include_smiles"],
                    detached=True,
                )

            task_info = {
                "batch_num": batch_num,
                "script_name": script_name,
                "script_file": script_file,
                "batch_file": batch_file,
                "output_dir": output_dir,
                "start_time": time.time(),
                "config": config.copy(),
            }
            batch_tasks.append(task_info)

            st.success(f"✅ 批次 {batch_num} 已启动: {script_name}")

            # 显示脚本文件和日志文件路径
            log_file = os.path.join(output_dir, f"{script_name}_output.log")
            st.info(f"📄 脚本文件: {script_file}")
            st.info(f"📝 日志文件: {log_file}")

            # 短暂延迟避免系统负载过高
            time.sleep(2)

        except Exception as e:
            st.error(f"❌ 批次 {batch_num} 启动失败: {e}")
            import traceback

            st.code(traceback.format_exc(), language="text")
            continue

    st.success(f"🎉 成功启动 {len(batch_tasks)} 个批次任务!")
    return batch_tasks


def check_all_batches_completed(batch_files, output_dir):
    """
    检查所有批次是否都完成了

    Args:
        batch_files: 批次文件列表
        output_dir: 输出目录

    Returns:
        tuple: (是否全部完成, 完成的批次结果文件列表)
    """
    completed_results = []

    for i, batch_file in enumerate(batch_files):
        batch_num = i + 1
        expected_result = os.path.join(
            output_dir, f"descriptors_batch_{batch_num:03d}.csv"
        )

        if os.path.exists(expected_result):
            completed_results.append(expected_result)
        else:
            # 如果有任何一个批次未完成，返回False
            return False, completed_results

    return True, completed_results


def calculate_molecule_descriptors(mol, calc, conf_ids=None):
    """计算单个分子的描述符"""
    if mol is None:
        return None

    try:
        if conf_ids is None:
            # 计算所有构象
            conf_ids = list(range(mol.GetNumConformers()))

        if not conf_ids:
            # 没有构象，尝试计算2D描述符
            try:
                result = calc(mol)

                # 处理描述符值，确保数据类型正确
                desc_values = []
                for desc_val in result.values():
                    try:
                        if desc_val is None:
                            desc_values.append(np.nan)
                        elif isinstance(desc_val, (int, float)):
                            if np.isfinite(desc_val):
                                desc_values.append(float(desc_val))
                            else:
                                desc_values.append(np.nan)
                        elif hasattr(desc_val, "__float__"):
                            # 尝试转换为浮点数
                            try:
                                float_val = float(desc_val)
                                desc_values.append(
                                    float_val if np.isfinite(float_val) else np.nan
                                )
                            except (ValueError, TypeError, OverflowError):
                                desc_values.append(np.nan)
                        elif isinstance(desc_val, str):
                            # 字符串类型，尝试转换
                            if desc_val.lower() in [
                                "nan",
                                "inf",
                                "-inf",
                                "none",
                                "error",
                            ]:
                                desc_values.append(np.nan)
                            else:
                                try:
                                    num_val = float(desc_val)
                                    desc_values.append(
                                        num_val if np.isfinite(num_val) else np.nan
                                    )
                                except ValueError:
                                    desc_values.append(np.nan)
                        else:
                            # 其他类型转为NaN
                            desc_values.append(np.nan)
                    except Exception:
                        desc_values.append(np.nan)

                return [desc_values]
            except Exception as e:
                st.warning(f"2D描述符计算失败: {e}")
                return None

        conformer_descriptors = []
        for conf_id in conf_ids:
            try:
                # 创建只包含当前构象的分子副本
                mol_copy = Chem.Mol(mol)
                conf = mol.GetConformer(conf_id)
                new_conf = Chem.Conformer(mol_copy.GetNumAtoms())
                for i in range(mol_copy.GetNumAtoms()):
                    new_conf.SetAtomPosition(i, conf.GetAtomPosition(i))
                mol_copy.AddConformer(new_conf, assignId=True)

                # 计算该构象的描述符
                result = calc(mol_copy)

                # 处理描述符值，确保数据类型正确
                desc_values = []
                for desc_val in result.values():
                    try:
                        if desc_val is None:
                            desc_values.append(np.nan)
                        elif isinstance(desc_val, (int, float)):
                            if np.isfinite(desc_val):
                                desc_values.append(float(desc_val))
                            else:
                                desc_values.append(np.nan)
                        elif hasattr(desc_val, "__float__"):
                            # 尝试转换为浮点数
                            try:
                                float_val = float(desc_val)
                                desc_values.append(
                                    float_val if np.isfinite(float_val) else np.nan
                                )
                            except (ValueError, TypeError, OverflowError):
                                desc_values.append(np.nan)
                        elif isinstance(desc_val, str):
                            # 字符串类型，尝试转换
                            if desc_val.lower() in [
                                "nan",
                                "inf",
                                "-inf",
                                "none",
                                "error",
                            ]:
                                desc_values.append(np.nan)
                            else:
                                try:
                                    num_val = float(desc_val)
                                    desc_values.append(
                                        num_val if np.isfinite(num_val) else np.nan
                                    )
                                except ValueError:
                                    desc_values.append(np.nan)
                        else:
                            # 其他类型转为NaN
                            desc_values.append(np.nan)
                    except Exception:
                        desc_values.append(np.nan)

                conformer_descriptors.append(desc_values)
            except Exception as e:
                st.warning(f"构象 {conf_id} 描述符计算失败: {e}")
                continue

        return conformer_descriptors if conformer_descriptors else None

    except Exception as e:
        st.error(f"分子描述符计算失败: {e}")
        return None


def aggregate_conformer_descriptors(conformer_descriptors, method="mean"):
    """聚合多个构象的描述符"""
    if not conformer_descriptors:
        return None

    try:
        # 预处理：将所有值转换为数值类型
        processed_descriptors = []
        for conf_desc in conformer_descriptors:
            processed_conf = []
            for desc_value in conf_desc:
                try:
                    # 尝试转换为浮点数
                    if desc_value is None:
                        processed_conf.append(np.nan)
                    elif isinstance(desc_value, (int, float)):
                        if np.isfinite(desc_value):
                            processed_conf.append(float(desc_value))
                        else:
                            processed_conf.append(np.nan)
                    elif isinstance(desc_value, str):
                        # 字符串类型，尝试转换
                        if desc_value.lower() in [
                            "nan",
                            "inf",
                            "-inf",
                            "none",
                            "error",
                        ]:
                            processed_conf.append(np.nan)
                        else:
                            try:
                                num_val = float(desc_value)
                                processed_conf.append(
                                    num_val if np.isfinite(num_val) else np.nan
                                )
                            except ValueError:
                                processed_conf.append(np.nan)
                    else:
                        # 其他类型，转为NaN
                        processed_conf.append(np.nan)
                except Exception:
                    processed_conf.append(np.nan)

            processed_descriptors.append(processed_conf)

        # 转换为numpy数组
        descriptors_array = np.array(processed_descriptors, dtype=float)

        # 聚合计算
        if method == "mean":
            result = np.nanmean(descriptors_array, axis=0)
        elif method == "max":
            result = np.nanmax(descriptors_array, axis=0)
        elif method == "min":
            result = np.nanmin(descriptors_array, axis=0)
        elif method == "std":
            result = np.nanstd(descriptors_array, axis=0)
        elif method == "median":
            result = np.nanmedian(descriptors_array, axis=0)
        else:
            result = np.nanmean(descriptors_array, axis=0)

        return result

    except Exception as e:
        st.error(f"聚合描述符时出错: {e}")
        # 返回NaN数组作为备选
        if conformer_descriptors and len(conformer_descriptors) > 0:
            return np.full(len(conformer_descriptors[0]), np.nan)
        else:
            return None


# 文件选择界面
st.subheader("1. 选择输入文件")

# 获取可用文件夹
folders = list_data_folders()

if not folders:
    st.warning("data目录下没有找到任何文件夹")
else:
    selected_folder = st.selectbox("选择数据文件夹:", folders)

    if selected_folder:
        # 获取该文件夹中的SDF文件
        sdf_files = list_sdf_files_in_folder(selected_folder)

        if not sdf_files:
            st.warning(f"文件夹 {selected_folder} 中没有SDF文件")
        else:
            selected_file = st.selectbox("选择SDF文件:", sdf_files)

            if selected_file:
                file_path = os.path.join(DATA_DIR, selected_folder, selected_file)

                # 显示文件信息
                file_info = get_file_info(file_path)
                if file_info:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("文件大小", f"{file_info['size_mb']:.1f} MB")
                    with col2:
                        st.metric("修改时间", file_info["modified"])
                    with col3:
                        with st.spinner("统计分子数..."):
                            mol_count = count_molecules_in_sdf(file_path)
                        st.metric("分子数量", mol_count)

                # 描述符计算配置
                st.subheader("2. 描述符计算配置")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**描述符类型**")
                    include_3d = st.checkbox(
                        "包含3D描述符", value=True, help="需要分子具有3D坐标"
                    )

                    st.markdown("**处理范围**")
                    processing_option = st.selectbox(
                        "选择处理范围:",
                        ["处理所有分子", "仅处理前1000个分子（测试用）"],
                        index=0,
                        help="默认处理所有分子，支持百万级化合物库；测试选项用于快速验证",
                    )

                    if processing_option == "仅处理前1000个分子（测试用）":
                        processing_limit = 1000
                        st.info("🧪 测试模式: 仅处理前1000个分子")
                    else:
                        processing_limit = float("inf")  # 不设限制，处理所有分子
                        st.info("🚀 生产模式: 处理文件中的所有分子（支持百万级）")

                with col2:
                    st.markdown("**多构象聚合策略**")
                    aggregation_method = st.selectbox(
                        "聚合方法:",
                        ["mean", "max", "min", "std", "median"],
                        help="如何聚合同一分子多个构象的描述符",
                    )

                    include_smiles = st.checkbox(
                        "包含SMILES", value=True, help="在输出中包含分子SMILES"
                    )

                # 执行配置
                st.subheader("3. 执行配置")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**计算模式**")
                    if BACKGROUND_DESCRIPTOR_AVAILABLE:
                        execution_mode = "智能后台执行"
                        st.success("✅ 智能后台执行 (断点恢复，不受页面刷新影响)")
                    else:
                        st.error("❌ 智能后台执行不可用")
                        st.stop()

                with col2:
                    st.markdown("**并行配置**")
                    max_workers = multiprocessing.cpu_count()
                    num_workers = st.number_input(
                        "并行进程数",
                        min_value=1,
                        max_value=max_workers * 2,  # 允许超线程
                        value=34,
                        step=1,
                        help=f"系统有{max_workers}个CPU核心，推荐34进程",
                    )
                    st.info(f"🚀 将使用{num_workers}个进程")

                # 输出配置
                st.subheader("4. 输出配置")

                output_filename = st.text_input(
                    "输出文件名",
                    value=f"descriptors_{selected_file.replace('.sdf', '')}.csv",
                )

                # 计算按钮
                if st.button("🚀 开始计算描述符", type="primary"):
                    # 检查是否已有正在运行的任务
                    running_tasks = []
                    for task in st.session_state.background_descriptor_tasks:
                        if BACKGROUND_DESCRIPTOR_AVAILABLE:
                            status_info = check_background_descriptor_status(
                                task["output_dir"], task["script_name"]
                            )
                            if status_info["status"] == "running":
                                running_tasks.append(task)

                    if running_tasks:
                        st.warning(f"⚠️ 已有 {len(running_tasks)} 个任务正在运行中！")
                        st.info(
                            "请在下方监控区域查看进度，或先停止当前任务再启动新任务。"
                        )
                        for task in running_tasks:
                            st.info(f"- 运行中任务: {task['script_name']}")
                        st.stop()

                    st.info(f"开始处理文件: {file_path}")

                    # 创建输出路径
                    output_path = os.path.join(
                        DATA_DIR, selected_folder, output_filename
                    )

                    try:
                        # 创建Mordred计算器
                        with st.spinner("初始化Mordred计算器..."):
                            calc = create_mordred_calculator(include_3d=include_3d)
                            descriptor_names = [str(d) for d in calc.descriptors]

                        st.success(
                            f"✅ 计算器就绪，共 {len(descriptor_names)} 个描述符"
                        )

                        # 读取SDF文件并按分子ID分组构象
                        with st.spinner("读取SDF文件并按分子ID分组..."):
                            supplier = Chem.ForwardSDMolSupplier(
                                file_path, removeHs=False, sanitize=True
                            )
                            molecules_by_id = {}

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            count = 0
                            for mol in supplier:
                                if (
                                    processing_limit != float("inf")
                                    and count >= processing_limit
                                ):
                                    break

                                if mol is not None:
                                    # 获取分子ID/名称
                                    mol_id = (
                                        mol.GetProp("_Name")
                                        if mol.HasProp("_Name")
                                        else f"mol_{count}"
                                    )
                                    if not mol_id.strip():  # 如果_Name为空
                                        mol_id = (
                                            mol.GetProp("IDNUMBER")
                                            if mol.HasProp("IDNUMBER")
                                            else f"mol_{count}"
                                        )

                                    # 提取分子属性以避免MolBlock转换时丢失
                                    mol_props = {
                                        prop_name: mol.GetProp(prop_name)
                                        for prop_name in mol.GetPropNames()
                                    }

                                    # 按分子ID分组 - 合并同一分子的多个构象
                                    if mol_id not in molecules_by_id:
                                        molecules_by_id[mol_id] = {
                                            "mol": Chem.Mol(mol),  # 创建分子副本
                                            "props": mol_props,
                                            "conformer_count": 0,
                                        }
                                        # 清除已有构象
                                        molecules_by_id[mol_id][
                                            "mol"
                                        ].RemoveAllConformers()

                                    # 添加当前构象到分子对象
                                    for conf_id in range(mol.GetNumConformers()):
                                        conf = mol.GetConformer(conf_id)
                                        new_conf = Chem.Conformer(
                                            molecules_by_id[mol_id]["mol"].GetNumAtoms()
                                        )
                                        for i in range(
                                            molecules_by_id[mol_id]["mol"].GetNumAtoms()
                                        ):
                                            new_conf.SetAtomPosition(
                                                i, conf.GetAtomPosition(i)
                                            )
                                        molecules_by_id[mol_id]["mol"].AddConformer(
                                            new_conf, assignId=True
                                        )
                                        molecules_by_id[mol_id]["conformer_count"] += 1

                                    count += 1

                                    if count % 100 == 0:
                                        progress = (
                                            min(count / processing_limit, 1.0)
                                            if processing_limit != float("inf")
                                            else count / 10000
                                        )
                                        progress_bar.progress(min(progress, 1.0))
                                        status_text.text(
                                            f"已读取 {count} 个分子，分组为 {len(molecules_by_id)} 个独特分子..."
                                        )

                            # 转换为列表格式供后续处理
                            molecules = list(molecules_by_id.values())
                            smiles_list = []

                            # 为每个分子生成SMILES
                            if include_smiles:
                                for mol_info in molecules:
                                    try:
                                        smiles = Chem.MolToSmiles(mol_info["mol"])
                                        smiles_list.append(smiles)
                                    except Exception:
                                        smiles_list.append("Invalid")

                        total_conformers = sum(
                            mol_info["conformer_count"] for mol_info in molecules
                        )
                        st.success(
                            f"✅ 成功读取并分组：{len(molecules)} 个独特分子，共 {total_conformers} 个构象"
                        )

                        # 检查是否需要拆分大化合物库
                        BATCH_SIZE = 100000  # 每批10万个分子
                        need_batch_processing = len(
                            molecules
                        ) > BATCH_SIZE and processing_limit == float("inf")

                        if need_batch_processing:
                            st.warning(
                                f"🔄 检测到大化合物库 ({len(molecules)} 个分子)，将拆分成批次处理"
                            )
                            st.info(
                                f"📦 每批处理 {BATCH_SIZE:,} 个分子，共需 {(len(molecules) + BATCH_SIZE - 1) // BATCH_SIZE} 批次"
                            )

                            # 询问用户是否继续
                            proceed_batch = st.button(
                                "✅ 确认开始批次处理",
                                type="primary",
                                key="confirm_batch",
                            )
                            if not proceed_batch:
                                st.stop()

                        # 智能后台执行
                        st.subheader("5. 智能后台计算")
                        st.info(
                            f"🚀 使用{num_workers}个进程并行计算{len(molecules)}个分子的描述符"
                        )

                        # 确定输出目录
                        abs_work_dir = os.path.abspath(
                            os.path.join(DATA_DIR, selected_folder)
                        )

                        if need_batch_processing:
                            # 大化合物库批次处理模式
                            st.subheader("📦 大化合物库批次处理")

                            try:
                                # 创建批次文件
                                base_filename = selected_file.replace(".sdf", "")
                                batch_files = create_batch_files(
                                    molecules_by_id,
                                    BATCH_SIZE,
                                    abs_work_dir,
                                    base_filename,
                                )

                                # 启动批次处理
                                batch_config = {
                                    "num_workers": num_workers,
                                    "include_3d": include_3d,
                                    "aggregation_method": aggregation_method,
                                    "include_smiles": include_smiles,
                                }

                                batch_tasks = process_batch_intelligently(
                                    batch_files, abs_work_dir, batch_config
                                )

                                # 保存批次任务信息到session state
                                batch_info = {
                                    "type": "batch_intelligent",
                                    "batch_files": batch_files,
                                    "batch_tasks": batch_tasks,
                                    "output_dir": abs_work_dir,
                                    "final_output_file": output_path,
                                    "start_time": time.time(),
                                    "total_molecules": len(molecules),
                                    "batch_size": BATCH_SIZE,
                                }

                                # 添加到智能后台任务列表
                                for task in batch_tasks:
                                    st.session_state.background_descriptor_tasks.append(
                                        task
                                    )

                                st.success("🎉 批次任务已添加到后台任务列表！")
                                st.info("💡 所有批次完成后，请手动合并结果文件")

                                # 提供合并指令
                                st.subheader("📋 批次完成后的合并指令")
                                st.info(
                                    "当所有批次完成后，可以使用以下Python代码合并结果："
                                )

                                merge_code = f"""
import pandas as pd
import os

# 批次结果文件
batch_files = {[os.path.join(abs_work_dir, f"descriptors_batch_{i+1:03d}.csv") for i in range(len(batch_files))]}

# 合并所有批次
combined_dfs = []
for batch_file in batch_files:
    if os.path.exists(batch_file):
        df = pd.read_csv(batch_file)
        combined_dfs.append(df)
        print(f"加载: {{os.path.basename(batch_file)}} ({{len(df)}} 个分子)")

if combined_dfs:
    final_df = pd.concat(combined_dfs, ignore_index=True)
    final_df.to_csv('{output_path}', index=False)
    print(f"合并完成: {{len(final_df)}} 个分子 -> {output_path}")
else:
    print("未找到批次结果文件")
"""
                                st.code(merge_code, language="python")

                            except Exception as e:
                                st.error(f"❌ 启动批次处理失败: {e}")
                                st.code(str(e), language="text")

                        else:
                            # 标准智能后台执行
                            st.info("🚀 准备启动智能后台描述符计算...")

                            # 确定输入文件路径
                            abs_input_file_path = os.path.abspath(file_path)

                            try:
                                st.info(f"📂 输入文件: {abs_input_file_path}")
                                st.info(f"📁 输出目录: {abs_work_dir}")
                                st.info(
                                    f"⚙️ 配置: {num_workers}进程, {processing_limit}分子, {aggregation_method}聚合"
                                )

                                # 调用智能后台描述符计算工具
                                with st.spinner("启动智能后台计算..."):
                                    script_file, script_name = (
                                        run_background_descriptor_calculation(
                                            input_file=abs_input_file_path,
                                            output_dir=abs_work_dir,
                                            processing_limit=processing_limit,
                                            num_workers=num_workers,
                                            include_3d=include_3d,
                                            aggregation_method=aggregation_method,
                                            include_smiles=include_smiles,
                                            detached=True,
                                        )
                                    )

                                st.success("✅ 智能后台描述符计算已启动！")
                                st.info(f"📄 脚本文件: {script_file}")
                                st.info(f"🏷️ 任务名称: {script_name}")

                                # 显示日志文件路径
                                log_file = os.path.join(
                                    abs_work_dir, f"{script_name}_output.log"
                                )
                                st.info(f"📝 日志文件: {log_file}")

                                # 保存任务信息到session state
                                task_info = {
                                    "script_name": script_name,
                                    "script_file": script_file,
                                    "output_dir": abs_work_dir,
                                    "start_time": time.time(),
                                    "input_file": abs_input_file_path,
                                    "config": {
                                        "num_workers": num_workers,
                                        "processing_limit": processing_limit,
                                        "include_3d": include_3d,
                                        "aggregation_method": aggregation_method,
                                        "include_smiles": include_smiles,
                                    },
                                }
                                st.session_state.background_descriptor_tasks.append(
                                    task_info
                                )

                                st.success(
                                    "🎉 任务已添加到后台任务列表，可以安全关闭页面或刷新！"
                                )

                                # 提供监控命令
                                st.code(
                                    f"# 监控命令\ntail -f {log_file}", language="bash"
                                )

                                # 提供停止命令
                                st.code(
                                    f"# 停止命令\npkill -f {script_name}",
                                    language="bash",
                                )

                            except Exception as e:
                                st.error(f"❌ 启动智能后台描述符计算失败: {e}")
                                import traceback

                                st.code(traceback.format_exc(), language="text")

                    except Exception as e:
                        st.error(f"处理过程中出错: {e}")
                        import traceback

                        st.code(traceback.format_exc(), language="text")

# 智能后台任务监控区域
if BACKGROUND_DESCRIPTOR_AVAILABLE:
    st.header("🤖 智能后台任务监控")

    if st.session_state.background_descriptor_tasks:
        for idx, task in enumerate(st.session_state.background_descriptor_tasks):
            with st.expander(f"📋 任务 {idx+1}: {task['script_name']}", expanded=True):
                col1, col2, col3 = st.columns(3)

                with col1:
                    elapsed = time.time() - task["start_time"]
                    st.metric("运行时间", f"{elapsed/60:.1f} 分钟")

                with col2:
                    # 检查任务状态
                    status_info = check_background_descriptor_status(
                        task["output_dir"], task["script_name"]
                    )
                    status_display = {
                        "running": "🟢 运行中",
                        "completed": "✅ 已完成",
                        "error": "❌ 出错",
                        "unknown": "❓ 未知",
                    }
                    st.metric(
                        "状态", status_display.get(status_info["status"], "❓ 未知")
                    )

                with col3:
                    config = task["config"]
                    st.metric(
                        "配置",
                        f"{config['num_workers']}进程/{config['processing_limit']}分子",
                    )

                # 详细信息
                st.code(
                    f"""
任务名称: {task['script_name']}
输入文件: {task['input_file']}
输出目录: {task['output_dir']}
脚本文件: {task['script_file']}
聚合方法: {config['aggregation_method']}
包含3D: {config['include_3d']}
包含SMILES: {config['include_smiles']}
""",
                    language="text",
                )

                # 操作按钮
                col_btn1, col_btn2, col_btn3, col_btn4 = st.columns(4)

                with col_btn1:
                    if st.button(f"📖 查看日志 {idx+1}", key=f"desc_log_{idx}"):
                        log_file = os.path.join(
                            task["output_dir"], f"{task['script_name']}.log"
                        )
                        if os.path.exists(log_file):
                            try:
                                with open(log_file, "r", encoding="utf-8") as f:
                                    log_content = f.read()

                                # 只显示最后100行
                                log_lines = log_content.split("\n")
                                recent_logs = "\n".join(log_lines[-100:])
                                st.code(recent_logs, language="text")

                            except Exception as e:
                                st.error(f"读取日志失败: {e}")
                        else:
                            st.warning("日志文件不存在")

                with col_btn2:
                    if st.button(f"📊 检查结果 {idx+1}", key=f"desc_result_{idx}"):
                        if status_info["status"] == "completed":
                            st.success("🎉 任务已完成！")
                            st.code(status_info.get("details", ""), language="text")

                            # 查找输出文件
                            output_files = []
                            for file in os.listdir(task["output_dir"]):
                                if file.startswith("descriptors_") and file.endswith(
                                    ".csv"
                                ):
                                    output_files.append(file)

                            if output_files:
                                for output_file in output_files:
                                    full_path = os.path.join(
                                        task["output_dir"], output_file
                                    )
                                    file_size = os.path.getsize(full_path) / (
                                        1024 * 1024
                                    )
                                    st.info(
                                        f"📄 输出文件: {output_file} ({file_size:.1f} MB)"
                                    )

                                    # 提供下载按钮
                                    with open(full_path, "rb") as f:
                                        st.download_button(
                                            f"📥 下载 {output_file}",
                                            f.read(),
                                            file_name=output_file,
                                            mime="text/csv",
                                            key=f"desc_download_{idx}_{output_file}",
                                        )

                                    # 显示数据预览
                                    try:
                                        df = pd.read_csv(full_path)
                                        st.markdown("**结果统计:**")
                                        col_s1, col_s2, col_s3 = st.columns(3)
                                        with col_s1:
                                            st.metric("分子数", len(df))
                                        with col_s2:
                                            numeric_cols = df.select_dtypes(
                                                include=[np.number]
                                            ).columns
                                            st.metric("描述符数", len(numeric_cols))
                                        with col_s3:
                                            if len(numeric_cols) > 0:
                                                valid_desc = (
                                                    df[numeric_cols].notna().sum().sum()
                                                )
                                                total_desc = len(df) * len(numeric_cols)
                                                coverage = (
                                                    valid_desc / total_desc * 100
                                                    if total_desc > 0
                                                    else 0
                                                )
                                                st.metric(
                                                    "有效覆盖率", f"{coverage:.1f}%"
                                                )

                                        st.markdown("**数据预览 (前5行):**")
                                        st.dataframe(df.head(5))
                                    except Exception as e:
                                        st.warning(f"无法预览数据: {e}")
                            else:
                                st.warning("未找到输出文件")

                        elif status_info["status"] == "error":
                            st.error("❌ 任务执行出错")
                            st.code(status_info.get("details", ""), language="text")
                        elif status_info["status"] == "running":
                            if "memory_mb" in status_info:
                                st.info(
                                    f"🔄 任务运行中 (PID: {status_info['pid']}, 内存: {status_info['memory_mb']:.1f}MB)"
                                )
                            else:
                                st.info("🔄 任务运行中...")
                        else:
                            st.warning("❓ 任务状态未知")

                with col_btn3:
                    if st.button(f"⛔ 停止任务 {idx+1}", key=f"desc_stop_{idx}"):
                        if status_info["status"] == "running" and "pid" in status_info:
                            try:
                                import psutil

                                process = psutil.Process(int(status_info["pid"]))
                                process.terminate()
                                st.success(f"✅ 任务 {status_info['pid']} 已停止")
                            except Exception as e:
                                st.error(f"停止任务失败: {e}")
                        else:
                            st.warning("任务未在运行")

                with col_btn4:
                    if st.button(f"🗑️ 删除记录 {idx+1}", key=f"desc_delete_{idx}"):
                        st.session_state.background_descriptor_tasks.pop(idx)
                        st.success("✅ 任务记录已删除")
                        st.rerun()
    else:
        st.info("🔍 当前没有智能后台描述符计算任务")


# 分隔线
st.divider()

# 使用说明
with st.expander("📖 使用说明", expanded=False):
    st.markdown(
        """
    ### 功能说明
    
    1. **文件选择**: 从data目录中选择已优化的SDF文件
    2. **描述符配置**: 选择3D描述符和聚合策略
    3. **执行配置**: 选择计算模式和进程数
    4. **输出配置**: 设置输出文件名
    5. **描述符计算**: 使用Mordred库计算1800+种分子描述符
    6. **多构象处理**: 对同一分子的多个构象进行聚合
    7. **大化合物库批次处理**: 超过10万分子自动拆分为10万一批的子任务
    8. **结果输出**: 生成CSV格式的描述符矩阵
    
    ### 执行模式说明
    
    - **智能后台执行**: 具有断点恢复功能，不受页面刷新影响，支持大规模批次处理
    
    ### 处理范围说明
    
    - **处理所有分子**: 生产模式，无数量限制，支持百万级化合物库
    - **仅处理前1000个分子（测试用）**: 快速验证功能和参数设置
    
    ### 大化合物库批次处理
    
    当化合物库超过10万个分子时，系统会自动启用批次处理模式：
    
    #### 🔄 批次拆分
    - **自动拆分**: 将大化合物库拆分为10万分子一批的子文件
    - **保留属性**: 每个批次文件保留原始分子的所有属性和构象信息
    - **命名规则**: batch_001.sdf, batch_002.sdf, ...
    
    #### 🚀 批次执行
    - **智能后台模式**: 每个批次作为独立的智能后台任务运行，支持断点恢复
    
    #### 📊 批次监控
    - **实时状态**: 监控每个批次的运行状态和完成情况
    - **日志查看**: 查看每个批次的详细计算日志
    - **进度跟踪**: 显示整体进度和预估完成时间
    
    #### 🔗 结果合并
    - **自动检测**: 检测所有批次是否完成
    - **一键合并**: 自动合并所有批次结果为单个CSV文件
    - **数据完整性**: 确保合并后数据的完整性和一致性
    
    #### 💡 批次处理优势
    - **容错性**: 单个批次失败不影响其他批次
    - **可监控**: 实时监控每个批次的进度
    - **可恢复**: 失败的批次可以单独重新运行
    - **内存友好**: 避免大文件一次性加载到内存
    
    ### 聚合策略说明
    
    - **mean**: 平均值（推荐）
    - **max**: 最大值
    - **min**: 最小值  
    - **std**: 标准差
    - **median**: 中位数
    
    ### 输出格式
    
    生成的CSV文件包含：
    - SMILES列（可选）
    - 每个描述符一列
    - 每行对应一个分子
    
    ### 智能后台任务监控
    
    - **实时监控**: 查看任务状态、运行时间和进度
    - **日志查看**: 实时查看计算日志，了解详细进度  
    - **完成检查**: 自动检测任务完成，提供结果下载
    - **任务管理**: 支持停止和删除后台任务
    
    ### 注意事项
    
    - 确保输入的SDF文件包含3D坐标
    - **百万级化合物库**: 支持大规模处理，建议使用34进程
    - **内存使用**: 大文件可能需要较多内存，监控系统资源
    - **测试建议**: 首次使用建议先用测试模式验证
    - **智能后台**: 任务启动后可安全关闭页面，支持断点恢复
    - **处理时间**: 百万级数据可能需要数小时，可通过日志监控进度
    
    #### 批次处理注意事项
    
    - **磁盘空间**: 批次文件会占用额外磁盘空间，确保有足够空间
    - **系统负载**: 多个批次同时运行会增加系统负载，合理分配资源
    - **中途停止**: 可以随时停止单个或所有批次，已完成的批次结果会保留
    - **断点恢复**: 智能后台模式支持断点恢复，失败的批次可自动重启
    - **合并检查**: 所有批次完成后提供合并指令和示例代码
    - **文件清理**: 处理完成后可以选择删除临时批次文件以节省空间
    """
    )
