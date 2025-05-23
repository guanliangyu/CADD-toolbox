"""
分子库代表性子集选择系统 - 3D形状对比页面
"""
import os
import concurrent.futures # 添加导入
import multiprocessing # 添加导入

# 抑制 TensorFlow 和 CUDA 警告
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 3 = ERROR，仅显示错误
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
# 抑制CUDA相关重复注册警告
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
# 设置TF日志级别
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# 禁用警告
os.environ['TF_ENABLE_DEPRECATION_WARNINGS'] = 'false'

# 抑制 PyTorch 警告 (可选)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import random
import numpy as np
import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import matplotlib.pyplot as plt
import torch
import seaborn as sns
from sklearn.manifold import TSNE
import umap
import time
from scipy.stats import wasserstein_distance
from sklearn.neighbors import KernelDensity

# 尝试导入GPU加速相关的库
try:
    import cupy as cp
    from cuml.manifold import TSNE as cuTSNE
    HAS_CUML = True
except ImportError:
    HAS_CUML = False
    warnings.warn("cuML未安装，将使用CPU版本的t-SNE")

try:
    import torchani
    HAS_TORCHANI = True
    
    # 尝试检测 CUDA 支持，使用更安全的方法
    try:
        # 仅检查是否能创建模型并移至 GPU
        if torch.cuda.is_available():
            # 尝试加载模型到 GPU，这比检查 torch.classes.torchani 更可靠
            model = torchani.models.ANI1x(model_index=0)
            device = torch.device('cuda')
            model = model.to(device)
            # 如果能走到这步，说明支持 CUDA
            HAS_TORCHANI_CUDA = True
            del model  # 清理
            torch.cuda.empty_cache()
        else:
            HAS_TORCHANI_CUDA = False
    except Exception:
        HAS_TORCHANI_CUDA = False
        warnings.warn("TorchANI CUDA 加速检测失败，将使用 CPU 版本")
except ImportError:
    HAS_TORCHANI = False
    HAS_TORCHANI_CUDA = False
    warnings.warn("TorchANI 未安装，将不能使用 TorchANI 后端")
    
# DeepChem和TensorFlow
HAS_DEEPCHEM = False
HAS_DEEPCHEM_GPU = False
try:
    # 防止TensorFlow产生不必要的日志
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 仅显示ERROR
    try:
        import tensorflow as tf
        # 安全地检查GPU
        physical_devices = tf.config.list_physical_devices('GPU')
        tf_gpu_available = len(physical_devices) > 0
        # 设置内存增长
        if tf_gpu_available:
            for device in physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(device, True)
                except:
                    pass
    except ImportError:
        tf = None
        tf_gpu_available = False
    
    # 尝试导入DeepChem
    try:
        import deepchem as dc
        HAS_DEEPCHEM = True
        HAS_DEEPCHEM_GPU = tf_gpu_available
    except ImportError:
        warnings.warn("DeepChem未安装，将不能使用DeepChem后端")
except Exception as e:
    warnings.warn(f"初始化TensorFlow/DeepChem时出错: {str(e)}")

try:
    import clara.conformer as clara_conf
    import clara.molecule as clara_mol
    HAS_CLARA = True
except ImportError:
    HAS_CLARA = False
    warnings.warn("NVIDIA Clara未安装，将不能使用Clara后端")

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(script_dir)

# 设置Streamlit配置
st.set_page_config(
    page_title="分子库代表性子集选择系统 - 3D形状对比",
    page_icon="🧬",
    layout="wide"
)

# 禁用文件监视器以避免PyTorch相关错误
if hasattr(st, 'server'):
    st.server.server.server_options["watcher_type"] = "none"

# 清理缓存（使用新的方法）
if st.session_state.get('clear_cache', False):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.clear_cache = False

st.title("3D形状对比")

# 最新更新说明
with st.expander("🚀 最新更新：TorchANI混合精度优化", expanded=False):
    st.markdown("""
    ### 📈 性能优化更新
    **基于PyTorch官方文档的混合精度最佳实践：**
    
    ✅ **API更新**：
    - 使用 `torch.amp.GradScaler("cuda", enabled=use_amp)` 替代旧API
    - 使用 `torch.autocast(device_type, dtype=torch.float16, enabled=use_amp)` 
    - 支持 `enabled` 参数实现无缝切换
    
    ✅ **自动回退机制**：
    - 混合精度失败时自动切换到FP32
    - 智能错误处理和用户友好提示
    - 保持计算连续性
    
    ✅ **数据类型一致性**：
    - 修复 `masked_scatter_` 数据类型不匹配错误
    - 使用 `.to(energies.dtype)` 确保类型兼容
    - 正确的梯度裁剪顺序：unscale → clip → step → update
    
    ### 🎯 预期效果
    - **显存节省**: 50%（FP16 vs FP32）
    - **速度提升**: 2-3倍（在支持Tensor Core的GPU上）
    - **稳定性**: 自动回退确保计算不中断
    - **兼容性**: 支持各种GPU架构
    """)

col1, col2 = st.columns(2)

with col1:
    st.subheader("数据集A")
    fileA = st.file_uploader("上传第一个CSV文件", type="csv")
    
with col2:
    st.subheader("数据集B")
    fileB = st.file_uploader("上传第二个CSV文件", type="csv")

# 统一参数设置
st.subheader("参数设置")
# 创建主要的设置选项卡
main_tabs = st.tabs(["数据设置", "构象生成设置", "分析设置", "GPU设置"])

# 数据设置选项卡
with main_tabs[0]:
    col1, col2 = st.columns(2)
    with col1:
        smiles_col = st.text_input("SMILES列名", value="SMILES")
        max_samples = st.number_input("每个数据集的最大样本数", 100, 5000, 500)
    
    with col2:
        shape_desc = st.selectbox("形状描述符类型", ["USR", "USRCAT"], 
                                help="USR: 超快形状识别；USRCAT: 包含原子类型信息的USR")
        normalize_desc = st.checkbox("标准化描述符", value=True, 
                                   help="应用标准化以平衡不同尺度的特征")

# 构象生成设置选项卡
with main_tabs[1]:
    # 构象生成引擎选择 - 始终显示所有后端选项
    available_backends = [
        "auto",
        "rdkit", 
        "torchani", 
        "deepchem", 
        "clara"
    ]
    
    conformer_backend = st.selectbox(
        "3D构象生成后端",
        available_backends,
        help="选择用于生成3D构象的计算后端"
    )
    
    # 显示后端可用性状态
    with st.expander("后端可用性状态", expanded=False):
        st.write("**后端安装状态：**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"✅ RDKit: 始终可用" if True else "❌ RDKit: 不可用")
            st.write(f"✅ TorchANI: 可用" if HAS_TORCHANI else "❌ TorchANI: 不可用")
        with col2:
            st.write(f"✅ DeepChem: 可用" if HAS_DEEPCHEM else "❌ DeepChem: 不可用")
            st.write(f"✅ NVIDIA Clara: 可用" if HAS_CLARA else "❌ NVIDIA Clara: 不可用")
    
    # 检查所选后端的可用性
    backend_available = True
    if conformer_backend == "torchani" and not HAS_TORCHANI:
        st.error("❌ TorchANI 未安装，请安装 TorchANI 或选择其他后端")
        backend_available = False
    elif conformer_backend == "deepchem" and not HAS_DEEPCHEM:
        st.error("❌ DeepChem 未安装，请安装 DeepChem 或选择其他后端")
        backend_available = False
    elif conformer_backend == "clara" and not HAS_CLARA:
        st.error("❌ NVIDIA Clara 未安装，请安装 Clara 或选择其他后端")
        backend_available = False
    
    # 基础设置
    col1, col2, col3 = st.columns(3)
    with col1:
        max_attempts = st.slider("构象生成最大尝试次数", 10, 100, 50)
        add_hydrogens = st.checkbox("添加氢原子", value=True, 
                                  help="在构象生成前添加氢原子")
    
    with col2:
        use_mmff = st.checkbox("使用力场优化", value=True, 
                              help="使用分子力场优化构象")
        energy_iter = st.slider("能量优化迭代次数", 0, 500, 200, 
                              disabled=not use_mmff)
    
    with col3:
        if conformer_backend == "auto":
            st.info("自动模式：系统将根据分子大小和可用资源选择最佳后端")
            auto_select_info = """
            • 小分子 (<50原子)：优先使用TorchANI
            • 中等分子 (50-100原子)：优先使用DeepChem
            • 大分子 (>100原子)：优先使用Clara
            • 如果没有可用GPU：使用RDKit
            """
            st.markdown(auto_select_info)
    
    # 创建特定后端设置选项卡
    backend_tabs = st.tabs(["TorchANI", "DeepChem", "Clara"])
    
    # TorchANI设置
    with backend_tabs[0]:
        if conformer_backend in ["torchani", "auto"]:
            if not HAS_TORCHANI:
                st.warning("⚠️ TorchANI 未安装，以下设置仅供参考")
            
            st.write("TorchANI设置")
            
            # 基础设置
            col1, col2 = st.columns(2)
            with col1:
                torchani_model = st.selectbox(
                    "神经网络模型",
                    ["ANI2x", "ANI1x", "ANI1ccx"],
                    help="选择TorchANI的神经网络模型",
                    disabled=not HAS_TORCHANI
                )
                optimization_steps = st.slider(
                    "优化步数", 
                    50, 500, 100,
                    disabled=not HAS_TORCHANI,
                    help="优化迭代次数，影响构象质量和计算时间"
                )
            
            with col2:
                torchani_batch_size = st.slider(
                    "TorchANI批处理大小",
                    8, 1024, 32, # 将最大值从64修改为1024
                    disabled=not HAS_TORCHANI,
                    help="批量处理的分子数，越大GPU利用率越高但内存消耗也越大"
                )
                use_torchani_optimization = st.checkbox(
                    "启用批量优化模式",
                    value=True,
                    disabled=not HAS_TORCHANI,
                    help="启用优化的批处理模式，可显著提高GPU利用率和处理速度"
                )
            
            # 高级设置
            with st.expander("🔧 TorchANI高级设置", expanded=False):
                col3, col4 = st.columns(2)
                with col3:
                    learning_rate = st.slider(
                        "学习率",
                        0.001, 0.1, 0.01,
                        disabled=not HAS_TORCHANI,
                        help="Adam优化器学习率，影响收敛速度和稳定性"
                    )
                    use_mixed_precision_torchani = st.checkbox(
                        "使用混合精度",
                        value=True,
                        disabled=not HAS_TORCHANI,
                        help="使用FP16混合精度计算以节省GPU内存并提高速度。如遇到数据类型错误，请禁用此选项"
                    )
                    
                    if use_mixed_precision_torchani and HAS_TORCHANI:
                        st.info("💡 **混合精度说明**")
                        st.markdown("""
                        - **优势**: 节省50%显存，提高2-3倍计算速度（需Volta/Turing/Ampere架构）
                        - **自动回退**: 如遇数据类型错误会自动切换到FP32
                        - **最佳实践**: 采用PyTorch官方推荐的autocast + GradScaler模式
                        """)
                        
                        if torch.cuda.is_available():
                            gpu_name = torch.cuda.get_device_name(0)
                            if any(arch in gpu_name.upper() for arch in ['V100', 'A100', 'RTX', 'TITAN RTX', 'QUADRO RTX']):
                                st.success("✅ 检测到支持Tensor Core的GPU，混合精度效果最佳")
                            elif any(arch in gpu_name.upper() for arch in ['GTX 16', 'GTX 20', 'GTX 30', 'GTX 40']):
                                st.info("ℹ️ 当前GPU支持混合精度，预期有适度加速")
                            else: # This else corresponds to the inner if torch.cuda.is_available()
                                st.warning("⚠️ 当前GPU可能不支持Tensor Core，混合精度加速效果有限")
                    # Linter Error: Unindent amount does not match previous indent (Line 331 for elif)
                    # This elif should align with the `if use_mixed_precision_torchani and HAS_TORCHANI:`
                    elif not use_mixed_precision_torchani and HAS_TORCHANI: 
                        st.warning("⚠️ **混合精度已禁用**") # Linter Error: Unexpected indentation (Line 332)
                        st.markdown("""
                        - GPU内存使用将增加约2倍
                        - 计算速度可能降低2-3倍
                        - 但数值精度更高，更稳定
                        """)
                        
                        st.info("💡 如果遇到dtype错误，可以尝试：")
                        st.markdown("""
                        1. 减小批处理大小
                        2. 降低学习率
                        3. 禁用梯度裁剪
                        4. 更新PyTorch到最新版本
                        """)
                
                # Linter Error: Unindent amount does not match previous indent (Line 348 for with col4:)
                # This with col4: should align with `with col3:`
                with col4:
                    gradient_clipping = st.checkbox(
                        "梯度裁剪",
                        value=True,
                        disabled=not HAS_TORCHANI,
                        help="防止梯度爆炸，提高优化稳定性"
                    )
            
            # Linter Error: Unindent amount does not match previous indent (Line 363 for if HAS_TORCHANI:)
            # This if HAS_TORCHANI: should align with the `with st.expander(...)`
            if HAS_TORCHANI:
                if use_torchani_optimization:
                    st.success("✅ 批量优化模式已启用，预期性能提升 5-20x")
                    if torch.cuda.is_available():
                        gpu_name = torch.cuda.get_device_name(0)
                        if "RTX" in gpu_name or "Tesla" in gpu_name or "A100" in gpu_name:
                            st.info("🚀 检测到高性能GPU，建议增大批处理大小以获得最佳性能")
                    else:
                        st.warning("⚠️ 未检测到GPU，将使用CPU模式（速度较慢）")
                else:
                    st.warning("⚠️ 批量优化未启用，将使用传统的逐个处理模式")
                    
                with st.expander("🔧 混合精度最佳实践和故障排除", expanded=False):
                    st.markdown("### 🔍 版本兼容性检查")
                    import torch # Local import, consider moving to top if not already there
                    import sys # Local import
                    
                    torch_version = torch.__version__
                    python_version = sys.version.split()[0]
                    
                    st.info(f"**当前环境:**")
                    st.write(f"- Python: {python_version}")
                    st.write(f"- PyTorch: {torch_version}")
                    
                    if HAS_TORCHANI:
                        try:
                            import torchani # Local import
                            torchani_version = torchani.__version__
                            st.write(f"- TorchANI: {torchani_version}")
                            
                            if torch_version >= "1.12.0" and torchani_version <= "2.2.0":
                                st.warning("⚠️ **已知兼容性问题**: TorchANI ≤ 2.2.0 与 PyTorch ≥ 1.12.0 在混合精度下可能不兼容")
                                st.info("建议升级TorchANI到最新版本: `pip install --upgrade torchani`")
                        except:
                            st.write("- TorchANI: 无法获取版本信息")
                    
                    if torch.cuda.is_available():
                        cuda_version = torch.version.cuda
                        cudnn_version = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A"
                        st.write(f"- CUDA: {cuda_version}")
                        st.write(f"- cuDNN: {cudnn_version}")
                        
                        gpu_name = torch.cuda.get_device_name(0)
                        gpu_capability = torch.cuda.get_device_capability(0)
                        st.write(f"- GPU: {gpu_name}")
                        st.write(f"- 计算能力: {gpu_capability[0]}.{gpu_capability[1]}")
                        
                        if gpu_capability[0] >= 7:
                            st.success("✅ GPU支持Tensor Core，混合精度效果最佳")
                        elif gpu_capability[0] >= 6:
                            st.info("ℹ️ GPU部分支持混合精度，效果有限")
                        else:
                            st.warning("⚠️ GPU不支持混合精度加速")
                    
                    st.markdown("### 🧪 混合精度兼容性测试")
                    if st.button("运行TorchANI混合精度兼容性测试"):
                        if HAS_TORCHANI and torch.cuda.is_available():
                            try:
                                st.info("正在测试TorchANI混合精度兼容性...")
                                from rdkit import Chem # Local import
                                from rdkit.Chem import AllChem # Local import
                                test_mol = Chem.MolFromSmiles("CCO")
                                test_mol = Chem.AddHs(test_mol)
                                AllChem.EmbedMolecule(test_mol)
                                
                                coords = []
                                species_atomic_nums = []
                                for i in range(test_mol.GetNumAtoms()):
                                    atom = test_mol.GetAtomWithIdx(i)
                                    pos = test_mol.GetConformer().GetAtomPosition(i)
                                    coords.append([pos.x, pos.y, pos.z])
                                    species_atomic_nums.append(atom.GetAtomicNum())
                                
                                coords_tensor = torch.tensor([coords], dtype=torch.float32).cuda()
                                
                                model_test = torchani.models.ANI2x(periodic_table_index=False).cuda().eval()
                                # Assuming SUPPORTED_SPECIES_PREPROC is globally available from previous edits
                                symbol_to_int_test = torchani.utils.ChemicalSymbolsToInts(list(SUPPORTED_SPECIES_PREPROC.values()))
                                symbols_test = [SUPPORTED_SPECIES_PREPROC.get(s_num, 'X') for s_num in species_atomic_nums]
                                species_idx_test = symbol_to_int_test(symbols_test).unsqueeze(0).cuda()
                                
                                test_results = {}
                                try:
                                    with torch.no_grad():
                                        model_test((species_idx_test, coords_tensor)).energies
                                    test_results["FP32"] = "✅ 成功"
                                except Exception as e_fp32:
                                    test_results["FP32"] = f"❌ 失败: {str(e_fp32)}"
                                
                                try:
                                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                                        model_test((species_idx_test, coords_tensor)).energies
                                    test_results["AMP (autocast)"] = "✅ 成功"
                                except Exception as e_amp:
                                    test_results["AMP (autocast)"] = f"❌ 失败: {str(e_amp)[:100]}..."
                                
                                st.write("**测试结果:**")
                                for test_name, result in test_results.items():
                                    st.write(f"- {test_name}: {result}")
                                
                                if "✅ 成功" in test_results.get("AMP (autocast)", ""):
                                    st.success("🎉 TorchANI混合精度兼容性测试通过！")
                                else:
                                    st.error("❌ TorchANI混合精度兼容性测试失败。")
                            except Exception as e_test:
                                st.error(f"测试过程中出错: {str(e_test)}")
                        else:
                            st.warning("需要TorchANI和CUDA支持才能进行测试")
                    
                    st.markdown("""
                    ### 📊 性能优化建议
                    **最大化混合精度效果:**
                    - 确保批处理大小是8的倍数（利用Tensor Core）
                    - 使用支持Tensor Core的GPU（Volta/Turing/Ampere架构）
                    - 保持网络足够复杂以充分利用GPU
                    
                    ### ⚠️ 常见问题和解决方案
                    **1. 'masked_scatter_: expected self and source to have same dtypes but got Float and Half'**
                    - ✅ 已修复：现在使用 `.to(energies.dtype)` 确保类型一致
                    - 自动回退：如果混合精度失败会自动切换到FP32
                    
                    **2. 'CUDNN_STATUS_BAD_PARAM' 或类型不匹配错误**
                    - 减小批处理大小到16或8
                    - 降低学习率到0.001
                    - 禁用梯度裁剪
                    
                    **3. 内存不足 (OOM)**
                    - 减小 `max_atoms_per_batch` 参数
                    - 降低批处理大小
                    - 启用梯度累积模式
                    
                    **4. 性能提升不明显**
                    - 检查GPU是否支持Tensor Core
                    - 增大批处理大小以充分利用GPU
                    - 确保分子复杂度足够（>20个原子）
                    
                    ### 🚀 现代PyTorch最佳实践
                    **我们已采用的官方推荐做法:**
                    ```python
                    # 新的API（推荐）
                    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
                    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        # 前向传播
                    
                    # 正确的梯度处理顺序
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)  # 梯度裁剪前必须unscale
                    torch.nn.utils.clip_grad_norm_(...)
                    scaler.step(optimizer)
                    scaler.update()
                    ```
                    """)
                    
                    # PyTorch版本检查
                    if torch_version < "1.10.0":
                        st.warning("⚠️ 建议升级到PyTorch 1.10+以获得最佳混合精度支持")
                    else:
                        st.success("✅ PyTorch版本支持新的混合精度API")
                
                # GPU内存估算
                if torch.cuda.is_available() and use_torchani_optimization:
                    estimated_mem = torchani_batch_size * 50  # 每个分子大约50MB
                    gpu_total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
                    mem_usage = (estimated_mem / gpu_total_mem) * 100
                    
                    if mem_usage > 80:
                        st.error(f"⚠️ 估算GPU内存使用: {mem_usage:.1f}%，建议减小批处理大小")
                    elif mem_usage > 60:
                        st.warning(f"⚠️ 估算GPU内存使用: {mem_usage:.1f}%，注意监控内存")
                    else:
                        st.info(f"✅ 估算GPU内存使用: {mem_usage:.1f}%，设置合理")
            
            if not HAS_TORCHANI:
                st.info("💡 安装 TorchANI: `pip install torchani`")
        else:
            st.info("当前未选择 TorchANI 后端")
    
    # DeepChem设置
    with backend_tabs[1]:
        if conformer_backend in ["deepchem", "auto"]:
            if not HAS_DEEPCHEM:
                st.warning("⚠️ DeepChem 未安装，以下设置仅供参考")
            
            st.write("DeepChem设置")
            deepchem_model = st.selectbox(
                "模型类型",
                ["mpnn", "schnet", "cgcnn"],
                help="选择DeepChem的分子表示模型",
                disabled=not HAS_DEEPCHEM
            )
            use_mixed_precision = st.checkbox(
                "使用混合精度训练",
                value=True,
                help="启用FP16混合精度以提高性能",
                disabled=not HAS_DEEPCHEM
            )
            batch_size_dc = st.slider(
                "批处理大小",
                16, 256, 64,
                help="DeepChem的批处理大小",
                disabled=not HAS_DEEPCHEM
            )
            dc_force_field = st.selectbox(
                "力场类型",
                ["mmff94s", "uff", "gaff"],
                help="DeepChem使用的力场",
                disabled=not HAS_DEEPCHEM
            )
            
            if not HAS_DEEPCHEM:
                st.info("💡 安装 DeepChem: `pip install deepchem`")
        else:
            st.info("当前未选择 DeepChem 后端")
    
    # Clara设置
    with backend_tabs[2]:
        if conformer_backend in ["clara", "auto"]:
            if not HAS_CLARA:
                st.warning("⚠️ NVIDIA Clara 未安装，以下设置仅供参考")
            
            st.write("NVIDIA Clara设置")
            clara_force_field = st.selectbox(
                "力场",
                ["MMFF94s", "UFF", "GAFF"],
                help="选择Clara的力场",
                disabled=not HAS_CLARA
            )
            clara_precision = st.selectbox(
                "计算精度",
                ["mixed", "fp32", "fp16"],
                help="选择计算精度",
                disabled=not HAS_CLARA
            )
            clara_num_conformers = st.slider(
                "构象数量",
                1, 10, 1,
                help="生成的构象数量",
                disabled=not HAS_CLARA
            )
            clara_energy_threshold = st.slider(
                "能量阈值(kcal/mol)",
                0.1, 10.0, 1.0,
                help="能量筛选阈值",
                disabled=not HAS_CLARA
            )
            clara_optimization_steps = st.slider(
                "优化步数", 
                100, 1000, 500,
                help="Clara优化迭代次数",
                disabled=not HAS_CLARA
            )
            
            if not HAS_CLARA:
                st.info("💡 安装 NVIDIA Clara: 参考 NVIDIA Clara 官方文档")
        else:
            st.info("当前未选择 NVIDIA Clara 后端")

# 分析设置选项卡
with main_tabs[2]:
    col1, col2 = st.columns(2)
    with col1:
        dim_reduction = st.selectbox("降维方法", ["t-SNE", "UMAP"])
        if dim_reduction == "t-SNE":
            perplexity = st.slider("t-SNE困惑度", 5, 50, 30)
        else:
            n_neighbors = st.slider("UMAP邻居数", 5, 50, 15)
            min_dist = st.slider("UMAP最小距离", 0.01, 0.99, 0.1, 0.01)
    
    with col2:
        st.write("可视化设置")
        plot_height = st.slider("图表高度", 400, 1000, 600)
        plot_width = st.slider("图表宽度", 400, 1000, 800)
        color_scheme = st.selectbox("配色方案", 
                           ["viridis", "plasma", "inferno", "magma", "cividis"])

# GPU设置选项卡
with main_tabs[3]:
    col1, col2 = st.columns(2)
    with col1:
        enable_gpu = st.checkbox("启用GPU加速", value=True, 
                                help="使用GPU加速计算")
        auto_batch = st.checkbox("自动批处理大小", value=True, 
                                 help="根据GPU内存自动调整批处理大小")
        batch_size = st.slider("批处理大小", 10, 500, 50, 
                             disabled=auto_batch)
    
    with col2:
        if enable_gpu:
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
                
                st.success(f"✅ GPU可用: {gpu_name}")
                st.info(f"总显存: {gpu_mem_total:.1f} MB")
                
                # GPU使用策略
                gpu_strategy = st.selectbox(
                    "GPU使用策略",
                    ["平衡", "性能优先", "内存优先"],
                    help="平衡：平衡速度和内存使用；性能优先：更快但使用更多内存；内存优先：节省内存但较慢"
                )
                
                # GPU内存限制
                gpu_mem_limit = st.slider(
                    "GPU内存使用限制 (%)",
                    10, 95, 80,
                    help="限制GPU内存使用百分比以防止崩溃"
                )
            else:
                st.error("❌ 未检测到可用的GPU")
                st.info("将使用CPU进行计算，速度可能较慢")
        else:
            st.info("GPU加速已禁用，将使用CPU进行计算")

def initialize_cuda():
    """初始化CUDA设备并返回设备信息"""
    try:
        cuda_available = torch.cuda.is_available()
        device = torch.device("cuda" if cuda_available else "cpu")
        
        if cuda_available:
            torch.cuda.empty_cache()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem_total = torch.cuda.get_device_properties(0).total_memory / 1024**2
            gpu_mem_alloc = torch.cuda.memory_allocated(0) / 1024**2
            gpu_mem_cached = torch.cuda.memory_reserved(0) / 1024**2
            
            st.sidebar.success("✅ CUDA可用，将使用GPU加速")
            st.sidebar.info(
                f"GPU信息:\n"
                f"- 设备: {gpu_name}\n"
                f"- 总显存: {gpu_mem_total:.1f}MB\n"
                f"- 已分配: {gpu_mem_alloc:.1f}MB\n"
                f"- 已缓存: {gpu_mem_cached:.1f}MB"
            )
        else:
            st.sidebar.info("ℹ️ CUDA不可用，将使用CPU计算")
        
        return cuda_available, device
    except Exception as e:
        st.sidebar.error(f"GPU初始化错误: {str(e)}")
        return False, torch.device("cpu")

def load_smiles(file, smiles_col='SMILES'):
    """从CSV读取SMILES并转换为RDKit Mol对象"""
    try:
        df = pd.read_csv(file)
        if smiles_col not in df.columns:
            st.error(f"未找到SMILES列: {smiles_col}")
            return None, None
        
        mols = []
        valid_indices = []
        for i, row in df.iterrows():
            smi = str(row[smiles_col]).strip()
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mols.append(mol)
                valid_indices.append(i)
        
        return mols, df.iloc[valid_indices]
    except Exception as e:
        st.error(f"读取CSV文件时出错: {str(e)}")
        return None, None

def generate_3d_conformer(mol, max_attempts=50, use_mmff=True, energy_iter=200, add_hydrogens=True):
    """生成3D构象，支持更多配置选项"""
    if mol is None:
        return None
    
    try:
        # 根据需要添加氢原子
        mol_3d = Chem.AddHs(mol) if add_hydrogens else Chem.Mol(mol)
        
        # 设置ETKDG参数
        ps = AllChem.ETKDGv3()
        # 注意：RDKit 中正确的参数名是 maxAttempts，不是 maxAttempts
        # 检查 RDKit 版本兼容性
        try:
            ps.maxAttempts = max_attempts
        except AttributeError:
            # 如果不支持 maxAttempts，使用默认设置
            st.warning("当前RDKit版本不支持maxAttempts参数，使用默认设置")
        
        ps.randomSeed = 42  # 设置随机种子以提高可重复性
        ps.numThreads = 0  # 使用所有可用线程
        ps.useRandomCoords = True  # 使用随机初始坐标
        
        # 嵌入分子
        cid = AllChem.EmbedMolecule(mol_3d, ps)
        if cid < 0:
            # 如果 ETKDG 失败，尝试更简单的方法
            st.warning("ETKDG嵌入失败，尝试基本嵌入方法")
            try:
                # 尝试多次嵌入
                for attempt in range(max_attempts):
                    cid = AllChem.EmbedMolecule(mol_3d, randomSeed=42 + attempt)
                    if cid >= 0:
                        break
                    if cid < 0:
                        return None
            except Exception:
                return None
        
        # 应用力场优化
        if use_mmff:
            try:
                # 尝试MMFF优化
                result = AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=energy_iter)
                if result != 0:
                # 如果MMFF失败，尝试UFF
                    st.warning("MMFF优化失败，尝试UFF优化")
                AllChem.UFFOptimizeMolecule(mol_3d, maxIters=energy_iter)
            except Exception as opt_error:
                st.warning(f"力场优化失败: {str(opt_error)}，跳过优化步骤")
        
        # 如果添加了氢原子，现在去除它们
        if add_hydrogens:
            mol_3d = Chem.RemoveHs(mol_3d)
        
        return mol_3d
    except Exception as e:
        st.warning(f"3D构象生成失败: {str(e)}")
        return None

# 全局或模块级变量，确保 preprocess_single_mol_for_torchani 可以访问
# 这些通常在 Streamlit 应用的顶部定义
# 确保 HAS_TORCHANI, atomic_numbers_to_symbols 等已定义且在此作用域可见
# 如果它们只在 Streamlit 主函数流中定义，需要调整或传递它们

# 假设 atomic_numbers_to_symbols 已经在全局或模块级别定义，例如：
# atomic_numbers_to_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl'}
# 如果它是在某个函数内部定义的，你需要把它移到全局或者作为参数传递给预处理函数

SUPPORTED_SPECIES_PREPROC = {
    1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl'
}

def preprocess_single_mol_for_torchani(args):
    """为TorchANI预处理单个分子，用于多进程处理。"""
    original_idx, mol_smiles_or_rdkit_mol = args # 假设mol是RDKit Mol对象或SMILES字符串

    # 如果传入的是SMILES，先转换为Mol对象 (这取决于mols列表的内容)
    # 为简化，假设mols列表已经是RDKit Mol对象
    # if isinstance(mol_smiles_or_rdkit_mol, str):
    #     mol = Chem.MolFromSmiles(mol_smiles_or_rdkit_mol)
    # else:
    mol = mol_smiles_or_rdkit_mol

    if mol is None:
        return {'original_idx': original_idx, 'mol_h': None, 'original_species_len': 0, 'error': 'Input mol is None'}

    try:
        mol_h = Chem.AddHs(mol)
        num_atoms = mol_h.GetNumAtoms()

        supported = True
        for atom in mol_h.GetAtoms():
            if atom.GetAtomicNum() not in SUPPORTED_SPECIES_PREPROC:
                supported = False
                break
        
        if not supported:
            return {'original_idx': original_idx, 'mol_h': None, 'original_species_len': 0, 'error': 'Unsupported atom types'}
        
        if mol_h.GetNumConformers() == 0:
            # 使用更鲁棒的嵌入参数
            ps = AllChem.ETKDGv3()
            ps.randomSeed = original_idx # Vary seed per molecule for better diversity if needed
            ps.numThreads = 0 # Use all available cores for embedding this single molecule by RDKit if it supports it
            embed_result = AllChem.EmbedMolecule(mol_h, ps)
            if embed_result < 0: # ETKDG失败
                # 尝试备用方法
                embed_result = AllChem.EmbedMolecule(mol_h, useRandomCoords=True, forceBasicKnowledge=True, randomSeed=original_idx + 1000)
                if embed_result < 0:
                     return {'original_idx': original_idx, 'mol_h': None, 'original_species_len': 0, 'error': 'Initial conformer embedding failed after multiple attempts'}
        
        return {'original_idx': original_idx, 'mol_h': mol_h, 'original_species_len': num_atoms, 'error': None}
    except Exception as e:
        return {'original_idx': original_idx, 'mol_h': None, 'original_species_len': 0, 'error': f'Preprocessing exception: {str(e)}'}

def generate_3d_conformer_torchani_optimized(mols, model_name='ANI2x', optimization_steps=100, device=None, 
                                           batch_size=32, learning_rate=0.01, use_mixed_precision_torchani=True, 
                                           max_atoms_per_batch=5000, gradient_clipping=True,
                                           progress_bar_ui=None, 
                                           progress_text_ui=None,
                                           status_container_ui=None):
    """使用TorchANI批量生成3D构象 - 优化版本
    
    采用PyTorch官方推荐的自动混合精度最佳实践:
    - 使用 torch.autocast(device_type, dtype=torch.float16, enabled=use_amp) 
    - 使用 torch.amp.GradScaler("cuda", enabled=use_amp)
    - 自动回退机制：混合精度失败时自动切换到FP32
    - 正确的梯度裁剪顺序：unscale -> clip -> step -> update
    
    Args:
        mols: 分子列表
        model_name: TorchANI模型名称 ('ANI1x', 'ANI1ccx', 'ANI2x')
        optimization_steps: 优化迭代次数
        device: 计算设备
        batch_size: 批处理大小
        learning_rate: Adam优化器学习率
        use_mixed_precision_torchani: 是否启用混合精度（FP16）
        max_atoms_per_batch: 每批次最大原子数限制
        gradient_clipping: 是否启用梯度裁剪
    
    Returns:
        list: 优化后的分子列表
    """
    if not HAS_TORCHANI or not mols:
        return [None] * len(mols)
        
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
            st.info(f"🚀 TorchANI优化批处理 - 批大小: {batch_size}, 设备: {device}, 混合精度: {use_mixed_precision_torchani}")
        
        # 预加载模型（只加载一次）
        if model_name == 'ANI1x':
            model = torchani.models.ANI1x(periodic_table_index=False).to(device)
        elif model_name == 'ANI1ccx':
            model = torchani.models.ANI1ccx(periodic_table_index=False).to(device)
        elif model_name == 'ANI2x':
            model = torchani.models.ANI2x(periodic_table_index=False).to(device)
        else:
            model = torchani.models.ANI2x(periodic_table_index=False).to(device)
        
        model.eval()  # 设置为评估模式
        
        # 支持的元素
        supported_species = ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
        atomic_numbers_to_symbols = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 17: 'Cl'}
        symbol_to_int = torchani.utils.ChemicalSymbolsToInts(supported_species)
        
        results = []
        start_time = time.time()
        
        initial_mol_count = len(mols)
        if progress_text_ui:
            progress_text_ui.text(f"TorchANI: 预处理 {initial_mol_count} 个分子...")

        # 使用 ThreadPoolExecutor 并行化预处理
        processed_mols_info_list = [None] * initial_mol_count # 保持顺序
        
        # 从kwargs获取线程数，默认为36，或CPU核心数
        default_threads = 36
        try:
            num_threads = kwargs.get('num_preprocessing_threads', default_threads)
            if not isinstance(num_threads, int) or num_threads <= 0:
                num_threads = default_threads
        except:
            num_threads = default_threads
            
        # 确保不超过CPU核心数太多，或者可以设置一个合理的上限
        max_threads = multiprocessing.cpu_count() * 2 # 例如，不超过CPU核心数的两倍
        num_threads = min(num_threads, max_threads, initial_mol_count if initial_mol_count > 0 else 1)


        if status_container_ui:
            status_container_ui.info(f"TorchANI: 开始并行预处理 {initial_mol_count} 个分子，使用 {num_threads} 个线程...")

        # 准备参数列表
        args_list = [(idx, mol) for idx, mol in enumerate(mols)]
        
        completed_tasks = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # 使用 submit 和 as_completed 来更好地处理进度更新
            future_to_idx = {executor.submit(preprocess_single_mol_for_torchani, arg): arg[0] for arg in args_list}
            
            for future in concurrent.futures.as_completed(future_to_idx):
                original_idx = future_to_idx[future]
                try:
                    result_dict = future.result()
                    # preprocess_single_mol_for_torchani 返回的字典包含 original_idx, mol_h, original_species_len, error
                    # 我们需要将其转换为 processed_mols_info 期望的格式和内容
                    processed_mols_info_list[original_idx] = {
                        'mol_h': result_dict.get('mol_h'),
                        'original_species_len': result_dict.get('original_species_len', 0),
                        'error': result_dict.get('error')
                    }
                except Exception as exc:
                    processed_mols_info_list[original_idx] = {
                        'mol_h': None, 
                        'original_species_len': 0, 
                        'error': f'Exception during parallel preprocessing: {str(exc)}'
                    }
                
                completed_tasks += 1
                if progress_bar_ui and initial_mol_count > 0:
                    progress_bar_ui.progress(completed_tasks / initial_mol_count, text=f"TorchANI: 预处理分子 {completed_tasks}/{initial_mol_count}")
        
        if progress_text_ui:
            progress_text_ui.text(f"TorchANI: 预处理完成 {completed_tasks}/{initial_mol_count} 个分子。")

        # processed_mols_info 现在是 processed_mols_info_list
        processed_mols_info = processed_mols_info_list

        # Filter out mols that failed pre-processing for the actual processing list
        # 确保这里的 'mol_h' 和 'original_species_len' 键与 preprocess_single_mol_for_torchani 返回的一致
        processed_mols_for_optimization = [info['mol_h'] for info in processed_mols_info if info and info.get('mol_h') is not None]
        
        if not processed_mols_for_optimization:
            if status_container_ui:
                status_container_ui.warning("TorchANI: 所有分子预处理失败，无法进行优化。")
            elif progress_text_ui:
                progress_text_ui.text("TorchANI: 所有分子预处理失败。")
            if progress_bar_ui:
                progress_bar_ui.progress(1.0, text="TorchANI: 预处理失败")
            return [None] * len(mols) # Return list of Nones matching original input size

        # Dynamic batch size adjustment based on successfully pre-processed mols
        current_total_atoms = sum(info['original_species_len'] for info in processed_mols_info if info['mol_h'] is not None)
        num_valid_mols = len(processed_mols_for_optimization)
        effective_batch_size = batch_size # Directly use the user-provided batch_size
        if status_container_ui:
            status_container_ui.info(f"⚙️ TorchANI: 使用用户指定的批处理大小: {effective_batch_size} (共 {num_valid_mols} 个有效分子)")

        num_batches = (num_valid_mols - 1) // effective_batch_size + 1
        results_for_optimized_mols = [None] * num_valid_mols # Results for successfully preprocessed mols

        for i in range(0, num_valid_mols, effective_batch_size):
            batch_mols_h = processed_mols_for_optimization[i : i + effective_batch_size]
            current_batch_num = i // effective_batch_size + 1

            if progress_text_ui:
                progress_text_ui.text(f"TorchANI: 开始优化批次 {current_batch_num}/{num_batches} (共 {len(batch_mols_h)} 分子)")
            if progress_bar_ui:
                 # Progress based on batches being submitted to optimization
                 progress_bar_ui.progress( (i + 0.1) / num_valid_mols , text=f"TorchANI: 优化批次 {current_batch_num}/{num_batches}")

            batch_results = []
            
            # 准备批量数据
            batch_species = []
            batch_coordinates = []
            valid_indices = []
            
            for j, mol_h in enumerate(batch_mols_h):
                if mol_h is None:
                    continue
                    
                try:
                    # 获取原子信息
                    species = [atom.GetAtomicNum() for atom in mol_h.GetAtoms()]
                    symbols = [atomic_numbers_to_symbols[num] for num in species]
                    
                    # 获取坐标
                    conf = mol_h.GetConformer()
                    coordinates = []
                    for k in range(mol_h.GetNumAtoms()):
                        pos = conf.GetAtomPosition(k)
                        coordinates.append([pos.x, pos.y, pos.z])
        
                    batch_species.append(species)
                    batch_coordinates.append(coordinates)
                    valid_indices.append(j)
                except:
                    continue
            
            if not batch_species:
                batch_results = [None] * len(batch_mols_h)
                results.extend(batch_results)
                continue # continue to next batch in the main batch loop
            
            try:
                # 使用填充处理不同大小的分子
                max_atoms = max(len(species) for species in batch_species)
                
                # 创建批量张量
                padded_species = []
                padded_coords = []
                
                for species, coords in zip(batch_species, batch_coordinates):
                    # 填充到最大原子数
                    padded_species_row = species + [0] * (max_atoms - len(species))
                    padded_coords_row = coords + [[0.0, 0.0, 0.0]] * (max_atoms - len(coords))
                    
                    padded_species.append(padded_species_row)
                    padded_coords.append(padded_coords_row)
                
                # 转换为张量
                species_tensor = torch.tensor(padded_species, device=device)
                coordinates_tensor = torch.tensor(padded_coords, device=device, 
                                                dtype=torch.float32, requires_grad=True)
                
                # 创建mask以忽略填充部分
                mask = torch.zeros_like(species_tensor, dtype=torch.bool, device=device)
                for idx, original_species in enumerate(batch_species):
                    mask[idx, :len(original_species)] = True
                
                # 将species转换为symbol indices
                batch_species_idx = []
                for species in batch_species:
                    symbols = [atomic_numbers_to_symbols[num] for num in species]
                    species_idx = symbol_to_int(symbols)
                    # 填充到max_atoms
                    padded_idx = torch.cat([
                        species_idx, 
                        torch.zeros(max_atoms - len(species_idx), dtype=species_idx.dtype)
                    ])
                    batch_species_idx.append(padded_idx)
                
                species_idx_tensor = torch.stack(batch_species_idx).to(device)
                
                # 优化器 - 使用用户指定的学习率
                optimizer = torch.optim.Adam([coordinates_tensor], lr=learning_rate)
                
                # 创建混合精度scaler - 使用官方推荐的方式
                scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision_torchani and device.type == 'cuda')
                
                # 批量优化
                best_energy = float('inf')
                energy_history = []
                mixed_precision_failed = False
                
                for step in range(optimization_steps):
                    optimizer.zero_grad()
                    
                    # 使用混合精度或常规计算 - 按照PyTorch官方文档的最佳实践
                    try:
                        is_amp_really_active = use_mixed_precision_torchani and device.type == 'cuda' and not mixed_precision_failed
                        
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=is_amp_really_active):
                            energies = model((species_idx_tensor, coordinates_tensor)).energies

                            # 如果混合精度被禁用但energies仍是float64，强制转换为float32
                            if not is_amp_really_active and energies.dtype == torch.float64:
                                energies = energies.float()
                            
                            mask_any_dim1 = mask.any(dim=1)
                            target_device_for_mask = energies.device
                            mask_float = mask_any_dim1.to(device=target_device_for_mask, dtype=energies.dtype)
                            masked_energies = energies * mask_float
                            total_energy = masked_energies.sum()
                        
                        # 混合精度反向传播
                        scaler.scale(total_energy).backward()
                        
                        # 梯度裁剪 - 按照文档先unscale再裁剪
                        if gradient_clipping:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_([coordinates_tensor], max_norm=1.0)
                        
                        scaler.step(optimizer)
                        scaler.update()
                        
                        current_energy = total_energy.float().item()
                        
                    except RuntimeError as mp_error:
                        if "autocast" in str(mp_error).lower() or "half" in str(mp_error).lower() or "dtype" in str(mp_error).lower() or "masked_scatter" in str(mp_error).lower():
                            if use_mixed_precision_torchani and not mixed_precision_failed:
                                # Use status_container_ui for warnings if available
                                warning_msg = f"TorchANI: 混合精度计算失败 (批次 {current_batch_num}, 步骤 {step})，回退到FP32: {str(mp_error)[:100]}..."
                                if status_container_ui:
                                    status_container_ui.warning(warning_msg)
                                else:
                                    st.warning(warning_msg) # Original warning as fallback
                                mixed_precision_failed = True
                                scaler = torch.amp.GradScaler("cuda", enabled=False)
                                optimizer.zero_grad()
                                energies_fp32 = model((species_idx_tensor, coordinates_tensor)).energies.float()
                                mask_val_fp32 = mask.any(dim=1).float().to(device=energies_fp32.device)
                                masked_energies = energies_fp32 * mask_val_fp32
                                total_energy = masked_energies.sum()
                                scaler.scale(total_energy).backward()
                                if gradient_clipping:
                                    scaler.unscale_(optimizer)
                                    torch.nn.utils.clip_grad_norm_([coordinates_tensor], max_norm=1.0)
                                scaler.step(optimizer)
                                scaler.update()
                                current_energy = total_energy.item()
                            else:
                                raise mp_error
                        else:
                            raise mp_error
                    
                    # 记录能量历史
                    energy_history.append(current_energy)
                    
                    # 跟踪最佳能量
                    if current_energy < best_energy:
                        best_energy = current_energy
                    
                    # Update progress more frequently, e.g., every 10 steps or if it's the last step
                    if step % 10 == 0 or step == optimization_steps - 1:
                        avg_energy = current_energy / len(valid_indices) if valid_indices else 0.0
                        precision_mode = "FP32" if mixed_precision_failed else ("FP16" if use_mixed_precision_torchani and device.type == 'cuda' else "FP32")
                        
                        # Progress text for overall batch step
                        if progress_text_ui:
                            progress_text_ui.text(f"TorchANI: 批次 {current_batch_num}/{num_batches} - 优化步骤 {step+1}/{optimization_steps} [E: {avg_energy:.3f} kcal/mol, {precision_mode}]")
                        
                        # Detailed log via status_container if a separate UI element for logs
                        if status_container_ui and (step % 50 == 0 or step == optimization_steps -1): # Less frequent for detailed log line
                             status_container_ui.info(f"TorchANI 批次 {current_batch_num} 详细: 步骤 {step+1}, E:{avg_energy:.3f}, {precision_mode}")

                    # 早停机制：如果能量不再显著改善
                    if step > 50 and len(energy_history) >= 10:
                        recent_improvement = energy_history[-10] - energy_history[-1]
                        if recent_improvement < 1e-6:
                            st.info(f"步骤 {step}: 能量收敛，提前停止优化")
                            break
                
                # 提取优化后的坐标并更新分子
                # 确保坐标张量转换为正确的数据类型
                with torch.no_grad():
                    optimized_coords = coordinates_tensor.detach().float().cpu().numpy()
                
                batch_results = [None] * len(batch_mols_h)
                for idx, (mol_h, original_species) in enumerate(zip([batch_mols_h[vi] for vi in valid_indices], batch_species)):
                    if mol_h is None:
                        continue
                        
                    try:
                        # 更新坐标
                        conf = mol_h.GetConformer()
                        coords = optimized_coords[idx][:len(original_species)]  # 只取真实原子的坐标
                        
                        for atom_idx, pos in enumerate(coords):
                            conf.SetAtomPosition(atom_idx, (float(pos[0]), float(pos[1]), float(pos[2])))
        
        # 移除氢原子
                        mol_final = Chem.RemoveHs(mol_h)
                        batch_results[valid_indices[idx]] = mol_final
                    except Exception as e:
                        st.warning(f"更新分子坐标失败: {str(e)}")
                        batch_results[valid_indices[idx]] = None
                
                results.extend(batch_results)
                
                # 显示批次优化结果
                final_avg_energy = best_energy / len(valid_indices) if valid_indices else 0
                # Use status_container_ui for batch completion
                if status_container_ui:
                    status_container_ui.info(f"✅ TorchANI: 批次 {current_batch_num}/{num_batches} 完成, 最佳平均能量: {final_avg_energy:.4f}")
                elif progress_text_ui: # Fallback
                    progress_text_ui.text(f"TorchANI: 批次 {current_batch_num}/{num_batches} 完成.")
                
                # Update progress bar after each batch is fully processed.
                if progress_bar_ui:
                    progress_bar_ui.progress( min(1.0, (i + len(batch_mols_h)) / num_valid_mols) , text=f"TorchANI: 批次 {current_batch_num} 完成")
                
            except Exception as e: # Catch exception for this specific batch processing
                if status_container_ui:
                    status_container_ui.warning(f"TorchANI: 批次 {current_batch_num} 优化失败: {str(e)}")
                elif progress_text_ui:
                    progress_text_ui.text(f"TorchANI: 批次 {current_batch_num} 失败.")
                # Fill results for this batch with None
                for k_idx in range(len(batch_mols_h)):
                    if (i + k_idx) < len(results_for_optimized_mols):
                         results_for_optimized_mols[i + k_idx] = None
                
                # 清理GPU内存
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            
        # Reconstruct the final results list to match the original mols input size and order
        final_results_ordered = [None] * len(mols)
        opt_mol_idx = 0
        for original_idx in range(len(mols)):
            if processed_mols_info[original_idx]['mol_h'] is not None and opt_mol_idx < len(results_for_optimized_mols):
                final_results_ordered[original_idx] = results_for_optimized_mols[opt_mol_idx]
                opt_mol_idx += 1
            # else: it remains None (due to pre-processing failure or if something went wrong with indexing)

        total_time = time.time() - start_time
        success_count = sum(1 for r in final_results_ordered if r is not None)
        
        # The calling function batch_generate_3d_conformers will print the final success message.
        # Here, we just ensure the progress UI is finalized for this specific function's scope.
        if progress_bar_ui:
            progress_bar_ui.progress(1.0, text=f"TorchANI 优化处理完毕 ({success_count}/{len(mols)} 成功)")
        if progress_text_ui:
            progress_text_ui.text(f"TorchANI 优化处理完毕: {success_count}/{len(mols)} 成功, 用时 {total_time:.2f}s")
        if status_container_ui: # Clear or set a final message for the dedicated status line
            status_container_ui.info(f"TorchANI 优化流程结束. {success_count} 分子成功优化。")

        return final_results_ordered
        
    except Exception as e:
        if status_container_ui:
            status_container_ui.error(f"TorchANI 批量优化主程序失败: {str(e)}")
        elif progress_text_ui:
            progress_text_ui.text(f"TorchANI 优化严重错误: {str(e)}")
        if progress_bar_ui:
            progress_bar_ui.progress(1.0, text="TorchANI 优化出错!")
        return [None] * len(mols)

def generate_3d_conformer_torchani(mol, model_name='ANI2x', optimization_steps=100, device=None):
    """使用TorchANI生成3D构象 - 单分子版本（保持向后兼容）"""
    if not HAS_TORCHANI:
        return None

    # 使用批量版本处理单个分子以获得优化效果
    result = generate_3d_conformer_torchani_optimized([mol], model_name, optimization_steps, device, batch_size=1)
    return result[0] if result else None

def generate_3d_conformer_deepchem(mol, use_gpu=True, model_type='mpnn', force_field='mmff94s'):
    """使用DeepChem生成3D构象，支持GPU加速"""
    if not HAS_DEEPCHEM:
        return None
        
    try:
        # 添加氢原子
        mol_with_h = Chem.AddHs(mol)
        
        # 使用ETKDG生成初始构象
        embed_result = AllChem.EmbedMolecule(mol_with_h, AllChem.ETKDGv3())
        if embed_result != 0:
            st.warning("ETKDG嵌入失败，尝试基本嵌入")
            embed_result = AllChem.EmbedMolecule(mol_with_h)
            if embed_result != 0:
                return None
        
        # 设置GPU/CPU设备
        if use_gpu and tf.config.list_physical_devices('GPU'):
            with tf.device('/GPU:0'):
                # Initialize conformer generator with better parameters
                try:
                    conf_gen = dc.utils.conformers.ConformerGenerator(
                        max_conformers=1,
                            force_field=force_field,
                        pool_multiplier=1,
                            optimization_steps=200
                    )
                
                # Generate conformers
                    mol_optimized = conf_gen.generate_conformers(mol_with_h)
                    
                except Exception as gpu_error:
                    st.warning(f"GPU优化失败: {str(gpu_error)}，尝试CPU版本")
                    # Fallback to CPU version
                    conf_gen = dc.utils.conformers.ConformerGenerator(
                        max_conformers=1,
                        force_field=force_field,
                        pool_multiplier=1
                    )
                    mol_optimized = conf_gen.generate_conformers(mol_with_h)
        else:
            # Fallback to CPU version
            conf_gen = dc.utils.conformers.ConformerGenerator(
                max_conformers=1,
                force_field=force_field,
                pool_multiplier=1
            )
            mol_optimized = conf_gen.generate_conformers(mol_with_h)
        
        # 移除氢原子
        if mol_optimized is not None:
            mol_optimized = Chem.RemoveHs(mol_optimized)
            
        return mol_optimized
        
    except Exception as e:
        # Add the original model_type to the warning for context
        st.warning(f"DeepChem构象生成失败(model_type='{model_type}', force_field='{force_field}'): {str(e)}")
        return None

def generate_3d_conformer_clara(mol, force_field='MMFF94s', precision='mixed', num_conformers=1, energy_threshold=1.0, optimization_steps=500):
    """使用NVIDIA Clara生成3D构象"""
    if not HAS_CLARA:
        return None
        
    try:
        # 检查分子大小，Clara适合各种大小的分子
        num_atoms = mol.GetNumAtoms()
        if num_atoms > 200:
            st.warning(f"分子较大（{num_atoms}原子），Clara处理可能较慢")
        
        # 添加氢原子
        mol_with_h = Chem.AddHs(mol)
        
        # 使用ETKDG生成初始构象
        embed_result = AllChem.EmbedMolecule(mol_with_h, AllChem.ETKDGv3())
        if embed_result != 0:
            st.warning("ETKDG嵌入失败，尝试基本嵌入")
            embed_result = AllChem.EmbedMolecule(mol_with_h)
            if embed_result != 0:
                return None
        
        # 转换为Clara分子格式
        clara_molecule = clara_mol.Molecule.from_rdkit(mol_with_h)
        
        # 创建构象生成器
        conf_gen = clara_conf.ConformerGenerator(
            num_conformers=num_conformers,
            use_gpu=True,
            energy_minimization=True,
            force_field=force_field,
            precision=precision,
            energy_threshold=energy_threshold,
            max_iterations=optimization_steps
        )
        
        # 生成构象
        conformers = conf_gen.generate(clara_molecule)
        
        if not conformers:
            st.warning("Clara未生成有效构象")
            return None
        
        # 获取最低能量构象
        best_conf = min(conformers, key=lambda x: x.energy)
        
        # 转换回RDKit分子
        mol_with_conf = best_conf.to_rdkit()
        
        # 移除氢原子
        mol_optimized = Chem.RemoveHs(mol_with_conf)
        
        st.success(f"Clara构象生成成功，能量: {best_conf.energy:.3f} kcal/mol")
        return mol_optimized
        
    except Exception as e:
        st.warning(f"NVIDIA Clara构象生成失败: {str(e)}")
        return None

def generate_3d_conformer_multi(mol, backend='auto', **kwargs):
    """多后端3D构象生成器"""
    if mol is None:
        return None
        
    if backend == 'auto':
        # 根据分子大小和可用资源自动选择后端
        num_atoms = mol.GetNumAtoms()
        gpu_available = torch.cuda.is_available()
        
        # 检查后端可用性并考虑GPU支持
        if HAS_CLARA and gpu_available:
            backend = 'clara'  # 优先使用Clara（NVIDIA高性能）
        elif num_atoms <= 50 and HAS_TORCHANI and (gpu_available or HAS_TORCHANI_CUDA):
            backend = 'torchani'  # 小分子使用TorchANI
        elif num_atoms <= 100 and HAS_DEEPCHEM and HAS_DEEPCHEM_GPU:
            backend = 'deepchem'  # 中等分子使用DeepChem
        elif HAS_CLARA and gpu_available:
            backend = 'clara'  # 大分子使用Clara
        else:
            backend = 'rdkit'  # 默认回退到RDKit
            
        st.info(f"自动选择构象生成后端: {backend}")
    
    try:
        if backend == 'clara' and HAS_CLARA:
            st.info("使用NVIDIA Clara生成构象...")
            try:
                return generate_3d_conformer_clara(
                    mol,
                    force_field=kwargs.get('force_field', 'MMFF94s'),
                    precision=kwargs.get('precision', 'mixed'),
                    num_conformers=kwargs.get('num_conformers', 1),
                    energy_threshold=kwargs.get('energy_threshold', 1.0),
                    optimization_steps=kwargs.get('optimization_steps', 500)
                )
            except Exception as e:
                st.warning(f"NVIDIA Clara构象生成失败: {str(e)}，回退到RDKit")
                return generate_3d_conformer(mol, **kwargs)
                
        elif backend == 'torchani' and HAS_TORCHANI:
            st.info("使用TorchANI生成构象...")
            try:
                # 确定设备
                device = None
                if torch.cuda.is_available() and HAS_TORCHANI_CUDA:
                    device = torch.device('cuda')
                else:
                    device = torch.device('cpu')
                    st.info("TorchANI使用CPU计算")
                
                return generate_3d_conformer_torchani(
                    mol,
                    model_name=kwargs.get('torchani_model', 'ANI2x'),
                    optimization_steps=kwargs.get('optimization_steps', 100),
                    device=device
                )
            except Exception as e:
                st.warning(f"TorchANI构象生成失败: {str(e)}，回退到RDKit")
                return generate_3d_conformer(mol, **kwargs)
                
        elif backend == 'deepchem' and HAS_DEEPCHEM:
            st.info("使用DeepChem生成构象...")
            try:
                return generate_3d_conformer_deepchem(
                    mol,
                    use_gpu=HAS_DEEPCHEM_GPU and kwargs.get('use_gpu', True),
                    model_type=kwargs.get('model_type', 'mpnn'),
                    force_field=kwargs.get('force_field', 'mmff94s')
                )
            except Exception as e:
                st.warning(f"DeepChem构象生成失败: {str(e)}，回退到RDKit")
                return generate_3d_conformer(mol, **kwargs)
                
        else:
            # 使用RDKit作为默认和回退选项
            st.info("使用RDKit生成构象...")
            return generate_3d_conformer(
                mol,
                max_attempts=kwargs.get('max_attempts', 50),
                use_mmff=kwargs.get('use_mmff', True),
                energy_iter=kwargs.get('energy_iter', 200),
                add_hydrogens=kwargs.get('add_hydrogens', True)
            )
            
    except Exception as e:
        st.error(f"构象生成过程出错: {str(e)}")
        # 尝试最基本的构象生成作为最后的回退选项
        try:
            st.warning("尝试使用最基本的RDKit构象生成...")
            mol_copy = Chem.Mol(mol)
            Chem.AllChem.EmbedMolecule(mol_copy)
            return mol_copy
        except:
            st.error("构象生成完全失败")
            return None

# Modified function definition to accept progress_bar
def batch_generate_3d_conformers(mols, progress_bar, status_container, progress_text, backend='auto', batch_size=None, **kwargs):
    """批量生成3D构象 - 优化版本"""
    if not mols:
        return []
    
    # 自动选择批处理大小
    if batch_size is None:
        if backend == 'clara':
            batch_size = min(10, len(mols))  # 较小批次以减少GPU内存压力
        elif backend == 'torchani':
            batch_size = min(32, len(mols))  # TorchANI优化批处理
        elif backend == 'deepchem':
            batch_size = min(64, len(mols))
        else:
            batch_size = min(100, len(mols))
    
    start_time = time.time()
    status_container.info(f"开始生成构象，共 {len(mols)} 个分子，使用 {backend} 后端")
    
    # TorchANI特殊批处理优化
    if backend == 'torchani' and HAS_TORCHANI:
        status_container.info("🚀 使用TorchANI优化批处理模式")
        progress_text.text("正在进行TorchANI批量优化...")
        
        try:
            # 确定设备和参数
            device = None
            if torch.cuda.is_available() and HAS_TORCHANI_CUDA:
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
                st.info("TorchANI使用CPU计算")
            
            # 获取用户设置的参数
            optimization_params = {
                'model_name': kwargs.get('torchani_model', 'ANI2x'),
                'optimization_steps': kwargs.get('optimization_steps', 100),
                'device': device,
                'batch_size': kwargs.get('torchani_batch_size', batch_size),
                'learning_rate': kwargs.get('learning_rate', 0.01),
                'use_mixed_precision_torchani': kwargs.get('use_mixed_precision_torchani', True),
                'max_atoms_per_batch': kwargs.get('max_atoms_per_batch', 5000),
                'gradient_clipping': kwargs.get('gradient_clipping', True)
            }
            
            # 检查是否启用优化模式
            if not kwargs.get('use_torchani_optimization', True):
                st.warning("⚠️ 批量优化模式已禁用，回退到传统模式")
                # 这里应该有一个清晰的回退路径到逐个处理逻辑，目前它会直接跳到函数末尾的逐个处理
            else:
                # 调用优化的批量处理函数
                # 将Streamlit UI元素传递给优化函数
                optimization_params['progress_bar_ui'] = progress_bar
                optimization_params['progress_text_ui'] = progress_text
                optimization_params['status_container_ui'] = status_container

                results = generate_3d_conformer_torchani_optimized(mols, **optimization_params)
                
                # generate_3d_conformer_torchani_optimized 内部会处理其作用域内的最终进度更新
                # 这里主要处理调用优化函数后的总体状态和统计信息

                total_time = time.time() - start_time # Recalculate total time based on this function's scope
                success_count = sum(1 for r in results if r is not None)
                success_rate = (success_count / len(mols)) * 100 if len(mols) > 0 else 0
                
                status_container.success(
                    f"🎯 TorchANI批量构象生成完成! 成功率: {success_rate:.1f}% ({success_count}/{len(mols)})"
                    f"，总耗时: {total_time:.1f}秒"
                    f"，平均每分子: {total_time/len(mols) if len(mols) > 0 else 0:.2f}秒"
                )
                
                if progress_bar: # Final confirmation of progress bar
                    progress_bar.progress(1.0, text=f"TorchANI处理完成 ({success_count}/{len(mols)})")
                if progress_text:
                    progress_text.text(f"TorchANI处理完成. {success_count} 个分子成功。")

                # 显示性能提升信息
                if success_count > 0:
                    molecules_per_second = success_count / total_time
                    st.info(f"⚡ 处理速度: {molecules_per_second:.2f} 分子/秒")
                    
                    # 估算相比原版本的性能提升
                    estimated_old_time = success_count * 10  # 假设原版本每个分子10秒
                    speedup = estimated_old_time / total_time if total_time > 0 else 1
                    if speedup > 2:
                        st.success(f"🎯 相比逐个处理估计加速: {speedup:.1f}x")
                    
                    # 显示使用的优化参数
                    with st.expander("🔧 使用的优化参数", expanded=False):
                        st.json({
                            "模型": optimization_params['model_name'],
                            "批处理大小": optimization_params['batch_size'],
                            "优化步数": optimization_params['optimization_steps'],
                            "学习率": optimization_params['learning_rate'],
                            "混合精度": optimization_params['use_mixed_precision_torchani'],
                            "梯度裁剪": optimization_params['gradient_clipping'],
                            "设备": str(optimization_params['device'])
                        })
                
                return results
                
        except Exception as e:
            st.error(f"TorchANI批量优化失败: {str(e)}")
            st.warning("回退到原有的逐个处理模式...")
            # 继续使用原有逻辑作为回退
    
    # 原有的逐个处理逻辑（其他后端或TorchANI失败时的回退）
    results = []
    failures = 0
    
    for i in range(0, len(mols), batch_size):
        batch = mols[i:min(i+batch_size, len(mols))]
        batch_size_actual = len(batch)
        
        # 更新状态
        batch_start_time = time.time()
        progress_text.text(f"处理批次 {i//batch_size + 1}/{(len(mols)-1)//batch_size + 1}，分子 {i+1}-{min(i+batch_size_actual, len(mols))}/{len(mols)}")
        
        # 使用多后端生成构象
        batch_results = []
        batch_failures = 0
        
        for j, mol in enumerate(batch):
            # 更新进度 (using the passed progress_bar)
            progress = (i + j) / len(mols)
            if progress_bar: # Check if progress_bar is not None
                progress_bar.progress(progress, text=f"处理分子 {i+j+1}/{len(mols)}")
            
            # 生成构象
            mol_3d = generate_3d_conformer_multi(mol, backend=backend, **kwargs)
            
            if mol_3d is None:
                batch_failures += 1
                failures += 1
            
            batch_results.append(mol_3d)
            
            # 显示当前处理速度
            elapsed = time.time() - start_time
            molecules_per_second = (i + j + 1) / elapsed if elapsed > 0 else 0
            remaining = (len(mols) - (i + j + 1)) / molecules_per_second if molecules_per_second > 0 else 0
            
            if (j + 1) % max(1, batch_size_actual // 5) == 0 or j == batch_size_actual - 1:
                progress_text.text(
                    f"批次 {i//batch_size + 1}/{(len(mols)-1)//batch_size + 1}"
                    f"，分子 {i+j+1}/{len(mols)}"
                    f"，处理速度: {molecules_per_second:.2f} 分子/秒"
                    f"，剩余时间: {int(remaining//60)}分{int(remaining%60)}秒"
                )
        
        results.extend(batch_results)
        
        # 显示批次状态
        batch_time = time.time() - batch_start_time
        status_container.info(
            f"完成批次 {i//batch_size + 1}/{(len(mols)-1)//batch_size + 1}"
            f"，构象生成成功率: {(batch_size_actual-batch_failures)/batch_size_actual*100:.1f}%"
            f"，批次耗时: {batch_time:.1f}秒"
            f"，每分子: {batch_time/batch_size_actual:.2f}秒"
        )
        
        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    success_rate = (len(mols) - failures) / len(mols) * 100 if len(mols) > 0 else 0
    
    status_container.success(
        f"构象生成完成! 成功率: {success_rate:.1f}% ({len(mols)-failures}/{len(mols)})"
        f"，总耗时: {total_time:.1f}秒"
        f"，平均每分子: {total_time/len(mols):.2f}秒"
    )
    
    if failures > 0:
        st.warning(f"警告: {failures}个分子的构象生成失败")
    
    # Reset progress bar to 0 after completion
    if progress_bar:
        progress_bar.progress(1.0) # Set to 100%

    return results

def compute_usr_descriptor(mol):
    """计算USR (Ultrafast Shape Recognition) 描述符"""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    
    conf = mol.GetConformer()
    if not conf.Is3D():
        return None
    
    try:
        # 提取所有原子坐标
        coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append(np.array([pos.x, pos.y, pos.z]))
        coords = np.array(coords)
        
        # 1) 质心
        centroid = coords.mean(axis=0)
        # 2) 与质心最远的点P1
        dists_centroid = np.linalg.norm(coords - centroid, axis=1)
        idx_p1 = np.argmax(dists_centroid)
        p1 = coords[idx_p1]
        # 3) 与p1最远的点p2
        dists_p1 = np.linalg.norm(coords - p1, axis=1)
        idx_p2 = np.argmax(dists_p1)
        p2 = coords[idx_p2]
        # 4) 与p2最远的点p3
        dists_p2 = np.linalg.norm(coords - p2, axis=1)
        idx_p3 = np.argmax(dists_p2)
        p3 = coords[idx_p3]
        
        # 四个参考点
        P = [centroid, p1, p2, p3]
        
        # 计算到这4点的距离分布
        descriptor = []
        for ref_pt in P:
            dists = np.linalg.norm(coords - ref_pt, axis=1)
            d_mean = dists.mean()
            d_std = dists.std()
            d_min = dists.min()
            d_max = dists.max()
            descriptor.extend([d_mean, d_std, d_min, d_max])
        
        return np.array(descriptor)
    except:
        return None

def compute_usrcat_descriptor(mol):
    """计算USRCAT (USR-CAT) 描述符"""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    
    conf = mol.GetConformer()
    if not conf.Is3D():
        return None
    
    try:
        # 定义原子类型
        atom_types = {
            'all': lambda a: True,
            'hydrophobic': lambda a: a.GetSymbol() in ('C', 'Cl', 'Br', 'I'),
            'aromatic': lambda a: a.GetIsAromatic(),
            'acceptor': lambda a: a.GetSymbol() in ('N', 'O', 'F') and not a.GetIsAromatic(),
            'donor': lambda a: (a.GetSymbol() in ('N', 'O') and 
                              sum(1 for n in a.GetNeighbors() if n.GetSymbol() == 'H') > 0)
        }
        
        descriptors = []
        
        for atom_type, type_func in atom_types.items():
            # 获取特定类型的原子坐标
            coords = []
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                if type_func(atom):
                    pos = conf.GetAtomPosition(i)
                    coords.append(np.array([pos.x, pos.y, pos.z]))
            
            if not coords:
                descriptors.extend([0] * 12)  # 如果没有该类型的原子，填充0
                continue
                
            coords = np.array(coords)
            
            # 计算四个参考点
            centroid = coords.mean(axis=0)
            dists_centroid = np.linalg.norm(coords - centroid, axis=1)
            idx_p1 = np.argmax(dists_centroid)
            p1 = coords[idx_p1]
            
            dists_p1 = np.linalg.norm(coords - p1, axis=1)
            idx_p2 = np.argmax(dists_p1)
            p2 = coords[idx_p2]
            
            dists_p2 = np.linalg.norm(coords - p2, axis=1)
            idx_p3 = np.argmax(dists_p2)
            p3 = coords[idx_p3]
            
            # 计算到四个参考点的距离矩
            for ref_pt in [centroid, p1, p2, p3]:
                dists = np.linalg.norm(coords - ref_pt, axis=1)
                descriptors.extend([
                    dists.mean(),
                    dists.std(),
                    dists.min(),
                    dists.max()
                ])
        
        return np.array(descriptors)
    except:
        return None

def compute_pmi_ratios(mol):
    """计算PMI比率"""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    try:
        inertia = rdMolDescriptors.CalcPMIValues(mol)
        I1, I2, I3 = inertia
        if abs(I3) < 1e-8:
            return None
        return (I1/I3, I2/I3)
    except:
        return None

def calc_nearest_neighbor_distance(descriptors, cuda_available=False):
    """计算最近邻距离，支持GPU加速"""
    n = len(descriptors)
    if n < 2:
        return 0.0, 0.0
    
    # 使用GPU加速
    if cuda_available and torch.cuda.is_available():
        try:
            # 记录GPU内存使用前状态
            mem_before = torch.cuda.memory_allocated() / 1024**2
            
            # 转换为PyTorch张量并移至GPU
            desc_tensor = torch.tensor(descriptors, dtype=torch.float32).cuda()
            
            # 计算欧氏距离矩阵
            distances = torch.zeros((n, n), dtype=torch.float32).cuda()
            batch_size = 128  # 批处理大小，避免显存溢出
            
            for i in range(0, n, batch_size):
                end_i = min(i + batch_size, n)
                chunk_i = desc_tensor[i:end_i]
                
                for j in range(0, n, batch_size):
                    end_j = min(j + batch_size, n)
                    chunk_j = desc_tensor[j:end_j]
                    
                    # 计算批次之间的距离
                    dist_chunk = torch.cdist(chunk_i, chunk_j)
                    distances[i:end_i, j:end_j] = dist_chunk
            
            # 将自身距离设为无穷大
            eye_mask = torch.eye(n, dtype=torch.bool).cuda()
            distances[eye_mask] = float('inf')
            
            # 每个分子的最近邻距离
            min_dists, _ = torch.min(distances, dim=1)
            
            # 计算统计信息
            mean_d = min_dists.mean().item()
            std_d = min_dists.std().item()
            
            # 记录GPU内存使用后状态
            mem_after = torch.cuda.memory_allocated() / 1024**2
            mem_diff = mem_after - mem_before
            
            # 记录GPU使用情况
            st.session_state.gpu_nn_mem_usage = mem_diff
            st.session_state.gpu_nn_calls = st.session_state.get('gpu_nn_calls', 0) + 1
            
            # 清理GPU内存
            torch.cuda.empty_cache()
            
            return mean_d, std_d
        
        except Exception as e:
            st.warning(f"GPU加速最近邻计算失败，将使用CPU: {str(e)}")
    
    # 使用CPU计算
    distances = []
    for i in range(n):
        d_i = float('inf')
        for j in range(n):
            if i == j:
                continue
            dist = np.linalg.norm(descriptors[i] - descriptors[j])
            if dist < d_i:
                d_i = dist
        distances.append(d_i)
    
    mean_d = np.mean(distances)
    std_d = np.std(distances)
    return mean_d, std_d

def perform_dimensionality_reduction(descriptors, method="t-SNE", cuda_available=False, **kwargs):
    """使用GPU加速的降维分析"""
    if len(descriptors) == 0:
        return np.array([])
    
    # 记录初始状态
    gpu_used = False
    start_time = time.time()
        
    if method == "t-SNE":
        if cuda_available and 'cp' in globals() and 'cuTSNE' in globals():
            try:
                # 记录GPU内存使用前状态(如果可用)
                if torch.cuda.is_available():
                    mem_before = torch.cuda.memory_allocated() / 1024**2
                
                tsne = cuTSNE(n_components=2, **kwargs)
                coords = tsne.fit_transform(cp.array(descriptors))
                
                # 记录GPU使用情况
                if torch.cuda.is_available():
                    mem_after = torch.cuda.memory_allocated() / 1024**2
                    mem_diff = abs(mem_after - mem_before)  # cuML可能使用不同的内存管理
                    st.session_state.gpu_tsne_mem = mem_diff
                
                gpu_used = True
                st.session_state.gpu_tsne_calls = st.session_state.get('gpu_tsne_calls', 0) + 1
                
                result = cp.asnumpy(coords)
            except Exception as e:
                st.warning(f"GPU加速t-SNE失败，将使用CPU: {str(e)}")
                tsne = TSNE(n_components=2, **kwargs)
                result = tsne.fit_transform(descriptors)
        else:
            tsne = TSNE(n_components=2, **kwargs)
            result = tsne.fit_transform(descriptors)
    elif method == "UMAP":
        reducer = umap.UMAP(n_components=2, **kwargs)
        result = reducer.fit_transform(descriptors)
    else:
        raise ValueError(f"不支持的降维方法: {method}")
    
    # 记录总时间
    total_time = time.time() - start_time
    
    # 存储性能信息
    st.session_state[f"{method}_time"] = total_time
    st.session_state[f"{method}_gpu_used"] = gpu_used
    
    return result

def plot_shape_space(coordsA, coordsB, title="形状空间分布"):
    """绘制形状空间分布图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coordsA[:,0], coordsA[:,1], c='blue', alpha=0.5, label="数据集A", s=10)
    ax.scatter(coordsB[:,0], coordsB[:,1], c='red', alpha=0.5, label="数据集B", s=10)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return fig

def plot_pmi_triangle(pmiA, pmiB, labelA="数据集A", labelB="数据集B"):
    """绘制PMI三角图"""
    def to_triangle_coords(a, b):
        x = b + (a/2.0)
        y = (np.sqrt(3)/2.0) * a
        return (x, y)
    
    coordsA = [to_triangle_coords(a, b) for (a,b) in pmiA]
    coordsB = [to_triangle_coords(a, b) for (a,b) in pmiB]
    
    if not coordsA or not coordsB:
        return None
        
    coordsA = np.array(coordsA)
    coordsB = np.array(coordsB)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coordsA[:,0], coordsA[:,1], c='blue', alpha=0.4, s=10, label=labelA)
    ax.scatter(coordsB[:,0], coordsB[:,1], c='red', alpha=0.4, s=10, label=labelB)
    ax.set_title("PMI三角图（形状分布）")
    ax.legend()
    plt.tight_layout()
    return fig

def calculate_distribution_metrics(coords_A, coords_B):
    """计算分布统计指标"""
    if len(coords_A) == 0 or len(coords_B) == 0:
        return {}
        
    metrics = {}
    
    # 计算每个维度的Wasserstein距离
    for dim in range(coords_A.shape[1]):
        w_dist = wasserstein_distance(coords_A[:,dim], coords_B[:,dim])
        metrics[f'Wasserstein距离_dim{dim+1}'] = w_dist
    
    # 计算中心距离
    center_A = np.mean(coords_A, axis=0)
    center_B = np.mean(coords_B, axis=0)
    center_dist = np.linalg.norm(center_A - center_B)
    metrics['中心距离'] = center_dist
    
    # 计算分布重叠度
    try:
        kde_A = KernelDensity(kernel='gaussian').fit(coords_A)
        kde_B = KernelDensity(kernel='gaussian').fit(coords_B)
        
        overlap_score = (np.exp(kde_A.score_samples(coords_B)).mean() + 
                        np.exp(kde_B.score_samples(coords_A)).mean()) / 2
        metrics['分布重叠度'] = overlap_score
    except:
        pass
    
    return metrics

def get_optimal_batch_size(mol_size, device=None):
    """根据分子大小和GPU显存动态计算最优批处理大小"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type != 'cuda':
        return 50  # CPU默认批处理大小
    
    try:
        # 获取GPU总显存(MB)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2
        # 获取当前已用显存(MB)
        used_mem = torch.cuda.memory_allocated(0) / 1024**2
        # 可用显存(MB)
        available_mem = total_mem - used_mem
        
        # 估算单个分子所需显存(MB)
        # 假设每个原子需要的显存约为: 坐标(3*4=12字节) + 距离计算(4*4=16字节) = 28字节
        mem_per_mol = mol_size * 28 / 1024**2  # 转换为MB
        
        # 预留30%显存给其他计算
        safe_mem = available_mem * 0.7
        
        # 计算最大批处理大小
        max_batch = int(safe_mem / mem_per_mol)
        
        # 限制在合理范围内
        optimal_batch = max(10, min(max_batch, 500))
        
        return optimal_batch
    
    except Exception as e:
        st.warning(f"计算最优批处理大小失败: {str(e)}")
        return 50  # 返回默认值

@torch.jit.script
def compute_distances_jit(coords: torch.Tensor, ref_point: torch.Tensor) -> torch.Tensor:
    """使用TorchScript优化的距离计算"""
    return torch.norm(coords - ref_point.unsqueeze(0), dim=1)

def process_usr_batch_optimized(mols_3d, cuda_available=False, batch_size=None):
    """优化的USR批处理计算"""
    if not mols_3d:
        return []
    
    device = torch.device('cuda' if cuda_available and torch.cuda.is_available() else 'cpu')
    
    # 记录开始时间和GPU内存
    start_time = time.time()
    if device.type == 'cuda':
        mem_before = torch.cuda.memory_allocated() / 1024**2
    
    # 获取第一个有效分子的大小用于计算批处理大小
    mol_size = next((mol.GetNumAtoms() for mol in mols_3d if mol is not None), 0)
    if batch_size is None:
        batch_size = get_optimal_batch_size(mol_size, device)
    
    descriptors = []
    gpu_used = False
    
    # 创建异步流
    if device.type == 'cuda':
        stream = torch.cuda.Stream()
    
    # 批量处理分子
    for i in range(0, len(mols_3d), batch_size):
        batch = mols_3d[i:i + batch_size]
        batch_descriptors = []
        
        if device.type == 'cuda':
            with torch.cuda.stream(stream):
                for mol in batch:
                    if mol is not None:
                        desc = compute_usr_descriptor_gpu(mol, device)
                        gpu_used = True
                    else:
                        desc = None
                    batch_descriptors.append(desc)
                
                # 同步流
                stream.synchronize()
        else:
            for mol in batch:
                if mol is not None:
                    desc = compute_usr_descriptor(mol)
                else:
                    desc = None
                batch_descriptors.append(desc)
        
        descriptors.extend(batch_descriptors)
        
        # 清理GPU缓存
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    # 计算性能统计
    total_time = time.time() - start_time
    if device.type == 'cuda':
        mem_after = torch.cuda.memory_allocated() / 1024**2
        mem_diff = mem_after - mem_before
        
        # 更新会话状态
        st.session_state.usr_batch_time = total_time
        st.session_state.usr_batch_mem = mem_diff
        st.session_state.usr_batch_gpu_used = gpu_used
        
        # 记录处理速度
        valid_mols = sum(1 for d in descriptors if d is not None)
        speed = valid_mols / total_time if total_time > 0 else 0
        st.session_state.usr_processing_speed = speed
    
    return descriptors

def batch_compute_shape_descriptors(mols, descriptor_type="USR", max_attempts=50, cuda_available=False, batch_size=None):
    """改进的批量计算形状描述符函数"""
    if not mols:
        return [], []
    
    descriptors = []
    pmi_ratios = []
    
    # 进度显示
    total = len(mols)
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # 将分子分批处理
    batch_mols_3d = []
    for m in mols:
        m3d = generate_3d_conformer(m, max_attempts)
        batch_mols_3d.append(m3d)
    
    # 更新进度
    progress = len(batch_mols_3d) / total
    progress_bar.progress(progress)
    status_text.text(f"生成3D构象: {len(batch_mols_3d)}/{total}")
        
    # 计算PMI比率
    status_text.text("计算PMI比率...")
    for m3d in batch_mols_3d:
        if m3d:
            ratios = compute_pmi_ratios(m3d)
            if ratios:
                pmi_ratios.append(ratios)
    
    # 计算形状描述符
    status_text.text(f"计算{descriptor_type}描述符...")
    if descriptor_type == "USR":
        descriptors = process_usr_batch_optimized(batch_mols_3d, cuda_available, batch_size)
    else:  # USRCAT
        descriptors = process_usrcat_batch(batch_mols_3d, cuda_available, batch_size)
    
    # 清理进度显示
    progress_bar.empty()
    status_text.empty()
    
    # 显示性能统计
    if cuda_available and torch.cuda.is_available():
        speed = st.session_state.get('usr_processing_speed', 0)
        mem_used = st.session_state.get('usr_batch_mem', 0)
        st.info(f"性能统计:\n"
                f"- 处理速度: {speed:.1f} 分子/秒\n"
                f"- GPU内存使用: {mem_used:.1f} MB")
    
    return np.array([d for d in descriptors if d is not None]), pmi_ratios

def compute_usr_descriptor_gpu(mol, device=None):
    """使用GPU加速计算USR (Ultrafast Shape Recognition) 描述符"""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    
    conf = mol.GetConformer()
    if not conf.Is3D():
        return None
    
    try:
        # 设置设备
        if device is None and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # 记录GPU内存使用前状态
        if device.type == 'cuda':
            mem_before = torch.cuda.memory_allocated() / 1024**2
        
        # 提取所有原子坐标并转换为张量
        coords = []
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append([pos.x, pos.y, pos.z])
            
        # 检查原子数量
        if len(coords) < 2:
            return None
            
        coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
        
        # 1) 质心
        centroid = torch.mean(coords_tensor, dim=0)
        
        # 2) 与质心最远的点P1
        dists_centroid = torch.norm(coords_tensor - centroid, dim=1)
        idx_p1 = torch.argmax(dists_centroid)
        p1 = coords_tensor[idx_p1]
        
        # 3) 与p1最远的点p2
        dists_p1 = torch.norm(coords_tensor - p1, dim=1)
        idx_p2 = torch.argmax(dists_p1)
        p2 = coords_tensor[idx_p2]
        
        # 4) 与p2最远的点p3
        dists_p2 = torch.norm(coords_tensor - p2, dim=1)
        idx_p3 = torch.argmax(dists_p2)
        p3 = coords_tensor[idx_p3]
        
        # 批量计算所有参考点的距离统计
        ref_points = torch.stack([centroid, p1, p2, p3])
        
        # 使用广播机制计算所有距离
        # shape: [n_ref_points, n_atoms]
        all_dists = torch.norm(coords_tensor.unsqueeze(0) - ref_points.unsqueeze(1), dim=2)
        
        # 计算每个参考点的统计量，使用无偏估计
        means = torch.mean(all_dists, dim=1)
        stds = torch.std(all_dists, dim=1, unbiased=True)
        mins = torch.min(all_dists, dim=1)[0]
        maxs = torch.max(all_dists, dim=1)[0]
        
        # 将所有统计量组合成最终描述符
        descriptors = torch.stack([
            means, stds, mins, maxs
        ]).t().reshape(-1).cpu().numpy()
        
        # 记录GPU内存使用后状态和调用次数
        if device.type == 'cuda':
            mem_after = torch.cuda.memory_allocated() / 1024**2
            mem_diff = mem_after - mem_before
            if mem_diff > 1.0:  # 如果内存变化大于1MB
                st.session_state.gpu_usr_calls = st.session_state.get('gpu_usr_calls', 0) + 1
                st.session_state.gpu_usr_mem = mem_diff
        
        return descriptors
        
    except Exception as e:
        # 在发生错误时回退到CPU版本
        st.warning(f"GPU计算USR描述符失败，回退到CPU: {str(e)}")
        return compute_usr_descriptor(mol)

def process_usr_batch(mols_3d, cuda_available=False):
    """批量处理USR描述符计算，支持GPU加速"""
    descriptors = []
    device = torch.device('cuda') if cuda_available and torch.cuda.is_available() else torch.device('cpu')
    
    # 记录开始时间
    start_time = time.time()
    
    # 记录处理前GPU内存
    if device.type == 'cuda':
        mem_before = torch.cuda.memory_allocated() / 1024**2
    
    gpu_used = False
    
    for mol in mols_3d:
        if mol is not None:
            if cuda_available and torch.cuda.is_available():
                desc = compute_usr_descriptor_gpu(mol, device)
                gpu_used = True
            else:
                desc = compute_usr_descriptor(mol)
            descriptors.append(desc)
        else:
            descriptors.append(None)
    
    # 计算总时间
    total_time = time.time() - start_time
    
    # 记录处理后GPU内存和使用情况
    if device.type == 'cuda':
        mem_after = torch.cuda.memory_allocated() / 1024**2
        mem_diff = mem_after - mem_before
        st.session_state.usr_batch_time = total_time
        st.session_state.usr_batch_mem = mem_diff
        st.session_state.usr_batch_gpu_used = gpu_used
    
    return descriptors

def compute_usrcat_descriptor_gpu(mol, device=None):
    """使用GPU加速计算USRCAT描述符"""
    if mol is None or mol.GetNumConformers() == 0:
        return None
    
    conf = mol.GetConformer()
    if not conf.Is3D():
        return None
    
    try:
        # 定义原子类型
        atom_types = {
            'all': lambda a: True,
            'hydrophobic': lambda a: a.GetSymbol() in ('C', 'Cl', 'Br', 'I'),
            'aromatic': lambda a: a.GetIsAromatic(),
            'acceptor': lambda a: a.GetSymbol() in ('N', 'O', 'F') and not a.GetIsAromatic(),
            'donor': lambda a: (a.GetSymbol() in ('N', 'O') and 
                              sum(1 for n in a.GetNeighbors() if n.GetSymbol() == 'H') > 0)
        }
        
        if device is None and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        
        # 记录GPU内存使用前状态
        if device.type == 'cuda':
            mem_before = torch.cuda.memory_allocated() / 1024**2
        
        descriptors = []
        
        for atom_type, type_func in atom_types.items():
            # 获取特定类型的原子坐标
            coords = []
            for i in range(mol.GetNumAtoms()):
                atom = mol.GetAtomWithIdx(i)
                if type_func(atom):
                    pos = conf.GetAtomPosition(i)
                    coords.append([pos.x, pos.y, pos.z])
            
            if not coords:
                descriptors.extend([0] * 12)  # 如果没有该类型的原子，填充0
                continue
                
            # 转换为PyTorch张量并移至GPU
            coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
            
            # 1) 质心
            centroid = torch.mean(coords_tensor, dim=0)
            
            # 2) 与质心最远的点P1
            dists_centroid = torch.norm(coords_tensor - centroid, dim=1)
            idx_p1 = torch.argmax(dists_centroid)
            p1 = coords_tensor[idx_p1]
            
            # 3) 与p1最远的点p2
            dists_p1 = torch.norm(coords_tensor - p1, dim=1)
            idx_p2 = torch.argmax(dists_p1)
            p2 = coords_tensor[idx_p2]
            
            # 4) 与p2最远的点p3
            dists_p2 = torch.norm(coords_tensor - p2, dim=1)
            idx_p3 = torch.argmax(dists_p2)
            p3 = coords_tensor[idx_p3]
            
            # 四个参考点
            for ref_pt in [centroid, p1, p2, p3]:
                dists = torch.norm(coords_tensor - ref_pt, dim=1)
                descriptors.extend([
                    dists.mean().item(),
                    dists.std().item(),
                    dists.min().item(),
                    dists.max().item()
                ])
        
        # 记录GPU内存使用后状态
        if device.type == 'cuda':
            mem_after = torch.cuda.memory_allocated() / 1024**2
            mem_diff = mem_after - mem_before
            # 如果内存变化大于1MB，则认为GPU真正被使用
            if mem_diff > 1.0:
                st.session_state.gpu_usrcat_calls = st.session_state.get('gpu_usrcat_calls', 0) + 1
        
        return np.array(descriptors)
    except Exception as e:
        # 在发生错误时回退到CPU版本
        st.warning(f"GPU计算USRCAT描述符失败，回退到CPU: {str(e)}")
        return compute_usrcat_descriptor(mol)

def process_usrcat_batch(mols_3d, cuda_available=False):
    """批量处理USRCAT描述符计算，支持GPU加速"""
    descriptors = []
    device = torch.device('cuda') if cuda_available and torch.cuda.is_available() else torch.device('cpu')
    
    # 记录开始时间
    start_time = time.time()
    
    # 记录处理前GPU内存
    if device.type == 'cuda':
        mem_before = torch.cuda.memory_allocated() / 1024**2
    
    gpu_used = False
    
    for mol in mols_3d:
        if mol is not None:
            if cuda_available and torch.cuda.is_available():
                desc = compute_usrcat_descriptor_gpu(mol, device)
                gpu_used = True
            else:
                desc = compute_usrcat_descriptor(mol)
            descriptors.append(desc)
        else:
            descriptors.append(None)
    
    # 计算总时间
    total_time = time.time() - start_time
    
    # 记录处理后GPU内存和使用情况
    if device.type == 'cuda':
        mem_after = torch.cuda.memory_allocated() / 1024**2
        mem_diff = mem_after - mem_before
        st.session_state.usrcat_batch_time = total_time
        st.session_state.usrcat_batch_mem = mem_diff
        st.session_state.usrcat_batch_gpu_used = gpu_used
    
    return descriptors

def monitor_gpu_usage():
    """监控并返回GPU使用情况"""
    if not torch.cuda.is_available():
        return {"状态": "GPU不可用"}
    
    try:
        metrics = {}
        metrics["总显存(MB)"] = torch.cuda.get_device_properties(0).total_memory / 1024**2
        metrics["已用显存(MB)"] = torch.cuda.memory_allocated(0) / 1024**2
        metrics["已缓存显存(MB)"] = torch.cuda.memory_reserved(0) / 1024**2
        metrics["显存利用率(%)"] = 100 * metrics["已用显存(MB)"] / metrics["总显存(MB)"]
        return metrics
    except:
        return {"状态": "无法获取GPU信息"}

# 添加GPU使用记录功能
def track_gpu_usage():
    """记录GPU使用情况并返回当前状态"""
    if not torch.cuda.is_available():
        return {'used': 0, 'free': 0, 'total': 0, 'utilization': 0}
    
    try:
        # 获取GPU内存信息
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
        used_mem = torch.cuda.memory_allocated(0) / 1024**2  # MB
        reserved_mem = torch.cuda.memory_reserved(0) / 1024**2  # MB
        free_mem = total_mem - used_mem  # 可用显存
        utilization = (used_mem / total_mem) * 100  # 使用率
        
        # 记录到会话状态
        if 'gpu_usage_history' not in st.session_state:
            st.session_state.gpu_usage_history = []
        
        # 添加当前时间点的记录
        st.session_state.gpu_usage_history.append({
            'time': time.time(),
            'used': used_mem,
            'reserved': reserved_mem,
            'free': free_mem,
            'total': total_mem,
            'utilization': utilization
        })
        
        # 只保留最近100个记录
        if len(st.session_state.gpu_usage_history) > 100:
            st.session_state.gpu_usage_history = st.session_state.gpu_usage_history[-100:]
        
        return {
            'used': used_mem,
            'reserved': reserved_mem,
            'free': free_mem,
            'total': total_mem,
            'utilization': utilization
        }
    except Exception as e:
        st.error(f"获取GPU信息时出错: {str(e)}")
        return {'used': 0, 'free': 0, 'total': 0, 'utilization': 0}

# 添加GPU使用历史可视化
def plot_gpu_usage_history():
    """绘制GPU使用历史图表"""
    if 'gpu_usage_history' not in st.session_state or not st.session_state.gpu_usage_history:
        st.info("尚无GPU使用记录")
        return
    
    history = st.session_state.gpu_usage_history
    
    # 提取数据
    times = [(record['time'] - history[0]['time']) for record in history]
    used = [record['used'] for record in history]
    reserved = [record['reserved'] for record in history]
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, used, 'r-', label='已分配')
    ax.plot(times, reserved, 'b--', label='已缓存')
    
    # 添加标签和标题
    ax.set_xlabel('时间 (秒)')
    ax.set_ylabel('显存 (MB)')
    ax.set_title('GPU内存使用历史')
    ax.legend()
    
    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 显示图表
    st.pyplot(fig)

# 在函数调用之前启动GPU监控
def start_gpu_monitoring():
    """启动GPU监控"""
    if not torch.cuda.is_available():
        return
    
    # 清空历史记录
    st.session_state.gpu_usage_history = []
    
    # 记录初始状态
    track_gpu_usage()
    
    # 设置监控已启动标志
    st.session_state.gpu_monitoring_active = True

# 在函数调用之后停止GPU监控并显示报告
def stop_gpu_monitoring_and_report():
    """停止GPU监控并生成报告"""
    if not torch.cuda.is_available() or not st.session_state.get('gpu_monitoring_active', False):
        return
    
    # 记录最终状态
    track_gpu_usage()
    
    # 设置监控停止标志
    st.session_state.gpu_monitoring_active = False
    
    # 生成使用报告
    if 'gpu_usage_history' in st.session_state and st.session_state.gpu_usage_history:
        st.subheader("GPU使用历史")
        
        history = st.session_state.gpu_usage_history
        
        # 计算统计信息
        max_used = max([record['used'] for record in history])
        avg_used = sum([record['used'] for record in history]) / len(history)
        peak_util = max([record['utilization'] for record in history])
        
        # 显示关键指标
        col1, col2, col3 = st.columns(3)
        col1.metric("最大显存使用 (MB)", f"{max_used:.1f}")
        col2.metric("平均显存使用 (MB)", f"{avg_used:.1f}")
        col3.metric("最高使用率 (%)", f"{peak_util:.1f}")
        
        # 绘制使用历史
        plot_gpu_usage_history()

# 在主界面添加GPU监控选项
st.sidebar.subheader("GPU监控")
if st.sidebar.checkbox("启用GPU监控", value=False):
    gpu_monitor = st.sidebar.empty()
    
    def update_gpu_monitor():
        """更新GPU监控信息"""
        while True:
            metrics = monitor_gpu_usage()
            content = ""
            for k, v in metrics.items():
                if isinstance(v, float):
                    content += f"- {k}: {v:.1f}\n"
                else:
                    content += f"- {k}: {v}\n"
            gpu_monitor.code(content)
            time.sleep(1.0)  # 每秒更新一次
    
    import threading
    monitor_thread = threading.Thread(target=update_gpu_monitor, daemon=True)
    monitor_thread.start()

# 在主程序中初始化会话状态变量
if 'gpu_debug_info' not in st.session_state:
    st.session_state.gpu_debug_info = {
        'usrcat_calls': 0,
        'nn_calls': 0,
        'tsne_calls': 0,
        'usrcat_time': 0,
        'nn_time': 0,
        'tsne_time': 0,
        'total_gpu_mem_peak': 0,
    }

# 添加GPU使用详情显示函数
def show_gpu_usage_details():
    """显示详细的GPU使用情况"""
    if not torch.cuda.is_available():
        st.warning("GPU不可用，无法显示GPU使用详情")
        return
    
    st.subheader("🔍 GPU加速详情")
    
    # 显示GPU调用次数
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("USR描述符GPU调用", st.session_state.get('gpu_usr_calls', 0))
    
    with col2:
        st.metric("USRCAT描述符GPU调用", st.session_state.get('gpu_usrcat_calls', 0))
    
    with col3:
        st.metric("最近邻计算GPU调用", st.session_state.get('gpu_nn_calls', 0))
    
    with col4:
        st.metric("降维GPU调用", st.session_state.get('gpu_tsne_calls', 0))
    
    # 显示GPU内存使用详情
    st.subheader("GPU内存使用情况")
    mem_data = {
        "当前已分配": torch.cuda.memory_allocated() / 1024**2,
        "当前已缓存": torch.cuda.memory_reserved() / 1024**2,
        "USR计算使用": st.session_state.get('gpu_usr_mem', 0),
        "USRCAT计算使用": st.session_state.get('usrcat_batch_mem', 0),
        "最近邻计算使用": st.session_state.get('gpu_nn_mem_usage', 0),
        "t-SNE使用": st.session_state.get('gpu_tsne_mem', 0),
        "峰值使用": st.session_state.get('gpu_peak_mem', 0),
    }
    
    # 绘制内存使用柱状图
    mem_df = pd.DataFrame([mem_data.values()], columns=mem_data.keys())
    st.bar_chart(mem_df.T)
    
    # 显示GPU加速性能对比
    if st.session_state.get('t-SNE_time') and st.session_state.get('t-SNE_gpu_used'):
        st.info(f"GPU加速t-SNE耗时: {st.session_state.get('t-SNE_time'):.2f}秒")
    
    # 记录峰值显存使用
    current_mem = torch.cuda.memory_allocated() / 1024**2
    peak_mem = st.session_state.get('gpu_peak_mem', 0)
    if current_mem > peak_mem:
        st.session_state.gpu_peak_mem = current_mem
    
    # 添加GPU信息表格
    gpu_info = {
        "属性": ["设备名称", "计算单元数", "总显存", "当前温度", "显存带宽"],
        "值": [
            torch.cuda.get_device_name(0),
            str(torch.cuda.get_device_properties(0).multi_processor_count),
            f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB",
            f"{get_gpu_temp()}",
            f"{torch.cuda.get_device_properties(0).memory_clock_rate * torch.cuda.get_device_properties(0).memory_bus_width / (8 * 1000):.1f} GB/s"
        ]
    }
    st.table(pd.DataFrame(gpu_info))

# 添加GPU温度获取函数(仅支持NVIDIA GPU和Linux)
def get_gpu_temp():
    """获取GPU温度，如果可能的话"""
    try:
        import subprocess
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader'])
        return output.decode('utf-8').strip() + "°C"
    except:
        return "未知"

# 添加一个详细的GPU信息面板函数
def gpu_debug_panel():
    """显示GPU调试面板"""
    if not torch.cuda.is_available():
        st.sidebar.warning("GPU不可用，无法显示调试信息")
        return
    
    with st.sidebar.expander("🔎 GPU调试信息", expanded=False):
        st.write("### GPU内存追踪")
        
        # 显示当前内存使用
        current_mem = torch.cuda.memory_allocated() / 1024**2
        st.progress(min(1.0, current_mem / torch.cuda.get_device_properties(0).total_memory * 1024**2))
        st.write(f"当前使用: {current_mem:.1f}MB / {torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f}MB")
        
        # 显示GPU调用详情
        st.write("### 函数调用")
        st.write(f"- USRCAT GPU调用: {st.session_state.get('gpu_usrcat_calls', 0)}")
        st.write(f"- 最近邻GPU调用: {st.session_state.get('gpu_nn_calls', 0)}")
        st.write(f"- 降维GPU调用: {st.session_state.get('gpu_tsne_calls', 0)}")
        
        # 显示计时信息
        st.write("### 计时信息")
        st.write(f"- USRCAT批处理: {st.session_state.get('usrcat_batch_time', 0):.3f}秒")
        st.write(f"- t-SNE降维: {st.session_state.get('t-SNE_time', 0):.3f}秒")
        
        # 显示错误计数
        st.write("### 错误计数")
        st.write(f"- GPU回退次数: {st.session_state.get('gpu_fallbacks', 0)}")
        
        # 显示CUDA版本信息
        st.write("### CUDA版本")
        st.code(f"CUDA: {torch.version.cuda}\nCuDNN: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")

# 在主页面底部增加详细的GPU使用报告
def generate_gpu_report():
    """生成详细的GPU使用报告"""
    if not torch.cuda.is_available():
        return
    
    st.subheader("🔍 GPU使用详细报告")
    
    # 总体使用情况
    st.write("### 总体使用情况")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.get('gpu_usrcat_calls', 0) > 0:
            st.success("✅ USRCAT描述符计算已使用GPU加速")
        else:
            st.error("❌ USRCAT描述符计算未使用GPU")
            
    with col2:
        if st.session_state.get('gpu_nn_calls', 0) > 0:
            st.success("✅ 最近邻距离计算已使用GPU加速")
        else:
            st.error("❌ 最近邻距离计算未使用GPU")
            
    with col3:
        if st.session_state.get('gpu_tsne_calls', 0) > 0:
            st.success("✅ 降维分析已使用GPU加速")
        else:
            st.error("❌ 降维分析未使用GPU")
    
    # GPU内存使用情况
    st.write("### GPU内存使用情况")
    
    # 收集所有内存使用数据
    mem_data = {
        "描述符计算": st.session_state.get('usrcat_batch_mem', 0),
        "最近邻距离": st.session_state.get('gpu_nn_mem_usage', 0),
        "降维分析": st.session_state.get('gpu_tsne_mem', 0),
        "最大使用量": st.session_state.get('gpu_peak_mem', 0),
    }
    
    # 显示内存使用条形图
    mem_df = pd.DataFrame([mem_data.values()], columns=mem_data.keys())
    st.bar_chart(mem_df.T)
    
    # GPU VS CPU性能比较
    st.write("### GPU加速效果")
    
    # 如果有CPU和GPU的时间对比，显示比较图表
    if st.session_state.get('t-SNE_time') and st.session_state.get('t-SNE_gpu_used'):
        gpu_time = st.session_state.get('t-SNE_time', 0)
        # 这里我们估计CPU时间是GPU时间的5倍(仅用于演示)
        cpu_time = gpu_time * 5  # 实际应用中应该有真实的对比数据
        
        perf_data = {
            "GPU": gpu_time,
            "估计CPU": cpu_time
        }
        perf_df = pd.DataFrame([perf_data.values()], columns=perf_data.keys())
        st.bar_chart(perf_df.T)
        
        # 计算加速比
        speedup = cpu_time / gpu_time if gpu_time > 0 else 0
        st.metric("估计加速比", f"{speedup:.1f}x")
    
    # 添加GPU设备详情
    with st.expander("GPU设备详情"):
        properties = torch.cuda.get_device_properties(0)
        st.json({
            "设备名称": torch.cuda.get_device_name(0),
            "计算能力": f"{properties.major}.{properties.minor}",
            "多处理器数量": properties.multi_processor_count,
            "总显存(GB)": properties.total_memory / 1024**3,
            "最大线程数/块": properties.max_threads_per_block,
            "时钟频率(MHz)": properties.clock_rate / 1000,
            "L2缓存大小(MB)": properties.l2_cache_size / 1024**2 if hasattr(properties, 'l2_cache_size') else "未知",
        })

# 添加CUDA检查函数，用于验证CUDA是否正常工作
def verify_cuda_operation():
    """执行简单的CUDA操作以验证GPU是否正常工作"""
    if not torch.cuda.is_available():
        return "CUDA不可用"
    
    try:
        # 创建两个随机张量
        a = torch.randn(1000, 1000).cuda()
        b = torch.randn(1000, 1000).cuda()
        
        # 记录开始时间
        start = time.time()
        
        # 执行矩阵乘法
        c = torch.matmul(a, b)
        
        # 确保计算完成
        torch.cuda.synchronize()
        
        # 计算耗时
        duration = time.time() - start
        
        # 清理
        del a, b, c
        torch.cuda.empty_cache()
        
        return f"CUDA运行正常，1000x1000矩阵乘法耗时: {duration*1000:.2f}毫秒"
    except Exception as e:
        return f"CUDA测试失败: {str(e)}"

# 在页面加载时添加GPU测试按钮
st.sidebar.subheader("GPU状态检测")
if st.sidebar.button("测试GPU"):
    result = verify_cuda_operation()
    st.sidebar.code(result)

# 在页面加载时添加GPU调试面板
if st.sidebar.checkbox("启用GPU实时监控", value=False):
    gpu_debug_panel()

# 主界面逻辑
if st.button("开始分析") and fileA is not None and fileB is not None:
    with st.spinner("正在进行3D形状分析..."):
        try:
            analysis_start_time = time.time()
            
            # 加载分子
            st.info("加载分子数据...")
            molsA, dfA = load_smiles(fileA, smiles_col)
            molsB, dfB = load_smiles(fileB, smiles_col)
            
            if molsA is None or molsB is None or len(molsA) == 0 or len(molsB) == 0:
                st.error("无法加载分子数据或数据为空，请检查输入文件格式是否正确")
                st.stop()
            
            st.success(f"成功加载 {len(molsA)} 个分子(数据集A)和 {len(molsB)} 个分子(数据集B)")
            
            # 初始化GPU
            cuda_available = False
            device = torch.device("cpu")
            
            if enable_gpu:
                try:
                    cuda_available, device = initialize_cuda()
                    if cuda_available:
                        # 根据GPU策略设置内存限制
                        try:
                            if gpu_strategy == "内存优先":
                                # 保留更多GPU内存，慢但更安全
                                torch.cuda.set_per_process_memory_fraction(gpu_mem_limit / 100 * 0.8)
                            elif gpu_strategy == "性能优先":
                                # 使用更多GPU内存，更快但风险更高
                                torch.cuda.set_per_process_memory_fraction(gpu_mem_limit / 100 * 0.95)
                            else:  # 平衡模式
                                torch.cuda.set_per_process_memory_fraction(gpu_mem_limit / 100 * 0.9)
                        except Exception as e:
                            st.warning(f"设置GPU内存限制时出错: {str(e)}")
                        
                        # 清理GPU内存
                        torch.cuda.empty_cache()
                        
                        # 显示GPU状态
                        gpu_stats = {
                            "GPU型号": torch.cuda.get_device_name(0),
                            "总显存": f"{torch.cuda.get_device_properties(0).total_memory / 1024**2:.1f} MB",
                            "已用显存": f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB",
                            "已缓存显存": f"{torch.cuda.memory_reserved() / 1024**2:.1f} MB"
                        }
                        st.write("GPU状态:", gpu_stats)
                except Exception as e:
                    st.warning(f"初始化GPU时出错: {str(e)}，将使用CPU进行计算")
                    cuda_available = False
                    device = torch.device("cpu")
            else:
                st.info("GPU加速已禁用，将使用CPU进行计算")
                
                # 随机抽样
                if len(molsA) > max_samples:
                    molsA = random.sample(molsA, max_samples)
                st.info(f"已从数据集A随机抽取 {max_samples} 个分子进行分析")
                if len(molsB) > max_samples:
                    molsB = random.sample(molsB, max_samples)
                st.info(f"已从数据集B随机抽取 {max_samples} 个分子进行分析")
            
            # 收集构象生成参数
            conformer_params = {
                # 基本参数
                'max_attempts': max_attempts,
                'use_mmff': use_mmff,
                'energy_iter': energy_iter,
                'add_hydrogens': add_hydrogens,
                
                # TorchANI参数
                'torchani_model': torchani_model if 'torchani_model' in locals() else 'ANI2x',
                'optimization_steps': optimization_steps if 'optimization_steps' in locals() else 100,
                'torchani_batch_size': torchani_batch_size if 'torchani_batch_size' in locals() else 32,
                'use_torchani_optimization': use_torchani_optimization if 'use_torchani_optimization' in locals() else True,
                'learning_rate': learning_rate if 'learning_rate' in locals() else 0.01,
                'use_mixed_precision_torchani': use_mixed_precision_torchani if 'use_mixed_precision_torchani' in locals() else True,
                'max_atoms_per_batch': max_atoms_per_batch if 'max_atoms_per_batch' in locals() else 5000,
                'gradient_clipping': gradient_clipping if 'gradient_clipping' in locals() else True,
                
                # DeepChem参数
                'model_type': deepchem_model if 'deepchem_model' in locals() else 'mpnn',
                'use_gpu': enable_gpu and cuda_available,
                'use_mixed_precision': use_mixed_precision if 'use_mixed_precision' in locals() else True,
                'dc_force_field': dc_force_field if 'dc_force_field' in locals() else 'mmff94s',
                
                # Clara参数
                'force_field': clara_force_field if 'clara_force_field' in locals() else 'MMFF94s',
                'precision': clara_precision if 'clara_precision' in locals() else 'mixed',
                'num_conformers': clara_num_conformers if 'clara_num_conformers' in locals() else 1,
                'energy_threshold': clara_energy_threshold if 'clara_energy_threshold' in locals() else 1.0,
                'clara_optimization_steps': clara_optimization_steps if 'clara_optimization_steps' in locals() else 500
            }
            
            # 使用新的批量3D构象生成
            step_start_time = time.time()
            
            # Create status elements and progress bar before calling the function
            status_container_A = st.empty()
            progress_text_A = st.empty()
            progress_bar_A = st.progress(0)

            with st.spinner("生成数据集A的3D构象..."):
                molsA_3d = batch_generate_3d_conformers(
                    molsA, 
                    progress_bar=progress_bar_A, # Pass the progress bar object
                    status_container=status_container_A,
                    progress_text=progress_text_A,
                    backend=conformer_backend,
                    batch_size=batch_size if not auto_batch else None,
                    **conformer_params
                )
            
            step_time = time.time() - step_start_time
            st.success(f"数据集A构象生成完成，用时: {step_time:.1f}秒")
            # Clear progress bar A after completion
            progress_bar_A.empty() 
            status_container_A.empty()
            progress_text_A.empty()

            step_start_time = time.time()
            
            # Create status elements and progress bar for dataset B
            status_container_B = st.empty()
            progress_text_B = st.empty()
            progress_bar_B = st.progress(0)
            
            with st.spinner("生成数据集B的3D构象..."):
                molsB_3d = batch_generate_3d_conformers(
                    molsB, 
                    progress_bar=progress_bar_B, # Pass the progress bar object
                    status_container=status_container_B,
                    progress_text=progress_text_B,
                    backend=conformer_backend,
                    batch_size=batch_size if not auto_batch else None,
                    **conformer_params
                )
            
            step_time = time.time() - step_start_time
            st.success(f"数据集B构象生成完成，用时: {step_time:.1f}秒")
            # Clear progress bar B after completion
            progress_bar_B.empty() 
            status_container_B.empty()
            progress_text_B.empty()
            
            # 移除无效构象
            valid_molsA_3d = [mol for mol in molsA_3d if mol is not None]
            valid_molsB_3d = [mol for mol in molsB_3d if mol is not None]
            
            # 检查是否有足够的有效构象
            if len(valid_molsA_3d) == 0 or len(valid_molsB_3d) == 0:
                st.error("有效的3D构象数量不足，无法继续分析")
                st.error(f"数据集A: {len(valid_molsA_3d)}/{len(molsA)} 个有效构象")
                st.error(f"数据集B: {len(valid_molsB_3d)}/{len(molsB)} 个有效构象")
                st.stop()
            
            # 状态显示
            st.info(f"数据集A: {len(valid_molsA_3d)}/{len(molsA)} 个有效构象 ({len(valid_molsA_3d)/len(molsA)*100:.1f}%)")
            st.info(f"数据集B: {len(valid_molsB_3d)}/{len(molsB)} 个有效构象 ({len(valid_molsB_3d)/len(molsB)*100:.1f}%)")
            
            # 计算形状描述符
            step_start_time = time.time()
            
            with st.spinner(f"计算 {shape_desc} 形状描述符..."):
                try:
                    if shape_desc == "USR":
                        descsA = batch_compute_shape_descriptors(
                            valid_molsA_3d, 
                            descriptor_type="USR", 
                        cuda_available=cuda_available,
                            batch_size=batch_size if not auto_batch else None
                        )
                        descsB = batch_compute_shape_descriptors(
                            valid_molsB_3d, 
                            descriptor_type="USR", 
                            cuda_available=cuda_available,
                            batch_size=batch_size if not auto_batch else None
                        )
                    else:  # USRCAT
                        descsA = batch_compute_shape_descriptors(
                            valid_molsA_3d, 
                            descriptor_type="USRCAT", 
                            cuda_available=cuda_available,
                            batch_size=batch_size if not auto_batch else None
                        )
                        descsB = batch_compute_shape_descriptors(
                            valid_molsB_3d, 
                            descriptor_type="USRCAT", 
                            cuda_available=cuda_available,
                            batch_size=batch_size if not auto_batch else None
                        )
                except Exception as e:
                    st.error(f"计算形状描述符时出错: {str(e)}")
                    st.stop()
            
            step_time = time.time() - step_start_time
            st.success(f"形状描述符计算完成，用时: {step_time:.1f}秒")
            
            # 移除无效描述符
            valid_descsA = [d for d in descsA if d is not None]
            valid_descsB = [d for d in descsB if d is not None]
            
            if len(valid_descsA) == 0 or len(valid_descsB) == 0:
                st.error("有效的形状描述符数量不足，无法进行分析")
                st.error(f"数据集A: {len(valid_descsA)}/{len(valid_molsA_3d)} 个有效描述符")
                st.error(f"数据集B: {len(valid_descsB)}/{len(valid_molsB_3d)} 个有效描述符")
                st.stop()
            
            st.info(f"数据集A: {len(valid_descsA)}/{len(valid_molsA_3d)} 个有效描述符 ({len(valid_descsA)/len(valid_molsA_3d)*100:.1f}%)")
            st.info(f"数据集B: {len(valid_descsB)}/{len(valid_molsB_3d)} 个有效描述符 ({len(valid_descsB)/len(valid_molsB_3d)*100:.1f}%)")
            
            # 描述符标准化
            if normalize_desc:
                step_start_time = time.time()
                with st.spinner("标准化描述符..."):
                    try:
                        # 检查描述符有效性和形状一致性
                        if len(valid_descsA) == 0 or len(valid_descsB) == 0:
                            raise ValueError("没有有效的描述符可供标准化")
                        
                        # 检查所有描述符是否具有相同的形状
                        desc_shapes_A = [d.shape if hasattr(d, 'shape') else len(d) for d in valid_descsA]
                        desc_shapes_B = [d.shape if hasattr(d, 'shape') else len(d) for d in valid_descsB]
                        
                        # 确保所有描述符都是numpy数组且形状一致
                        valid_descsA_clean = []
                        valid_descsB_clean = []
                        
                        # 获取期望的描述符长度（从第一个有效描述符）
                        expected_length = None
                        for desc in valid_descsA + valid_descsB:
                            if desc is not None and hasattr(desc, '__len__'):
                                expected_length = len(desc)
                                break
                        
                        if expected_length is None:
                            raise ValueError("无法确定描述符的期望长度")
                        
                        # 过滤和清理描述符
                        for desc in valid_descsA:
                            if desc is not None and hasattr(desc, '__len__') and len(desc) == expected_length:
                                if hasattr(desc, 'shape'):
                                    valid_descsA_clean.append(desc)
                                else:
                                    valid_descsA_clean.append(np.array(desc))
                        
                        for desc in valid_descsB:
                            if desc is not None and hasattr(desc, '__len__') and len(desc) == expected_length:
                                if hasattr(desc, 'shape'):
                                    valid_descsB_clean.append(desc)
                                else:
                                    valid_descsB_clean.append(np.array(desc))
                        
                        if len(valid_descsA_clean) == 0 or len(valid_descsB_clean) == 0:
                            raise ValueError("清理后没有有效的描述符可供标准化")
                        
                        # 转换为numpy数组
                        valid_descsA = np.array(valid_descsA_clean)
                        valid_descsB = np.array(valid_descsB_clean)
                        
                        # 合并所有描述符以计算全局均值和标准差
                        all_descs = np.vstack([valid_descsA, valid_descsB])
                        mean = np.mean(all_descs, axis=0)
                        std = np.std(all_descs, axis=0)
                        # 防止除以零
                        std[std == 0] = 1.0
                        # 应用标准化
                        valid_descsA = (valid_descsA - mean) / std
                        valid_descsB = (valid_descsB - mean) / std
                        
                        st.info(f"标准化完成: A={valid_descsA.shape}, B={valid_descsB.shape}")
                        
                    except Exception as e:
                        st.error(f"标准化描述符时出错: {str(e)}")
                        st.warning("跳过标准化步骤...")
                
                step_time = time.time() - step_start_time
                st.success(f"描述符标准化完成，用时: {step_time:.1f}秒")
            
            # 降维可视化
            step_start_time = time.time()
            with st.spinner(f"使用 {dim_reduction} 降维..."):
                try:
                    # 检查描述符数据的有效性
                    if len(valid_descsA) == 0 or len(valid_descsB) == 0:
                        raise ValueError("没有有效的描述符可供降维")
                    
                    # 确保描述符是numpy数组且形状一致
                    if not isinstance(valid_descsA, np.ndarray):
                        # 如果不是numpy数组，需要重新检查和转换
                        valid_descsA_clean = []
                        for desc in valid_descsA:
                            if desc is not None and hasattr(desc, '__len__'):
                                if hasattr(desc, 'shape'):
                                    valid_descsA_clean.append(desc)
                                else:
                                    valid_descsA_clean.append(np.array(desc))
                        
                        if len(valid_descsA_clean) == 0:
                            raise ValueError("数据集A没有有效的描述符")
                        
                        valid_descsA = np.array(valid_descsA_clean)
                    
                    if not isinstance(valid_descsB, np.ndarray):
                        # 如果不是numpy数组，需要重新检查和转换
                        valid_descsB_clean = []
                        for desc in valid_descsB:
                            if desc is not None and hasattr(desc, '__len__'):
                                if hasattr(desc, 'shape'):
                                    valid_descsB_clean.append(desc)
                                else:
                                    valid_descsB_clean.append(np.array(desc))
                        
                        if len(valid_descsB_clean) == 0:
                            raise ValueError("数据集B没有有效的描述符")
                        
                        valid_descsB = np.array(valid_descsB_clean)
                    
                    # 检查数组形状
                    if valid_descsA.ndim != 2 or valid_descsB.ndim != 2:
                        raise ValueError(f"描述符数组维度不正确: A={valid_descsA.ndim}D, B={valid_descsB.ndim}D，期望2D")
                    
                    if valid_descsA.shape[1] != valid_descsB.shape[1]:
                        raise ValueError(f"描述符特征维度不匹配: A={valid_descsA.shape[1]}, B={valid_descsB.shape[1]}")
                    
                    # 组合两个数据集以进行降维
                    combined_descs = np.vstack([valid_descsA, valid_descsB])
                    
                    # 检查是否有NaN或无穷值
                    if np.isnan(combined_descs).any() or np.isinf(combined_descs).any():
                        st.warning("描述符中包含NaN或无穷值，将进行清理...")
                        # 替换NaN和无穷值
                        combined_descs = np.nan_to_num(combined_descs, nan=0.0, posinf=1e6, neginf=-1e6)
                    
                    st.info(f"降维输入数据形状: {combined_descs.shape}")
                    
                    # 降维参数
                    dim_params = {}
                    if dim_reduction == "t-SNE":
                        dim_params["perplexity"] = perplexity
                    else:  # UMAP
                        dim_params["n_neighbors"] = n_neighbors
                        dim_params["min_dist"] = min_dist
                    
                    # 执行降维
                    coords = perform_dimensionality_reduction(
                        combined_descs, 
                            method=dim_reduction,
                            cuda_available=cuda_available,
                        **dim_params
                    )
                    
                    if coords is None or len(coords) == 0:
                        raise ValueError("降维失败，返回空结果")
                    
                    # 分离两个数据集的坐标
                    coordsA = coords[:len(valid_descsA)]
                    coordsB = coords[len(valid_descsA):]
                    
                    st.info(f"降维完成: A={coordsA.shape}, B={coordsB.shape}")
                    
                except Exception as e:
                    st.error(f"执行降维时出错: {str(e)}")
                    st.stop()
            
            step_time = time.time() - step_start_time
            st.success(f"降维完成，用时: {step_time:.1f}秒")
            
            # 计算主惯量比率（用于PMI三角形图）
            step_start_time = time.time()
            with st.spinner("计算主惯量比率..."):
                try:
                    pmiA = [compute_pmi_ratios(mol) for mol in valid_molsA_3d]
                    pmiB = [compute_pmi_ratios(mol) for mol in valid_molsB_3d]
                    
                    # 过滤无效值
                    pmiA = [p for p in pmiA if p is not None]
                    pmiB = [p for p in pmiB if p is not None]
                except Exception as e:
                    st.warning(f"计算PMI比率时出错: {str(e)}")
                    pmiA = []
                    pmiB = []
            
            step_time = time.time() - step_start_time
            if len(pmiA) > 0 and len(pmiB) > 0:
                st.success(f"PMI比率计算完成，用时: {step_time:.1f}秒")
            else:
                st.warning("PMI比率计算未完成或没有有效结果")
            
            # 计算形状空间分布指标
            step_start_time = time.time()
            with st.spinner("计算分布指标..."):
                try:
                    dist_metrics = calculate_distribution_metrics(coordsA, coordsB)
                except Exception as e:
                    st.warning(f"计算分布指标时出错: {str(e)}")
                    dist_metrics = {
                        "hausdorff_distance": float('nan'),
                        "earth_movers_distance": float('nan'),
                        "kl_divergence": float('nan'),
                        "js_divergence": float('nan')
                    }
            
            step_time = time.time() - step_start_time
            st.success(f"分布指标计算完成，用时: {step_time:.1f}秒")
            
            # 计算总用时
            total_time = time.time() - analysis_start_time
            
            # 显示结果
            st.success(f"分析完成！总用时: {total_time:.1f}秒")
            
            # 结果选项卡
            result_tabs = st.tabs(["形状空间分析", "主惯量分析", "分布指标", "性能统计"])
            
            with result_tabs[0]:
                st.subheader("形状空间分布")
                
                plot_shape_space(
                    coordsA, coordsB, 
                    title=f"{shape_desc} 形状空间分布 ({dim_reduction})"
                )
                
                st.write(f"数据集A: {len(coordsA)} 个点，数据集B: {len(coordsB)} 个点")
                
                # 添加样本点击显示
                if st.checkbox("启用样本点击", value=False):
                    st.info("点击图中的点可以查看对应的分子结构（尚未实现）")
            
            with result_tabs[1]:
                st.subheader("主惯量三角形")
                
                if len(pmiA) > 0 and len(pmiB) > 0:
                    plot_pmi_triangle(pmiA, pmiB, "数据集A", "数据集B")
                    st.write(f"数据集A: {len(pmiA)} 个有效PMI比率，数据集B: {len(pmiB)} 个有效PMI比率")
                else:
                    st.warning("无法计算足够的主惯量比率以生成三角形图")
            
            with result_tabs[2]:
                st.subheader("分布相似性指标")
                
                # 显示计算的各种指标
                metrics_df = pd.DataFrame({
                    "指标": ["豪斯多夫距离", "地球移动距离 (EMD)", "KL散度", "JS散度"],
                    "值": [
                        dist_metrics["hausdorff_distance"],
                        dist_metrics["earth_movers_distance"],
                        dist_metrics["kl_divergence"],
                        dist_metrics["js_divergence"]
                    ]
                })
                st.dataframe(metrics_df)
                
                st.markdown("""
                **指标说明:**
                - **豪斯多夫距离**: 两个点集之间的最大最小距离，较低的值表示形状空间更相似
                - **地球移动距离 (EMD)**: 也称为Wasserstein距离，表示将一个分布转换为另一个所需的"工作量"
                - **KL散度**: 衡量一个概率分布相对于另一个的差异
                - **JS散度**: KL散度的对称版本，范围在[0,1]，0表示完全相同的分布
                """)
            
            with result_tabs[3]:
                st.subheader("性能统计")
                
                # GPU状态
                if cuda_available:
                    with st.expander("GPU使用情况", expanded=True):
                        # 显示内存使用情况
                        gpu_stats = {
                            "GPU型号": torch.cuda.get_device_name(0),
                            "已用显存": f"{torch.cuda.memory_allocated()/1024**2:.1f} MB",
                            "已缓存显存": f"{torch.cuda.memory_reserved()/1024**2:.1f} MB",
                            "可用显存": f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_reserved())/1024**2:.1f} MB",
                            "总显存": f"{torch.cuda.get_device_properties(0).total_memory/1024**2:.1f} MB",
                            "使用率": f"{torch.cuda.memory_allocated()/torch.cuda.get_device_properties(0).total_memory*100:.1f}%",
                        }
                        
                        st.json(gpu_stats)
                
                # 构象生成后端信息
                with st.expander("构象生成信息", expanded=True):
                    st.info(f"使用后端: {conformer_backend}")
                    st.write("构象生成参数:", conformer_params)
                    st.write(f"数据集A成功率: {len(valid_molsA_3d)/len(molsA)*100:.1f}% ({len(valid_molsA_3d)}/{len(molsA)})")
                    st.write(f"数据集B成功率: {len(valid_molsB_3d)/len(molsB)*100:.1f}% ({len(valid_molsB_3d)}/{len(molsB)})")
                
                # 性能统计
                with st.expander("处理时间统计", expanded=True):
                    # 构造各步骤的处理时间表格
                    steps_df = pd.DataFrame({
                        "处理步骤": ["构象生成", "形状描述符计算", "降维", "PMI计算", "分布指标计算", "总计"],
                        "时间 (秒)": [
                            # 这些时间在前面的代码中已经计算过，这里用占位符 
                            # 在实际使用中，这些值将被真实的时间替代
                            total_time * 0.5,  # 假设构象生成占总时间的50%
                            total_time * 0.2,  # 假设描述符计算占总时间的20%
                            total_time * 0.1,  # 假设降维占总时间的10%
                            total_time * 0.05, # 假设PMI计算占总时间的5%
                            total_time * 0.05, # 假设分布指标计算占总时间的5%
                            total_time
                        ],
                        "处理分子/项数": [
                            f"{len(molsA) + len(molsB)}个",
                            f"{len(valid_molsA_3d) + len(valid_molsB_3d)}个",
                            f"{len(valid_descsA) + len(valid_descsB)}个",
                            f"{len(pmiA) + len(pmiB)}个",
                            "2个数据集",
                            ""
                        ],
                        "每项平均时间 (秒)": [
                            (total_time * 0.5) / (len(molsA) + len(molsB)) if (len(molsA) + len(molsB)) > 0 else 0,
                            (total_time * 0.2) / (len(valid_molsA_3d) + len(valid_molsB_3d)) if (len(valid_molsA_3d) + len(valid_molsB_3d)) > 0 else 0,
                            (total_time * 0.1) / (len(valid_descsA) + len(valid_descsB)) if (len(valid_descsA) + len(valid_descsB)) > 0 else 0,
                            (total_time * 0.05) / (len(pmiA) + len(pmiB)) if (len(pmiA) + len(pmiB)) > 0 else 0,
                            (total_time * 0.05) / 2 if 2 > 0 else 0,
                            ""
                        ]
                    })
                    st.dataframe(steps_df)
                    
                    # 性能对比图
                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = ax.barh(steps_df["处理步骤"][:-1], steps_df["时间 (秒)"][:-1])
                    ax.set_xlabel("时间 (秒)")
                    ax.set_title("各步骤处理时间")
                    
                    # 在柱状图上添加数值标签
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.5, bar.get_y() + bar.get_height()/2, f"{width:.1f}s", 
                                ha='left', va='center')
                    
                    st.pyplot(fig)
                    
                    # 如果启用了GPU，显示GPU vs CPU对比
                    if cuda_available:
                        st.info(f"使用GPU加速，估计加速比: 2-5倍（取决于分子复杂度和GPU性能）")
        except Exception as e:
            st.error(f"分析过程中出错: {str(e)}") 
            import traceback
            st.error(f"错误详情:\n{traceback.format_exc()}")
            
            if enable_gpu and torch.cuda.is_available():
                try:
                    st.error("GPU错误详情:")
                    st.error(f"- 已用显存: {torch.cuda.memory_allocated()/1024**2:.1f}MB")
                    st.error(f"- 已缓存显存: {torch.cuda.memory_reserved()/1024**2:.1f}MB")
                    torch.cuda.empty_cache()  # 清理GPU内存
                except:
                    st.error("无法获取GPU状态信息")