"""
分子处理工具模块 - 提供处理分子结构、计算特征等功能
"""
import os
import numpy as np
from typing import List, Dict, Tuple, Union, Optional, Any
import logging

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.AllChem import MMFFOptimizeMolecule, UFFOptimizeMolecule
from rdkit import DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator

logger = logging.getLogger(__name__)

class MoleculeProcessor:
    """分子处理类，提供分子标准化、3D构象生成、指纹计算等功能"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化分子处理器
        
        参数:
            config: 配置字典，包含处理参数
        """
        self.config = config
        self.filtering_config = config.get('data', {}).get('filtering', {})
        self.conformer_config = config.get('data', {}).get('conformers', {})
        self.charge_config = config.get('data', {}).get('charges', {})
    
    def prepare_molecule(self, smiles: str) -> Optional[Chem.Mol]:
        """
        从SMILES创建分子并进行初步处理
        
        参数:
            smiles: 分子的SMILES字符串
            
        返回:
            处理后的RDKit分子对象，如果处理失败则返回None
        """
        try:
            # 从SMILES创建分子
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # 分子清洗和标准化
            if self.filtering_config.get('standardize', True):
                mol = self._standardize_mol(mol)
                
            # 基本过滤
            if not self._passes_filters(mol):
                return None
                
            return mol
        except Exception as e:
            print(f"处理分子时出错: {e}")
            return None
    
    def _standardize_mol(self, mol: Chem.Mol) -> Chem.Mol:
        """标准化分子"""
        try:
            # 移除同位素标记
            for atom in mol.GetAtoms():
                atom.SetIsotope(0)
            
            # 如果配置允许，尝试中和分子电荷
            if self.filtering_config.get('neutralize', True):
                mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
            
            return mol
        except:
            return mol  # 返回原始分子，如果标准化失败
    
    def _passes_filters(self, mol: Chem.Mol) -> bool:
        """检查分子是否通过基本过滤条件"""
        if not mol:
            return False
            
        # 分子量过滤
        mw = Descriptors.MolWt(mol)
        min_mw = self.filtering_config.get('min_mw', 0)
        max_mw = self.filtering_config.get('max_mw', float('inf'))
        
        if mw < min_mw or mw > max_mw:
            return False
            
        # 原子数过滤
        num_atoms = mol.GetNumAtoms()
        max_atoms = self.filtering_config.get('max_atoms', float('inf'))
        
        if num_atoms > max_atoms:
            return False
            
        return True
    
    def generate_3d_conformer(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        为分子生成3D构象
        
        参数:
            mol: RDKit分子对象
            
        返回:
            带有3D构象的分子对象，如果生成失败则返回None
        """
        if mol is None or not self.conformer_config.get('enabled', True):
            return mol
            
        # 添加氢原子
        mol = Chem.AddHs(mol)
        
        # 设置构象生成参数
        method = self.conformer_config.get('method', 'ETKDG')
        max_iters = self.conformer_config.get('max_iters', 200)
        
        try:
            # 生成构象
            if method == 'ETKDG':
                ps = AllChem.ETKDGv3()
                ps.maxAttempts = 100
                cid = AllChem.EmbedMolecule(mol, ps)
            else:
                cid = AllChem.EmbedMolecule(mol)
                
            if cid < 0:
                return None
                
            # 力场优化
            force_field = self.conformer_config.get('force_field', 'MMFF94')
            if force_field == 'MMFF94':
                MMFFOptimizeMolecule(mol, maxIters=max_iters)
            elif force_field == 'UFF':
                UFFOptimizeMolecule(mol, maxIters=max_iters)
                
            # 移除氢原子（可选）
            if not self.conformer_config.get('keep_hydrogens', False):
                mol = Chem.RemoveHs(mol)
                
            return mol
        except Exception as e:
            print(f"生成3D构象时出错: {e}")
            return None
    
    def compute_gasteiger_charges(self, mol: Chem.Mol) -> Optional[List[float]]:
        """
        计算Gasteiger原子电荷
        
        参数:
            mol: RDKit分子对象
            
        返回:
            原子电荷列表，如果计算失败则返回None
        """
        if mol is None or not self.charge_config.get('enabled', True):
            return None
            
        try:
            # 计算Gasteiger电荷
            Chem.AllChem.ComputeGasteigerCharges(mol)
            charges = [float(a.GetProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0 
                    for a in mol.GetAtoms()]
            
            return charges
        except Exception as e:
            print(f"计算Gasteiger电荷时出错: {e}")
            return None
    
    @staticmethod
    def compute_fingerprint(mol: Chem.Mol, radius: int = 2, nBits: int = 1024, 
                          fp_type: str = 'morgan', use_features: bool = False) -> Optional[Union[DataStructs.ExplicitBitVect, 
                                                                    DataStructs.UIntSparseIntVect]]:
        """
        计算分子指纹
        
        参数:
            mol: RDKit分子对象
            radius: Morgan指纹半径
            nBits: 指纹位数
            fp_type: 指纹类型 ('morgan', 'rdkit', 'maccs', 'fcfp')
            use_features: 是否使用原子特征（用于FCFP）
            
        返回:
            RDKit指纹对象
        """
        if mol is None:
            return None
            
        try:
            if fp_type == 'morgan':
                # 使用新的MorganGenerator API
                morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits, useFeatures=use_features)
                return morgan_gen.GetFingerprint(mol)
            elif fp_type == 'fcfp':
                # 使用新的MorganGenerator API，设置useFeatures=True
                morgan_gen = GetMorganGenerator(radius=radius, fpSize=nBits, useFeatures=True)
                return morgan_gen.GetFingerprint(mol)
            elif fp_type == 'rdkit':
                # 使用新的RDKitFPGenerator API
                rdk_gen = GetRDKitFPGenerator(fpSize=nBits)
                return rdk_gen.GetFingerprint(mol)
            elif fp_type == 'maccs':
                return rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            else:
                logger.warning(f"不支持的指纹类型: {fp_type}")
                return None
        except Exception as e:
            logger.error(f"计算指纹时出错: {str(e)}")
            return None
    
    @staticmethod
    def compute_basic_descriptors(mol: Chem.Mol) -> Dict[str, float]:
        """
        计算基本分子描述符
        
        参数:
            mol: RDKit分子对象
            
        返回:
            描述符字典
        """
        if mol is None:
            return {}
            
        try:
            descriptors = {
                'mw': Descriptors.MolWt(mol),
                'logp': Descriptors.MolLogP(mol),
                'tpsa': Descriptors.TPSA(mol),
                'hba': rdMolDescriptors.CalcNumHBA(mol),
                'hbd': rdMolDescriptors.CalcNumHBD(mol),
                'rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                'heavy_atoms': mol.GetNumHeavyAtoms()
            }
            return descriptors
        except Exception as e:
            print(f"计算描述符时出错: {e}")
            return {}
    
    @staticmethod
    def compute_shape_descriptors(mol: Chem.Mol) -> Dict[str, float]:
        """
        计算分子形状描述符
        注意：分子必须有3D构象
        
        参数:
            mol: 带有3D构象的RDKit分子对象
            
        返回:
            形状描述符字典
        """
        if mol is None or mol.GetNumConformers() == 0:
            return {}
            
        try:
            # 计算惯性主轴
            conf = mol.GetConformer()
            principal_moments = list(rdMolDescriptors.CalcPBF(mol))  # 转换为列表
            
            if len(principal_moments) != 3:
                return {}
                
            # 计算体积与表面积
            shape_desc = {
                'principal_moment_1': float(principal_moments[0]),
                'principal_moment_2': float(principal_moments[1]),
                'principal_moment_3': float(principal_moments[2])
            }
            
            # 只在所有主轴都大于0时计算比率
            if all(m > 0 for m in principal_moments):
                shape_desc.update({
                    'sphericity': float(principal_moments[0] / principal_moments[2]),  # 球形度
                    'asphericity': float((principal_moments[2] - (principal_moments[0] + principal_moments[1])/2) / principal_moments[2])  # 非球度
                })
            
            return shape_desc
        except Exception as e:
            print(f"计算形状描述符时出错: {str(e)}")
            return {}
    
    @staticmethod
    def fp_to_numpy(fp: Union[DataStructs.ExplicitBitVect, DataStructs.UIntSparseIntVect, 
                             DataStructs.IntSparseIntVect]) -> np.ndarray:
        """
        将RDKit指纹转换为NumPy数组
        
        参数:
            fp: RDKit指纹对象
            
        返回:
            NumPy数组
        """
        if fp is None:
            return np.array([])
            
        if isinstance(fp, DataStructs.ExplicitBitVect):
            return np.array(fp)
            
        # 对于其他类型的指纹，转换为数组
        arr = np.zeros((0,), dtype=np.int32)
        DataStructs.ConvertToNumpyArray(fp, arr)
        return arr 