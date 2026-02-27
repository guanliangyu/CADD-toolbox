#!/usr/bin/env python3
import sys

sys.path.append("/home/liangyu/CADD-toolbox")
from rdkit import Chem
from rdkit.Chem import AllChem
from multiprocessing import Pool


def optimize_single_molecule(args):
    mol_data, orig_idx, steps = args
    mol, mol_props = mol_data

    try:
        # 恢复分子属性
        for prop_name, prop_value in mol_props.items():
            mol.SetProp(prop_name, prop_value)

        # 优化
        mol_copy = Chem.Mol(mol)
        mp = AllChem.MMFFGetMoleculeProperties(mol_copy)
        if mp is None:
            return {
                "orig_idx": orig_idx,
                "mol": None,
                "mol_props": None,
                "success": False,
                "message": "MMFF94力场初始化失败",
            }

        ff = AllChem.MMFFGetMoleculeForceField(mol_copy, mp, confId=0)
        if ff is None:
            return {
                "orig_idx": orig_idx,
                "mol": None,
                "mol_props": None,
                "success": False,
                "message": "MMFF94力场创建失败",
            }

        converged = ff.Minimize(maxIts=steps)

        # 关键：这里的逻辑判断
        result = {
            "orig_idx": orig_idx,
            "mol": mol_copy,  # 优化成功的分子
            "mol_props": mol_props,  # 属性信息
            "success": True,  # 优化成功
            "message": f"优化完成（收敛代码: {converged}）",
        }

        print(
            f"分子 {orig_idx}: success={result['success']}, mol存在={result['mol'] is not None}, props存在={result['mol_props'] is not None}"
        )
        return result

    except Exception as e:
        return {
            "orig_idx": orig_idx,
            "mol": None,
            "mol_props": None,
            "success": False,
            "message": f"异常: {str(e)}",
        }


# 测试
input_file = "data/20250605_LC_Stock_HTS_Compounds_Full/conformers_LC_Stock_HTS_Compounds_background_conformer_20250609_200112.sdf"
supplier = Chem.ForwardSDMolSupplier(input_file, removeHs=False, sanitize=True)
mol = next(supplier)
mol_clean = Chem.AddHs(mol)
mol_props = {prop: mol_clean.GetProp(prop) for prop in mol_clean.GetPropNames()}

process_args = [((mol_clean, mol_props), 0, 100)]

with Pool(processes=1) as pool:
    for result_data in pool.imap_unordered(optimize_single_molecule, process_args):
        print(f"返回结果: {result_data}")

        # 模拟后台脚本的判断
        condition = (
            result_data["success"] and result_data["mol"] and result_data["mol_props"]
        )
        print(f"后台脚本判断: {condition}")

        if condition:
            print("✅ 会被计入成功")
        else:
            print("❌ 会被标记为失败")
