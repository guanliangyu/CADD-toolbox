import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
import os
import logging
import sys
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sdf_to_csv(sdf_file, csv_file):
    """
    将SDF文件转换为CSV文件，包含SMILES、Name(从IDNUMBER获取)和MolWeight列
    
    参数:
    sdf_file: SDF文件路径
    csv_file: 输出CSV文件路径
    """
    logging.info(f"正在读取SDF文件: {sdf_file}")
    
    # 读取SDF文件
    molecules = []
    sdf_supplier = Chem.SDMolSupplier(sdf_file)
    total_mols = 0
    processed_mols = 0
    
    for mol in sdf_supplier:
        total_mols += 1
        if mol is not None:
            try:
                # 获取IDNUMBER作为名称
                name = mol.GetProp("IDNUMBER") if mol.HasProp("IDNUMBER") else ""
                name = name.strip()  # 移除空白字符
                
                # 生成SMILES
                smiles = Chem.MolToSmiles(mol)
                
                # 计算分子量（保留2位小数）
                mol_weight = round(Descriptors.ExactMolWt(mol), 2)
                
                molecules.append({
                    "SMILES": smiles,
                    "Name": name,
                    "MolWeight": mol_weight
                })
                processed_mols += 1
                
                # 每处理1000个分子显示一次进度
                if processed_mols % 1000 == 0:
                    logging.info(f"已处理 {processed_mols} 个分子...")
                    
            except Exception as e:
                logging.warning(f"处理分子时出错: {str(e)}")
                continue
    
    # 创建DataFrame
    df = pd.DataFrame(molecules)
    
    # 保存为CSV
    logging.info(f"正在保存CSV文件: {csv_file}")
    df.to_csv(csv_file, index=False)
    
    # 输出统计信息
    logging.info(f"转换完成!")
    logging.info(f"总分子数: {total_mols}")
    logging.info(f"成功处理: {processed_mols}")
    logging.info(f"失败数量: {total_mols - processed_mols}")
    
    # 显示前几行数据作为预览
    logging.info("\n数据预览:")
    print(df.head())
    
    # 检查空Name的情况
    empty_names = df[df['Name'].str.strip() == ''].shape[0]
    if empty_names > 0:
        logging.warning(f"警告: 有 {empty_names} 个分子没有IDNUMBER")

def main():
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("使用方法: python convert_sdf_to_csv.py <sdf文件路径>")
        print("示例: python convert_sdf_to_csv.py example.sdf")
        sys.exit(1)
    
    # 获取输入文件路径
    sdf_file = sys.argv[1]
    
    # 检查输入文件是否存在
    if not os.path.exists(sdf_file):
        logging.error(f"输入文件不存在: {sdf_file}")
        sys.exit(1)
    
    # 生成输出文件路径（将.sdf替换为.csv）
    csv_file = str(Path(sdf_file).with_suffix('.csv'))
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(csv_file) or '.', exist_ok=True)
    
    # 执行转换
    try:
        sdf_to_csv(sdf_file, csv_file)
    except Exception as e:
        logging.error(f"转换过程中出错: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 