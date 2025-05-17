"""
文件工具模块 - 用于页面间通过本地文件交换数据
"""
import os
import json
import pickle
import pandas as pd
import tempfile
from pathlib import Path

# 默认数据目录
DATA_DIR = Path(tempfile.gettempdir()) / "molecular_subset_data"

def ensure_data_dir():
    """确保数据目录存在"""
    os.makedirs(DATA_DIR, exist_ok=True)
    return DATA_DIR

def get_file_path(filename):
    """获取文件完整路径"""
    ensure_data_dir()
    return DATA_DIR / filename

def save_dataframe(df, filename="molecules.csv"):
    """保存DataFrame到文件
    
    参数:
        df: pandas DataFrame对象
        filename: 文件名
    """
    ensure_data_dir()
    file_path = get_file_path(filename)
    df.to_csv(file_path, index=False)
    return str(file_path)

def load_dataframe(filename="molecules.csv"):
    """从文件加载DataFrame
    
    参数:
        filename: 文件名
        
    返回:
        pandas DataFrame或None（如果文件不存在）
    """
    file_path = get_file_path(filename)
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

def save_json(data, filename="config.json"):
    """保存数据为JSON文件
    
    参数:
        data: 要保存的数据
        filename: 文件名
    """
    ensure_data_dir()
    file_path = get_file_path(filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(file_path)

def load_json(filename="config.json"):
    """从JSON文件加载数据
    
    参数:
        filename: 文件名
        
    返回:
        加载的数据或None（如果文件不存在）
    """
    file_path = get_file_path(filename)
    if file_path.exists():
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def save_pickle(data, filename="results.pkl"):
    """将数据保存为pickle文件
    
    参数:
        data: 要保存的数据
        filename: 文件名
    """
    ensure_data_dir()
    file_path = get_file_path(filename)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    return str(file_path)

def load_pickle(filename="results.pkl"):
    """从pickle文件加载数据
    
    参数:
        filename: 文件名
        
    返回:
        加载的数据或None（如果文件不存在）
    """
    file_path = get_file_path(filename)
    if file_path.exists():
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

def file_exists(filename):
    """检查文件是否存在
    
    参数:
        filename: 文件名
        
    返回:
        布尔值
    """
    file_path = get_file_path(filename)
    return file_path.exists()

def delete_file(filename):
    """删除文件
    
    参数:
        filename: 文件名
        
    返回:
        布尔值，表示是否成功删除
    """
    file_path = get_file_path(filename)
    if file_path.exists():
        os.remove(file_path)
        return True
    return False

def clear_all_data():
    """清除所有数据文件"""
    ensure_data_dir()
    for file in os.listdir(DATA_DIR):
        file_path = DATA_DIR / file
        if file_path.is_file():
            os.remove(file_path)
    return True

# 文件名常量
MOLECULES_FILE = "molecules.csv"
CONFIG_FILE = "config.json"
PROCESSED_RESULTS_FILE = "processed_results.pkl"
CLUSTERING_RESULTS_FILE = "clustering_results.pkl"
SELECTED_SUBSET_FILE = "selected_subset.csv"
SUBSET_INDICES_FILE = "subset_indices.json"
VALIDATION_RESULTS_FILE = "validation_results.pkl" 