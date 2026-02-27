"""2D分子描述符与指纹生成工具模块"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
try:
    from rdkit.Chem.Avalon import pyAvalonTools  # type: ignore
    AVALON_AVAILABLE = True
except ImportError:
    pyAvalonTools = None  # type: ignore
    AVALON_AVAILABLE = False
from rdkit.Chem.AtomPairs import Pairs, Torsions
from rdkit.Chem.QED import qed

logger = logging.getLogger(__name__)

__all__ = [
    "AVALON_AVAILABLE",
    "build_molecule_cache",
    "generate_molecular_descriptors",
    "generate_fingerprints",
    "merge_feature_frames",
    "process_chunk_worker",
]


def build_molecule_cache(smiles_series: pd.Series) -> Tuple[List[Optional[Chem.Mol]], pd.Series]:
    """预解析SMILES以缓存对应的RDKit分子对象"""
    mols: List[Optional[Chem.Mol]] = []
    valid_flags = np.zeros(len(smiles_series), dtype=bool)

    for idx, smiles in enumerate(smiles_series):
        if pd.isna(smiles):
            mols.append(None)
            continue

        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            valid_flags[idx] = True
        mols.append(mol)

    return mols, pd.Series(valid_flags, index=smiles_series.index)


def _descriptor_functions() -> Dict[str, Callable[[Chem.Mol], float]]:
    """返回2D分子描述符函数集合"""
    return {
        "MolWt": Descriptors.MolWt,
        "LogP": Descriptors.MolLogP,
        "NumHDonors": Descriptors.NumHDonors,
        "NumHAcceptors": Descriptors.NumHAcceptors,
        "TPSA": Descriptors.TPSA,
        "NumRotatableBonds": Descriptors.NumRotatableBonds,
        "NumAromaticRings": Descriptors.NumAromaticRings,
        "NumAliphaticRings": Descriptors.NumAliphaticRings,
        "NumHeavyAtoms": Descriptors.HeavyAtomCount,
        "NumHeteroatoms": Descriptors.NumHeteroatoms,
        "FractionCsp3": lambda mol: rdMolDescriptors.CalcFractionCSP3(mol),
        "Chi0v": Descriptors.Chi0v,
        "Chi1v": Descriptors.Chi1v,
        "Chi2v": Descriptors.Chi2v,
        "Chi3v": Descriptors.Chi3v,
        "Chi4v": Descriptors.Chi4v,
        "Kappa1": Descriptors.Kappa1,
        "Kappa2": Descriptors.Kappa2,
        "Kappa3": Descriptors.Kappa3,
        "BertzCT": Descriptors.BertzCT,
        "SlogP_VSA1": Descriptors.SlogP_VSA1,
        "SlogP_VSA2": Descriptors.SlogP_VSA2,
        "SMR_VSA1": Descriptors.SMR_VSA1,
        "SMR_VSA2": Descriptors.SMR_VSA2,
        "LabuteASA": Descriptors.LabuteASA,
        "PEOE_VSA1": Descriptors.PEOE_VSA1,
        "PEOE_VSA2": Descriptors.PEOE_VSA2,
        "QED": qed,
        "MaxAbsPartialCharge": Descriptors.MaxAbsPartialCharge,
        "MinAbsPartialCharge": Descriptors.MinAbsPartialCharge,
        "MaxPartialCharge": Descriptors.MaxPartialCharge,
        "MinPartialCharge": Descriptors.MinPartialCharge,
        "MolMR": Descriptors.MolMR,
        "BalabanJ": Descriptors.BalabanJ,
        "HallKierAlpha": Descriptors.HallKierAlpha,
        "NumSaturatedCarbocycles": Descriptors.NumSaturatedCarbocycles,
        "NumSaturatedHeterocycles": Descriptors.NumSaturatedHeterocycles,
        "NumAliphaticCarbocycles": Descriptors.NumAliphaticCarbocycles,
        "NumAliphaticHeterocycles": Descriptors.NumAliphaticHeterocycles,
        "NumAromaticCarbocycles": Descriptors.NumAromaticCarbocycles,
        "NumAromaticHeterocycles": Descriptors.NumAromaticHeterocycles,
        "RingCount": Descriptors.RingCount,
        "FpDensityMorgan1": Descriptors.FpDensityMorgan1,
        "FpDensityMorgan2": Descriptors.FpDensityMorgan2,
        "BCUT2D_MWHI": Descriptors.BCUT2D_MWHI,
        "BCUT2D_MWLOW": Descriptors.BCUT2D_MWLOW,
        "BCUT2D_CHGHI": Descriptors.BCUT2D_CHGHI,
        "BCUT2D_CHGLO": Descriptors.BCUT2D_CHGLO,
        "BCUT2D_LOGPHI": Descriptors.BCUT2D_LOGPHI,
        "BCUT2D_LOGPLOW": Descriptors.BCUT2D_LOGPLOW,
        "BCUT2D_MRHI": Descriptors.BCUT2D_MRHI,
        "BCUT2D_MRLOW": Descriptors.BCUT2D_MRLOW,
    }


def generate_molecular_descriptors(
    mol_cache: Sequence[Optional[Chem.Mol]],
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """根据分子缓存生成批量分子描述符"""
    descriptor_functions = _descriptor_functions()
    descriptor_names = list(descriptor_functions.keys())
    total = len(mol_cache)

    if total == 0:
        return pd.DataFrame(columns=descriptor_names)

    descriptors: List[Dict[str, Any]] = []

    for idx, mol in enumerate(mol_cache):
        row: Dict[str, Any] = {}

        if mol is None:
            for desc_name in descriptor_names:
                row[desc_name] = np.nan
        else:
            for desc_name, desc_func in descriptor_functions.items():
                try:
                    row[desc_name] = desc_func(mol)
                except Exception:  # pylint: disable=broad-except
                    row[desc_name] = np.nan

        descriptors.append(row)

        if progress_callback and ((idx + 1) % 100 == 0 or idx + 1 == total):
            progress_callback(idx + 1, total)

    return pd.DataFrame(descriptors, columns=descriptor_names)


def _bitvect_to_array(bitvect: DataStructs.ExplicitBitVect, size: int) -> np.ndarray:
    arr = np.zeros((size,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bitvect, arr)
    return arr


def generate_fingerprints(
    mol_cache: Sequence[Optional[Chem.Mol]],
    fp_type: str = "morgan",
    radius: int = 2,
    nbits: int = 1024,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> pd.DataFrame:
    """生成指定类型的分子指纹"""
    fingerprints: List[np.ndarray] = []
    total = len(mol_cache)

    if fp_type == "maccs":
        bit_length = 167
    else:
        bit_length = nbits

    if total == 0:
        if fp_type == "maccs":
            columns = [f"MACCS_{i}" for i in range(bit_length)]
        else:
            columns = [f"{fp_type.upper()}_FP_{i}" for i in range(bit_length)]
        return pd.DataFrame(columns=columns, dtype=np.uint8)

    for idx, mol in enumerate(mol_cache):
        if mol is None:
            fp_array = np.zeros((bit_length,), dtype=np.uint8)
        else:
            try:
                if fp_type == "morgan":
                    generator = GetMorganGenerator(radius=radius, fpSize=nbits)
                    fp = generator.GetFingerprint(mol)
                elif fp_type == "rdkit":
                    fp = Chem.RDKFingerprint(mol, fpSize=nbits)
                elif fp_type == "maccs":
                    fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
                elif fp_type == "avalon":
                    if not AVALON_AVAILABLE or pyAvalonTools is None:
                        raise ImportError(
                            "Avalon指纹生成需要RDKit编译Avalon模块，请使用支持Avalon的RDKit或关闭该指纹选项。"
                        )
                    fp = pyAvalonTools.GetAvalonFP(mol, bit_length)
                elif fp_type == "atompairs":
                    fp = Pairs.GetAtomPairFingerprintAsBitVect(mol, nBits=bit_length)
                elif fp_type == "torsions":
                    fp = Torsions.GetTopologicalTorsionFingerprintAsBitVect(mol, nBits=bit_length)
                else:
                    generator = GetMorganGenerator(radius=radius, fpSize=nbits)
                    fp = generator.GetFingerprint(mol)

                fp_array = _bitvect_to_array(fp, bit_length)
            except Exception:  # pylint: disable=broad-except
                fp_array = np.zeros((bit_length,), dtype=np.uint8)

        fingerprints.append(fp_array)

        if progress_callback and ((idx + 1) % 100 == 0 or idx + 1 == total):
            progress_callback(idx + 1, total)

    if fp_type == "maccs":
        columns = [f"MACCS_{i}" for i in range(bit_length)]
    else:
        columns = [f"{fp_type.upper()}_FP_{i}" for i in range(bit_length)]

    return pd.DataFrame(fingerprints, columns=columns, dtype=np.uint8)


def merge_feature_frames(
    base_df: pd.DataFrame,
    feature_frames: Sequence[pd.DataFrame],
    suffix: str = "_generated",
) -> pd.DataFrame:
    """将新特征DataFrame附加到基础数据框，并处理重名列"""
    combined = base_df.copy()
    existing = set(combined.columns)

    for frame in feature_frames:
        if frame is None or frame.empty:
            continue

        frame = frame.copy()
        rename_map: Dict[str, str] = {}

        for col in frame.columns:
            target = col

            if target in existing:
                target = f"{col}{suffix}"
                index = 1
                while target in existing:
                    index += 1
                    target = f"{col}{suffix}{index}"
                rename_map[col] = target

            existing.add(rename_map.get(col, col))

        if rename_map:
            frame = frame.rename(columns=rename_map)

        combined = pd.concat([combined, frame], axis=1)

    return combined


def _resolve_smiles_column_name(columns: Sequence[str], preferred: str | None = None) -> str | None:
    """根据优先列名与常见别名解析SMILES列名"""
    column_list = [str(col) for col in columns]
    if preferred and preferred in column_list:
        return preferred

    if "SMILES" in column_list:
        return "SMILES"

    normalized_map: Dict[str, str] = {col.strip().lower(): col for col in column_list}
    preferred_aliases = (
        "smiles",
        "canonical_smiles",
        "canon_smiles",
        "isomeric_smiles",
        "input_smiles",
        "structure_smiles",
        "molecule_smiles",
    )
    for alias in preferred_aliases:
        if alias in normalized_map:
            return normalized_map[alias]

    for col in column_list:
        if "smiles" in col.lower():
            return col

    return None


def process_chunk_worker(
    args: Tuple[Any, ...]
) -> pd.DataFrame:
    """子进程处理器：根据配置批量生成描述符并返回结果"""
    if len(args) >= 5:
        chunk_records, descriptor_types, fingerprint_types, fp_config, smiles_column = args[:5]
    else:
        chunk_records, descriptor_types, fingerprint_types, fp_config = args
        smiles_column = None

    chunk_df = pd.DataFrame(chunk_records)

    if chunk_df.empty:
        return chunk_df

    row_ids = chunk_df["__row_id"].to_numpy()
    chunk_df = chunk_df.drop(columns="__row_id").reset_index(drop=True)

    smiles_col_name = _resolve_smiles_column_name(chunk_df.columns.tolist(), preferred=smiles_column)
    if smiles_col_name is None:
        available_cols = ", ".join(map(str, chunk_df.columns[:20]))
        raise KeyError(f"未找到SMILES列，当前可用列: {available_cols}")

    smiles_series = chunk_df[smiles_col_name]
    mol_cache, _ = build_molecule_cache(smiles_series)

    feature_frames: List[pd.DataFrame] = []

    if "molecular_descriptors" in descriptor_types:
        feature_frames.append(generate_molecular_descriptors(mol_cache))

    for fp_type in fingerprint_types:
        if fp_type == "morgan":
            config = fp_config.get("morgan", {"radius": 2, "nbits": 1024})
            fp_df = generate_fingerprints(
                mol_cache,
                fp_type,
                radius=int(config.get("radius", 2)),
                nbits=int(config.get("nbits", 1024)),
            )
        elif fp_type == "maccs":
            fp_df = generate_fingerprints(mol_cache, fp_type)
        else:
            config = fp_config.get(fp_type, {"nbits": 1024})
            fp_df = generate_fingerprints(
                mol_cache,
                fp_type,
                nbits=int(config.get("nbits", 1024)),
            )

        feature_frames.append(fp_df)

    combined = merge_feature_frames(chunk_df, feature_frames)

    combined["__row_id"] = row_ids
    return combined
