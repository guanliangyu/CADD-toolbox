#!/usr/bin/env python3
"""
结果验证工具 - 检查构象优化结果的完整性
帮助用户了解中断任务的实际完成情况
"""

import os
import argparse
from datetime import datetime


def validate_sdf_file(sdf_file):
    """验证SDF文件的完整性和内容"""

    if not os.path.exists(sdf_file):
        return {"status": "error", "message": f"文件不存在: {sdf_file}"}

    try:
        from rdkit import Chem
    except ImportError:
        return {"status": "error", "message": "RDKit未安装，无法验证分子结构"}

    try:
        file_size = os.path.getsize(sdf_file)
        supplier = Chem.ForwardSDMolSupplier(sdf_file, removeHs=False, sanitize=True)

        mol_count = 0
        valid_mol_count = 0
        conformer_count = 0
        empty_count = 0

        mol_names = set()

        print(f"🔍 正在验证文件: {sdf_file}")
        print(f"📏 文件大小: {file_size / (1024*1024):.1f} MB")

        for i, mol in enumerate(supplier):
            mol_count += 1

            if mol is not None:
                valid_mol_count += 1
                conformer_count += mol.GetNumConformers()

                # 检查分子名称
                if mol.HasProp("_Name"):
                    mol_names.add(mol.GetProp("_Name"))

                # 每10000个分子显示一次进度
                if mol_count % 10000 == 0:
                    print(f"⏳ 已验证: {mol_count:,} 分子 (有效: {valid_mol_count:,})")
            else:
                empty_count += 1

        result = {
            "status": "success",
            "file_size_mb": file_size / (1024 * 1024),
            "total_entries": mol_count,
            "valid_molecules": valid_mol_count,
            "empty_entries": empty_count,
            "total_conformers": conformer_count,
            "unique_names": len(mol_names),
            "avg_conformers_per_mol": (
                conformer_count / valid_mol_count if valid_mol_count > 0 else 0
            ),
        }

        return result

    except Exception as e:
        return {"status": "error", "message": f"验证过程出错: {str(e)}"}


def check_log_completion(log_file):
    """检查日志文件，了解任务完成情况"""

    if not os.path.exists(log_file):
        return {"status": "no_log", "message": "日志文件不存在"}

    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        total_lines = len(lines)
        last_progress = None
        completion_found = False
        error_found = False

        # 查找最后的进度信息
        for line in reversed(lines[-100:]):  # 只查看最后100行
            if "已完成:" in line and "/" in line:
                parts = line.split("已完成:")[1].strip().split()
                if parts and "/" in parts[0]:
                    last_progress = parts[0]
                    break
            elif "全部完成" in line or "构象生成完成" in line:
                completion_found = True
                break
            elif "ERROR" in line or "失败" in line:
                error_found = True
                break

        result = {
            "status": "analyzed",
            "total_lines": total_lines,
            "last_progress": last_progress,
            "completion_found": completion_found,
            "error_found": error_found,
        }

        # 解析进度
        if last_progress and "/" in last_progress:
            try:
                completed, total = last_progress.split("/")
                completed_num = int(completed)
                total_num = int(total)
                progress_percent = (completed_num / total_num) * 100
                result["completed_count"] = completed_num
                result["total_count"] = total_num
                result["progress_percent"] = progress_percent
            except Exception:
                pass

        return result

    except Exception as e:
        return {"status": "error", "message": f"读取日志失败: {str(e)}"}


def analyze_work_directory(work_dir):
    """分析工作目录，找出所有相关文件"""

    if not os.path.exists(work_dir):
        return {"status": "error", "message": f"工作目录不存在: {work_dir}"}

    files = os.listdir(work_dir)

    sdf_files = [f for f in files if f.endswith(".sdf")]
    log_files = [f for f in files if f.endswith(".log")]
    py_files = [f for f in files if f.endswith(".py")]
    done_files = [f for f in files if f.endswith(".done") or f.endswith(".completed")]
    error_files = [f for f in files if f.endswith(".error")]

    return {
        "status": "success",
        "total_files": len(files),
        "sdf_files": sdf_files,
        "log_files": log_files,
        "py_files": py_files,
        "done_files": done_files,
        "error_files": error_files,
    }


def print_analysis_report(work_dir):
    """生成详细的分析报告"""

    print(f"\n{'='*60}")
    print("📊 工作目录分析报告")
    print(f"{'='*60}")
    print(f"📁 目录: {work_dir}")
    print(f"🕐 分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 分析目录结构
    dir_analysis = analyze_work_directory(work_dir)

    if dir_analysis["status"] == "error":
        print(f"❌ {dir_analysis['message']}")
        return

    print("📋 文件统计:")
    print(f"  📄 总文件数: {dir_analysis['total_files']}")
    print(f"  🧪 SDF文件: {len(dir_analysis['sdf_files'])}")
    print(f"  📝 日志文件: {len(dir_analysis['log_files'])}")
    print(f"  🐍 Python脚本: {len(dir_analysis['py_files'])}")
    print(f"  ✅ 完成标志: {len(dir_analysis['done_files'])}")
    print(f"  ❌ 错误标志: {len(dir_analysis['error_files'])}")
    print()

    # 分析SDF文件
    if dir_analysis["sdf_files"]:
        print("🧪 SDF文件分析:")
        for sdf_file in dir_analysis["sdf_files"]:
            full_path = os.path.join(work_dir, sdf_file)
            print(f"\n  📄 {sdf_file}:")

            validation = validate_sdf_file(full_path)
            if validation["status"] == "success":
                print("    ✅ 验证成功")
                print(f"    📏 文件大小: {validation['file_size_mb']:.1f} MB")
                print(f"    🔢 总条目: {validation['total_entries']:,}")
                print(f"    ✅ 有效分子: {validation['valid_molecules']:,}")
                print(f"    ❌ 空条目: {validation['empty_entries']:,}")
                print(f"    🧬 总构象数: {validation['total_conformers']:,}")
                print(
                    f"    📊 平均构象/分子: {validation['avg_conformers_per_mol']:.1f}"
                )

                if validation["valid_molecules"] == validation["total_entries"]:
                    print("    🎉 所有分子都有效！")
                elif validation["valid_molecules"] > 0:
                    success_rate = (
                        validation["valid_molecules"] / validation["total_entries"]
                    ) * 100
                    print(f"    📈 成功率: {success_rate:.1f}%")
            else:
                print(f"    ❌ 验证失败: {validation['message']}")

    # 分析日志文件
    if dir_analysis["log_files"]:
        print("\n📝 日志文件分析:")
        for log_file in dir_analysis["log_files"]:
            full_path = os.path.join(work_dir, log_file)
            print(f"\n  📄 {log_file}:")

            log_analysis = check_log_completion(full_path)
            if log_analysis["status"] == "analyzed":
                print(f"    📏 日志行数: {log_analysis['total_lines']:,}")

                if log_analysis["completion_found"]:
                    print("    ✅ 发现完成标志")
                elif log_analysis["error_found"]:
                    print("    ❌ 发现错误标志")
                elif log_analysis["last_progress"]:
                    print(f"    ⏳ 最后进度: {log_analysis['last_progress']}")
                    if "progress_percent" in log_analysis:
                        print(f"    📊 完成度: {log_analysis['progress_percent']:.1f}%")
                else:
                    print("    ❓ 无明确进度信息")
            else:
                print(f"    ❌ 分析失败: {log_analysis['message']}")

    # 给出建议
    print("\n💡 建议:")

    # 检查是否有未完成的优化
    has_input_sdf = any("generated_conformers" in f for f in dir_analysis["sdf_files"])
    has_output_sdf = any("optimized" in f for f in dir_analysis["sdf_files"])
    has_errors = len(dir_analysis["error_files"]) > 0
    has_completion = len(dir_analysis["done_files"]) > 0

    if has_completion:
        print("  ✅ 任务已正常完成")
    elif has_errors:
        print("  ❌ 任务执行时出现错误，检查错误文件")
    elif has_input_sdf and not has_output_sdf:
        print("  🔄 发现输入文件但无输出文件，可能需要重新运行优化")
        print("  💡 使用新的智能后台优化功能重新启动任务")
    elif has_input_sdf and has_output_sdf:
        print("  ✅ 发现输入和输出文件，任务可能已完成")
        print("  🔍 建议验证输出文件的完整性")
    else:
        print("  ❓ 目录状态不明确，建议检查文件内容")

    print(f"\n{'='*60}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="构象优化结果验证工具")
    parser.add_argument("work_dir", help="工作目录路径")
    parser.add_argument("--sdf", help="特定SDF文件路径（可选）")
    parser.add_argument("--log", help="特定日志文件路径（可选）")

    args = parser.parse_args()

    if args.sdf:
        # 验证特定SDF文件
        print(f"🔍 验证SDF文件: {args.sdf}")
        result = validate_sdf_file(args.sdf)
        print(f"结果: {result}")
    elif args.log:
        # 分析特定日志文件
        print(f"📝 分析日志文件: {args.log}")
        result = check_log_completion(args.log)
        print(f"结果: {result}")
    else:
        # 全面分析工作目录
        print_analysis_report(args.work_dir)


if __name__ == "__main__":
    main()
