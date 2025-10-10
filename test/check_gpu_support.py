#!/usr/bin/env python3
"""GPU支持情况检测脚本"""
from __future__ import annotations

import textwrap
from dataclasses import dataclass


@dataclass
class ModuleStatus:
    name: str
    available: bool
    detail: str = ""


def check_torch() -> ModuleStatus:
    try:
        import torch  # type: ignore

        info = [f"version={torch.__version__}"]
        cuda_ok = torch.cuda.is_available()
        info.append(f"cuda_available={cuda_ok}")
        if cuda_ok:
            try:
                dev_name = torch.cuda.get_device_name(0)
                info.append(f"device={dev_name}")
            except Exception as exc:  # pragma: no cover
                info.append(f"device_query_failed: {exc}")
        else:
            info.append("reason=CUDA unavailable")
        return ModuleStatus("torch", True, ", ".join(info))
    except Exception as exc:  # pragma: no cover
        return ModuleStatus("torch", False, f"error={exc}")


def check_faiss() -> ModuleStatus:
    try:
        import faiss  # type: ignore

        info_parts = []
        version = getattr(faiss, "__version__", "unknown")
        info_parts.append(f"version={version}")
        try:
            num_gpu = faiss.get_num_gpus()
            info_parts.append(f"gpu_count={num_gpu}")
        except Exception as exc:  # pragma: no cover
            info_parts.append(f"gpu_query_failed={exc}")
        return ModuleStatus("faiss", True, ", ".join(info_parts))
    except Exception as exc:  # pragma: no cover
        return ModuleStatus("faiss", False, f"error={exc}")


def check_cuml() -> ModuleStatus:
    try:
        import cuml  # type: ignore

        return ModuleStatus("cuml", True, f"version={cuml.__version__}")
    except Exception as exc:  # pragma: no cover
        return ModuleStatus("cuml", False, f"error={exc}")


def check_cupy() -> ModuleStatus:
    try:
        import cupy  # type: ignore

        return ModuleStatus("cupy", True, f"version={cupy.__version__}")
    except Exception as exc:  # pragma: no cover
        return ModuleStatus("cupy", False, f"error={exc}")


def main() -> None:
    checks = [check_torch(), check_faiss(), check_cuml(), check_cupy()]
    print("GPU Support Report")
    print("=" * 60)
    for status in checks:
        availability = "✅" if status.available else "❌"
        print(f"{availability} {status.name}: {status.detail}")

    failed = [c for c in checks if not c.available]
    if failed:
        print("\nDetected issues:")
        for status in failed:
            print(f"- {status.name}: {status.detail}")
    else:
        print("\nAll GPU-dependent modules are available.")


if __name__ == "__main__":
    main()
