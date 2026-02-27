"""JAX可用性检查脚本（可选依赖）"""

from __future__ import annotations


def main() -> int:
    try:
        import jax
        import jaxlib
    except ImportError as exc:
        print(f"JAX is not installed: {exc}")
        print("Tip: JAX is optional for this project, skip this check if you do not use it.")
        return 0

    print(f"JAX version: {jax.__version__}")
    print(f"jaxlib version: {jaxlib.__version__}")

    print("\nJAX available devices:")
    for i, device in enumerate(jax.devices()):
        print(f"  Device {i}: {device.device_kind} ({device.platform}) - ID: {device.id}")

    print(f"\nJAX default backend: {jax.default_backend()}")

    gpu_devices = [
        d for d in jax.devices()
        if "gpu" in d.platform.lower() or "cuda" in d.device_kind.lower()
    ]
    if gpu_devices:
        print(f"\nSUCCESS: JAX found the following GPU devices: {gpu_devices}")
    else:
        print("\nWARNING: JAX did NOT find any GPU devices. It will likely use CPU.")

    # 可选：检查 TensorFlow（如果怀疑与 JAX 共享 CUDA 有问题）
    try:
        import tensorflow as tf

        print(f"\nTensorFlow version: {tf.__version__}")
        gpu_devices_tf = tf.config.list_physical_devices("GPU")
        if gpu_devices_tf:
            print(f"SUCCESS: TensorFlow found the following GPU devices: {gpu_devices_tf}")
            for device in gpu_devices_tf:
                details = tf.config.experimental.get_device_details(device)
                print(
                    "  Details: "
                    f"{details.get('device_name', 'Unknown GPU')} "
                    f"({details.get('compute_capability', 'N/A')})"
                )
        else:
            print("WARNING: TensorFlow did NOT find any GPU devices.")
    except ImportError:
        print("\nTensorFlow is not installed.")
    except Exception as exc:
        print(f"\nError during TensorFlow GPU check: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
