import jax
import jaxlib

print(f"JAX version: {jax.__version__}")
print(f"jaxlib version: {jaxlib.__version__}")

print("\nJAX available devices:")
for i, device in enumerate(jax.devices()):
    print(f"  Device {i}: {device.device_kind} ({device.platform}) - ID: {device.id}")

print(f"\nJAX default backend: {jax.default_backend()}")

gpu_devices = [d for d in jax.devices() if 'gpu' in d.platform.lower() or 'cuda' in d.device_kind.lower()]
if gpu_devices:
    print(f"\nSUCCESS: JAX found the following GPU devices: {gpu_devices}")
else:
    print("\nWARNING: JAX did NOT find any GPU devices. It will likely use CPU.")

# 新增：检查 TensorFlow (如果之前与 JAX 有CUDA共享问题)
try:
    import tensorflow as tf
    print(f"\nTensorFlow version: {tf.__version__}")
    gpu_devices_tf = tf.config.list_physical_devices('GPU')
    if gpu_devices_tf:
        print(f"SUCCESS: TensorFlow found the following GPU devices: {gpu_devices_tf}")
        for device in gpu_devices_tf:
            details = tf.config.experimental.get_device_details(device)
            print(f"  Details: {details.get('device_name', 'Unknown GPU')} ({details.get('compute_capability', 'N/A')})")
    else:
        print("WARNING: TensorFlow did NOT find any GPU devices.")
except ImportError:
    print("\nTensorFlow is not installed.")
except Exception as e:
    print(f"\nError during TensorFlow GPU check: {e}")
