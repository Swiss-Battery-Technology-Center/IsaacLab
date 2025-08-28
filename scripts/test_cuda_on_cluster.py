import torch

print("="*60)
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / 1024**3:.1f} GB")

    try:
        x = torch.rand((2, 2), device="cuda")
        y = x * 2
        print("Tensor test succeeded on device:", x.device)
        print(y)
    except Exception as e:
        print("Tensor test failed:", e)
else:
    print("No CUDA device found.")

print("="*60)
