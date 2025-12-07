from safetensors import safe_open
import sys

model_path = "/Users/Suhas/CS259_Project/hmt_model/model.safetensors"

try:
    with safe_open(model_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            print(f"{key} {f.get_tensor(key).shape}")
except Exception as e:
    print(f"Error reading safetensors: {e}")
