import json
import os
import shutil

# Define paths
project_dir = "/Users/Suhas/CS259_Project"
safetensors_file = os.path.join(project_dir, "llama3.2-1b-pg19-final-hmt.safetensors")
model_dir = os.path.join(project_dir, "hmt_model")

# Create model directory
os.makedirs(model_dir, exist_ok=True)

# Move safetensors file
if os.path.exists(safetensors_file):
    shutil.move(safetensors_file, os.path.join(model_dir, "model.safetensors"))
    print(f"Moved {safetensors_file} to {model_dir}")
else:
    print(f"Warning: {safetensors_file} not found. Please ensure it is in {model_dir}")

# Create config.json
config = {
    "architectures": ["HMTModel"],
    "model_type": "llama",
    "vocab_size": 128256,
    "hidden_size": 2048,
    "intermediate_size": 8192,
    "num_hidden_layers": 16,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "hidden_act": "silu",
    "max_position_embeddings": 131072,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-05,
    "use_cache": True,
    "rope_theta": 500000.0,
    "attention_bias": False,
    "tie_word_embeddings": True
}

with open(os.path.join(model_dir, "config.json"), "w") as f:
    json.dump(config, f, indent=2)
print(f"Created config.json in {model_dir}")

print("\nIMPORTANT: You still need to add tokenizer files (tokenizer.json, tokenizer_config.json, special_tokens_map.json) to the 'hmt_model' directory. You can copy these from a standard Llama 3.2 1B model.")
