import torch
from safetensors.torch import save_file
import numpy as np
import os
import json

def create_dummy_safetensors(file_path="dummy_model.safetensors", sharded=False):
    if sharded:
        output_dir = file_path # file_path is now treated as a directory
        os.makedirs(output_dir, exist_ok=True)

        tensors1 = {
            "model.embed_tokens.weight": torch.from_numpy(np.random.rand(10, 5).astype(np.float32)),
            "model.layers.0.self_attn.q_proj.weight": torch.from_numpy(np.random.rand(5, 5).astype(np.float32)),
        }
        shard1_path = os.path.join(output_dir, "model-00001-of-00002.safetensors")
        save_file(tensors1, shard1_path)

        tensors2 = {
            "model.layers.0.mlp.down_proj.weight": torch.from_numpy(np.random.rand(5, 10).astype(np.float32)),
        }
        shard2_path = os.path.join(output_dir, "model-00002-of-00002.safetensors")
        save_file(tensors2, shard2_path)

        # Create index file
        index_data = {
            "metadata": {"total_size": sum(t.numel() * t.element_size() for t in tensors1.values()) + sum(t.numel() * t.element_size() for t in tensors2.values())},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.mlp.down_proj.weight": "model-00002-of-00002.safetensors",
            }
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), 'w') as f:
            json.dump(index_data, f, indent=4)

        print(f"Dummy sharded safetensors model created at {output_dir}")

    else:
        tensors = {
            "model.embed_tokens.weight": torch.from_numpy(np.random.rand(10, 5).astype(np.float32)),
            "model.layers.0.self_attn.q_proj.weight": torch.from_numpy(np.random.rand(5, 5).astype(np.float32)),
            "model.layers.0.mlp.down_proj.weight": torch.from_numpy(np.random.rand(5, 10).astype(np.float32)),
        }
        save_file(tensors, file_path)
        print(f"Dummy single safetensors file created at {file_path}")

if __name__ == "__main__":
    # Create a single file dummy model
    create_dummy_safetensors("E:\\ggml-tool\\safetensor_to_gguf_converter\\dummy_model.safetensors", sharded=False)
    # Create a sharded dummy model
    create_dummy_safetensors("E:\\ggml-tool\\safetensor_to_gguf_converter\\dummy_sharded_model", sharded=True)
