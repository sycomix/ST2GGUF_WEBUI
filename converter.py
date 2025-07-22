import argparse
import struct
import numpy as np
from safetensors import safe_open
import torch
import os
import json

# GGUF Magic and Version
GGUF_MAGIC = 0x46554747  # GGUF in little-endian
GGUF_VERSION = 2

# GGUF DTYPE mapping
GGUF_DTYPE_MAP = {
    np.float32: 0,  # F32
    np.float16: 1,  # F16
    np.int8: 2,     # I8
    np.uint8: 3,    # U8
    np.int16: 4,    # I16
    np.uint16: 5,   # U16
    np.int32: 6,    # I32
    np.uint32: 7,   # U32
    np.bool_: 8,    # BOOL
    # Custom GGUF quantization types
    "Q4_0": 9, # Placeholder for Q4_0, actual value might differ in spec
    "Q8_0": 10, # Placeholder for Q8_0
}

def _write_string(f, s):
    s_bytes = s.encode('utf-8')
    f.write(struct.pack('<Q', len(s_bytes)))  # Length of string
    f.write(s_bytes)

def _write_int(f, i):
    f.write(struct.pack('<Q', i)) # 64-bit unsigned integer

def _write_float(f, fl):
    f.write(struct.pack('<f', fl)) # 32-bit float

def _write_tensor_info(f, name, n_dims, shape, dtype_id, offset):
    _write_string(f, name)
    f.write(struct.pack('<I', n_dims)) # Number of dimensions
    for dim in shape:
        f.write(struct.pack('<Q', dim)) # Shape dimensions
    f.write(struct.pack('<I', dtype_id)) # Dtype ID
    f.write(struct.pack('<Q', offset)) # Offset

def _align_offset(offset, alignment):
    return (offset + alignment - 1) // alignment * alignment

def load_safetensors(input_path: str):
    """Loads safetensors files, handling sharded models, and returns their tensors."""
    tensors = {}
    
    if os.path.isdir(input_path):
        # Look for index file in the directory
        index_file_path = os.path.join(input_path, "model.safetensors.index.json")
        if os.path.exists(index_file_path):
            with open(index_file_path, 'r') as f:
                index_data = json.load(f)
            
            # Load tensors from sharded files
            for tensor_name, tensor_info in index_data["weight_map"].items():
                shard_file = os.path.join(input_path, tensor_info)
                with safe_open(shard_file, framework="pt") as f_shard:
                    tensor = f_shard.get_tensor(tensor_name)
                    if tensor.dtype == torch.bfloat16:
                        tensor = tensor.to(torch.float32)
                    tensors[tensor_name] = tensor.numpy()
        else:
            # Load all .safetensors files in the directory as a single model
            for filename in os.listdir(input_path):
                if filename.endswith(".safetensors"):
                    file_path = os.path.join(input_path, filename)
                    with safe_open(file_path, framework="pt") as f:
                        for key in f.keys():
                            tensor = f.get_tensor(key)
                            if tensor.dtype == torch.bfloat16:
                                tensor = tensor.to(torch.float32)
                            tensors[key] = tensor.numpy()
    elif os.path.isfile(input_path) and input_path.endswith(".safetensors"):
        # Load a single .safetensors file
        with safe_open(input_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                tensors[key] = tensor.numpy()
    else:
        raise ValueError("Input path must be a .safetensors file or a directory containing safetensors files.")

    return tensors

def quantize_q4_0(tensor: np.ndarray):
    """Quantizes a float32 numpy array to Q4_0 format.
    This is a simplified implementation for demonstration.
    Q4_0: 32 weights per block, 1 float scale per block.
    """
    if tensor.dtype != np.float32:
        raise ValueError(f"Q4_0 quantization only supports float32 input, got {tensor.dtype}")

    block_size = 32
    num_blocks = (tensor.size + block_size - 1) // block_size
    
    # Reshape to process in blocks
    # Pad if necessary to make it a multiple of block_size
    padded_size = num_blocks * block_size
    padded_tensor = np.pad(tensor.flatten(), (0, padded_size - tensor.size), 'constant').reshape(num_blocks, block_size)

    # Calculate scales and quantized values
    scales = np.zeros(num_blocks, dtype=np.float32)
    q_values = np.zeros((num_blocks, block_size), dtype=np.int8)

    for i in range(num_blocks):
        block = padded_tensor[i]
        # Find the maximum absolute value in the block
        amax = np.max(np.abs(block))
        
        # Calculate scale (avoid division by zero)
        scale = amax / 7.0 if amax > 0 else 0.0 # 4-bit signed int range is -8 to 7
        scales[i] = scale
        
        # Quantize values
        if scale > 0:
            q_values[i] = np.round(block / scale).astype(np.int8)
        else:
            q_values[i] = np.zeros(block_size, dtype=np.int8)

    # Flatten quantized values and scales for storage
    # The GGUF format for Q4_0 stores scales first, then quantized values
    # For simplicity here, we'll return them separately.
    return q_values.tobytes(), scales.tobytes(), len(q_values.tobytes()) + len(scales.tobytes())

def quantize_q8_0(tensor: np.ndarray):
    """Quantizes a float32 numpy array to Q8_0 format.
    This is a simplified placeholder implementation.
    Q8_0: 32 weights per block, 1 float scale per block.
    """
    if tensor.dtype != np.float32:
        raise ValueError(f"Q8_0 quantization only supports float32 input, got {tensor.dtype}")

    # For Q8_0, it's typically just scaling to int8 range and storing the scale
    # This is a very basic representation, actual Q8_0 might involve more complex block structures
    amax = np.max(np.abs(tensor))
    scale = amax / 127.0 if amax > 0 else 0.0 # 8-bit signed int range is -128 to 127

    if scale > 0:
        q_values = np.round(tensor / scale).astype(np.int8)
    else:
        q_values = np.zeros_like(tensor, dtype=np.int8)

    return q_values.tobytes(), np.array([scale], dtype=np.float32).tobytes(), len(q_values.tobytes()) + len(np.array([scale], dtype=np.float32).tobytes())

def convert_to_gguf(tensors: dict, output_path: str, quantization_type: str = "none"):
    if quantization_type != "none":
        print(f"Quantization type {quantization_type} is not yet fully implemented beyond basic type mapping.")
        print("Tensors will be written unquantized for now.")

    with open(output_path, 'wb') as f:
        # --- Pass 1: Collect tensor info and calculate their sizes ---
        # This pass is to determine the total size of the tensor info block
        # and thus the starting offset for the actual tensor data.
        temp_tensor_infos = []
        for name, tensor in tensors.items():
            dtype_np = tensor.dtype.type
            
            if quantization_type == "Q4_0" and dtype_np == np.float32:
                q_data, scales_data, _ = quantize_q4_0(tensor)
                temp_tensor_infos.append({
                    "name": name + ".qdata",
                    "n_dims": len(tensor.shape),
                    "shape": tensor.shape,
                    "dtype_id": GGUF_DTYPE_MAP["Q4_0"],
                    "data_len": len(q_data),
                    "tensor_data": q_data # Store data temporarily
                })
                temp_tensor_infos.append({
                    "name": name + ".scales",
                    "n_dims": len(tensor.shape), 
                    "shape": tensor.shape, 
                    "dtype_id": GGUF_DTYPE_MAP[np.float32],
                    "data_len": len(scales_data),
                    "tensor_data": scales_data # Store data temporarily
                })
            elif quantization_type == "Q8_0" and dtype_np == np.float32:
                q_data, scales_data, _ = quantize_q8_0(tensor)
                temp_tensor_infos.append({
                    "name": name + ".qdata",
                    "n_dims": len(tensor.shape),
                    "shape": tensor.shape,
                    "dtype_id": GGUF_DTYPE_MAP[np.int8],
                    "data_len": len(q_data),
                    "tensor_data": q_data # Store data temporarily
                })
                temp_tensor_infos.append({
                    "name": name + ".scales",
                    "n_dims": 1, 
                    "shape": (1,), 
                    "dtype_id": GGUF_DTYPE_MAP[np.float32],
                    "data_len": len(scales_data),
                    "tensor_data": scales_data # Store data temporarily
                })
            else:
                if dtype_np not in GGUF_DTYPE_MAP:
                    raise ValueError(f"Unsupported dtype: {dtype_np} for tensor {name}")
                dtype_id = GGUF_DTYPE_MAP[dtype_np]
                temp_tensor_infos.append({
                    "name": name,
                    "n_dims": len(tensor.shape),
                    "shape": tensor.shape,
                    "dtype_id": dtype_id,
                    "data_len": tensor.nbytes,
                    "tensor_data": tensor.tobytes() # Store data temporarily
                })

        # Calculate the size of the header and tensor info block
        header_size = 4 + 4 + 8 + 8 # Magic, Version, n_tensors, n_kv_pairs
        tensor_info_block_size = 0
        for info in temp_tensor_infos:
            tensor_info_block_size += 8 # string length (Q)
            tensor_info_block_size += len(info["name"].encode('utf-8')) # string itself
            tensor_info_block_size += 4 # n_dims (I)
            tensor_info_block_size += 8 * len(info["shape"]) # shape dimensions (Q * n_dims)
            tensor_info_block_size += 4 # dtype_id (I)
            tensor_info_block_size += 8 # offset (Q)

        # Calculate the absolute starting offset for tensor data
        data_start_offset = _align_offset(header_size + tensor_info_block_size, 32)

        # --- Pass 2: Write GGUF Header, Tensor Info, and Tensor Data ---
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        _write_int(f, len(temp_tensor_infos))  # n_tensors
        _write_int(f, 0)  # n_kv_pairs (no metadata yet)

        # Write Tensor Info with calculated absolute offsets
        current_tensor_data_offset = data_start_offset
        for info in temp_tensor_infos:
            # Align the current_tensor_data_offset before assigning to tensor's offset
            current_tensor_data_offset = _align_offset(current_tensor_data_offset, 32)
            info["offset"] = current_tensor_data_offset
            _write_tensor_info(f, info["name"], info["n_dims"], info["shape"], info["dtype_id"], info["offset"])
            current_tensor_data_offset += info["data_len"]

        # Seek to the calculated start of tensor data block
        f.seek(data_start_offset)

        # Write Tensor Data
        for info in temp_tensor_infos:
            # Ensure we are at the correct offset before writing each tensor's data
            # This seek is crucial for ensuring data is written at its declared offset
            f.seek(info["offset"])
            f.write(info["tensor_data"])

    print(f"Successfully converted and saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Safetensors to GGUF format.")
    parser.add_argument("input_path", type=str, help="Path to the input .safetensors file or a directory containing safetensors files.")
    parser.add_argument("output_path", type=str, help="Path for the output .gguf file.")
    parser.add_argument("--quantization", type=str, default="none",
                        choices=["none", "Q4_0", "Q8_0"],
                        help="Type of quantization to apply (default: none).")

    args = parser.parse_args()

    print(f"Loading safetensors file(s) from: {args.input_path}")
    tensors = load_safetensors(args.input_path)
    print(f"Loaded {len(tensors)} tensors.")

    convert_to_gguf(tensors, args.output_path, args.quantization)

if __name__ == "__main__":
    main()