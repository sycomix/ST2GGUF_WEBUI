import unittest
import os
import numpy as np
import struct
import shutil
from safetensors import safe_open
from converter import load_safetensors, convert_to_gguf, GGUF_DTYPE_MAP
from create_dummy_safetensors import create_dummy_safetensors

class TestConverter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dummy_safetensors_path = "dummy_model.safetensors"
        cls.dummy_sharded_model_dir = "dummy_sharded_model"
        create_dummy_safetensors(cls.dummy_safetensors_path, sharded=False)
        create_dummy_safetensors(cls.dummy_sharded_model_dir, sharded=True)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(cls.dummy_safetensors_path):
            os.remove(cls.dummy_safetensors_path)
        if os.path.exists(cls.dummy_sharded_model_dir):
            shutil.rmtree(cls.dummy_sharded_model_dir)
        if os.path.exists("output.gguf"):
            os.remove("output.gguf")
        if os.path.exists("output_q4_0.gguf"):
            os.remove("output_q4_0.gguf")
        if os.path.exists("output_q8_0.gguf"):
            os.remove("output_q8_0.gguf")
        if os.path.exists("output_sharded.gguf"):
            os.remove("output_sharded.gguf")
        if os.path.exists("output_sharded_q4_0.gguf"):
            os.remove("output_sharded_q4_0.gguf")

    def test_load_safetensors_single_file(self):
        tensors = load_safetensors(self.dummy_safetensors_path)
        self.assertGreater(len(tensors), 0)
        self.assertIn("model.embed_tokens.weight", tensors)
        self.assertEqual(tensors["model.embed_tokens.weight"].shape, (10, 5))

    def test_load_safetensors_sharded_model(self):
        tensors = load_safetensors(self.dummy_sharded_model_dir)
        self.assertGreater(len(tensors), 0)
        self.assertIn("model.embed_tokens.weight", tensors)
        self.assertIn("model.layers.0.mlp.down_proj.weight", tensors)
        self.assertEqual(tensors["model.embed_tokens.weight"].shape, (10, 5))
        self.assertEqual(tensors["model.layers.0.mlp.down_proj.weight"].shape, (5, 10))

    def test_convert_to_gguf_no_quantization(self):
        output_path = "output.gguf"
        tensors = load_safetensors(self.dummy_safetensors_path)
        convert_to_gguf(tensors, output_path, "none")
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_convert_to_gguf_q4_0_quantization(self):
        output_path = "output_q4_0.gguf"
        tensors = load_safetensors(self.dummy_safetensors_path)
        convert_to_gguf(tensors, output_path, "Q4_0")
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

        # Basic check for GGUF structure (magic and version)
        with open(output_path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            version = struct.unpack('<I', f.read(4))[0]
            self.assertEqual(magic, 0x46554747) # GGUF_MAGIC
            self.assertEqual(version, 2) # GGUF_VERSION

    def test_convert_to_gguf_q8_0_quantization(self):
        output_path = "output_q8_0.gguf"
        tensors = load_safetensors(self.dummy_safetensors_path)
        convert_to_gguf(tensors, output_path, "Q8_0")
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

        # Basic check for GGUF structure (magic and version)
        with open(output_path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            version = struct.unpack('<I', f.read(4))[0]
            self.assertEqual(magic, 0x46554747) # GGUF_MAGIC
            self.assertEqual(version, 2) # GGUF_VERSION

    def test_convert_sharded_to_gguf_no_quantization(self):
        output_path = "output_sharded.gguf"
        tensors = load_safetensors(self.dummy_sharded_model_dir)
        convert_to_gguf(tensors, output_path, "none")
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

    def test_convert_sharded_to_gguf_q4_0_quantization(self):
        output_path = "output_sharded_q4_0.gguf"
        tensors = load_safetensors(self.dummy_sharded_model_dir)
        convert_to_gguf(tensors, output_path, "Q4_0")
        self.assertTrue(os.path.exists(output_path))
        self.assertGreater(os.path.getsize(output_path), 0)

if __name__ == '__main__':
    unittest.main()