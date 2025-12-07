Run the project from within build-local.

make sure to run cmake and make to set up llama-cli

The way I usually run it is ./bin/llama-cli -p "<Your prompt here>" -m model.gguf -b batch_size(necessary for hmt style segmentation) -c context_length(must fit prompt) -n num_output_tokens

You can similarly quantize a model with llama-quantize.

Getting a model into .gguf format can be done with convert_hf_to_gguf.py if you have a safetensors file in HMT format. In order to properly match the expected format, you need to have your file merge
the lora layers into a single layer. This can be done by running main in HMT-pytorch with the same setup we're used to for accelerate, which will output the safetensors file. I pretty much just commented everything
that actually has to do with running/training the model out, so it'll just set up HMT and save it in safetensors format. The file structure after this is relevantly different; you can look through it by downloading safetensors_explorer if you want how it is structured once it comes out.

when doing the conversion from safetensors to gguf, you need to have a separate directory with the safetensors file and the tokenizer.json, tokenizer_config.json, special_tokens_map.json, and config.json files from the original llama 3.2 1b model. We basically inherit all of those properties since we use llama 3.2 1b as our base model.

Display this file in raw to actually see the file structure

SafeTensors Explorer - hmt_model/model.safetensors (1/1)
Use ↑/↓ to navigate, Enter/Space to expand/collapse, / to search, q to quit
================================================================================
▼ 📁 🔧 Metadata (0 tensors, 0 B)
    🏷️  memory_cell.model.model.embed_tokens.weight [string]: memory_cell.model.lm_head.weight
▼ 📁 cross_attn (2 tensors, 32.0 MB)
  ▼ 📁 wk (1 tensors, 16.0 MB)
      📄 weight [BF16, (4096, 2048), 16.0 MB]
  ▼ 📁 wq (1 tensors, 16.0 MB)
      📄 weight [BF16, (4096, 2048), 16.0 MB]
  📄 mem [F32, (1, 2048), 8.0 KB]
▼ 📁 memory_cell (149 tensors, 4.6 GB)
  ▼ 📁 mem_map (2 tensors, 32.0 MB)
    ▼ 📁 inv_linear (1 tensors, 16.0 MB)
        📄 weight [F32, (2048, 2048), 16.0 MB]
    ▼ 📁 linear (1 tensors, 16.0 MB)
        📄 weight [F32, (2048, 2048), 16.0 MB]
    📄 memory [F32, (1, 2048), 8.0 KB]
  ▼ 📁 model (146 tensors, 4.6 GB)
    ▼ 📁 lm_head (1 tensors, 1002.0 MB)
        📄 weight [F32, (128256, 2048), 1002.0 MB]
    ▼ 📁 model (145 tensors, 3.6 GB)
      ▼ 📁 layers (144 tensors, 3.6 GB)
        ▼ 📁 0 (9 tensors, 232.0 MB)
          ▼ 📁 input_layernorm (1 tensors, 8.0 KB)
              📄 weight [F32, (2048), 8.0 KB]
          ▼ 📁 mlp (3 tensors, 192.0 MB)
            ▼ 📁 down_proj (1 tensors, 64.0 MB)
                📄 weight [F32, (2048, 8192), 64.0 MB]
            ▼ 📁 gate_proj (1 tensors, 64.0 MB)
                📄 weight [F32, (8192, 2048), 64.0 MB]
            ▼ 📁 up_proj (1 tensors, 64.0 MB)
                📄 weight [F32, (8192, 2048), 64.0 MB]
          ▼ 📁 post_attention_layernorm (1 tensors, 8.0 KB)
              📄 weight [F32, (2048), 8.0 KB]
          ▼ 📁 self_attn (4 tensors, 40.0 MB)
            ▼ 📁 k_proj (1 tensors, 4.0 MB)
                📄 weight [F32, (512, 2048), 4.0 MB]
            ▼ 📁 o_proj (1 tensors, 16.0 MB)
                📄 weight [F32, (2048, 2048), 16.0 MB]
            ▼ 📁 q_proj (1 tensors, 16.0 MB)
                📄 weight [F32, (2048, 2048), 16.0 MB]
            ▼ 📁 v_proj (1 tensors, 4.0 MB)
                📄 weight [F32, (512, 2048), 4.0 MB]
        ▶ 📁 1 (9 tensors, 232.0 MB)
        ▶ 📁 2 (9 tensors, 232.0 MB)
        ▶ 📁 3 (9 tensors, 232.0 MB)
        ▶ 📁 4 (9 tensors, 232.0 MB)
        ▶ 📁 5 (9 tensors, 232.0 MB)
        ▶ 📁 6 (9 tensors, 232.0 MB)
        ▶ 📁 7 (9 tensors, 232.0 MB)
        ▶ 📁 8 (9 tensors, 232.0 MB)
        ▶ 📁 9 (9 tensors, 232.0 MB)
        ▶ 📁 10 (9 tensors, 232.0 MB)
        ▶ 📁 11 (9 tensors, 232.0 MB)
        ▶ 📁 12 (9 tensors, 232.0 MB)
        ▶ 📁 13 (9 tensors, 232.0 MB)
        ▶ 📁 14 (9 tensors, 232.0 MB)
        ▶ 📁 15 (9 tensors, 232.0 MB)
      ▼ 📁 norm (1 tensors, 8.0 KB)
          📄 weight [F32, (2048), 8.0 KB]


SafeTensors Explorer - llama.cpp/build-local/hmt-model-f32.gguf (1/1)
Use ↑/↓ to navigate, Enter/Space to expand/collapse, / to search, q to quit
================================================================================
    🏷️  tokenizer.ggml.add_sep_token [bool]: false
    🏷️  tokenizer.ggml.bos_token_id [u32]: 128000
    🏷️  tokenizer.ggml.eos_token_id [u32]: 128001
    🏷️  tokenizer.ggml.merges [array]: ["Ġ Ġ", "Ġ ĠĠĠ", ..., "éĶ ¦" (280147)]
    🏷️  tokenizer.ggml.model [string]: "gpt2"
    🏷️  tokenizer.ggml.pre [string]: "llama-bpe"
    🏷️  tokenizer.ggml.token_type [array]: [1, 1, ..., 3 (128256)]
    🏷️  tokenizer.ggml.tokens [array]: ["!", """, ..., "<|reserved_special_token_247|>...
▼ 📁 blk (144 tensors, 3.6 GB)
  ▼ 📁 0 (9 tensors, 232.0 MB)
    ▼ 📁 attn_k (1 tensors, 4.0 MB)
        📄 weight [F32, (2048, 512), 4.0 MB]
    ▼ 📁 attn_norm (1 tensors, 8.0 KB)
        📄 weight [F32, (2048), 8.0 KB]
    ▼ 📁 attn_output (1 tensors, 16.0 MB)
        📄 weight [F32, (2048, 2048), 16.0 MB]
    ▼ 📁 attn_q (1 tensors, 16.0 MB)
        📄 weight [F32, (2048, 2048), 16.0 MB]
    ▼ 📁 attn_v (1 tensors, 4.0 MB)
        📄 weight [F32, (2048, 512), 4.0 MB]
    ▼ 📁 ffn_down (1 tensors, 64.0 MB)
        📄 weight [F32, (8192, 2048), 64.0 MB]
    ▼ 📁 ffn_gate (1 tensors, 64.0 MB)
        📄 weight [F32, (2048, 8192), 64.0 MB]
    ▼ 📁 ffn_norm (1 tensors, 8.0 KB)
        📄 weight [F32, (2048), 8.0 KB]
    ▼ 📁 ffn_up (1 tensors, 64.0 MB)
        📄 weight [F32, (2048, 8192), 64.0 MB]
  ▶ 📁 1 (9 tensors, 232.0 MB)
  ▶ 📁 2 (9 tensors, 232.0 MB)
  ▶ 📁 3 (9 tensors, 232.0 MB)
  ▶ 📁 4 (9 tensors, 232.0 MB)
  ▶ 📁 5 (9 tensors, 232.0 MB)
  ▶ 📁 6 (9 tensors, 232.0 MB)
  ▶ 📁 7 (9 tensors, 232.0 MB)
  ▶ 📁 8 (9 tensors, 232.0 MB)
  ▶ 📁 9 (9 tensors, 232.0 MB)
  ▶ 📁 10 (9 tensors, 232.0 MB)
  ▶ 📁 11 (9 tensors, 232.0 MB)
  ▶ 📁 12 (9 tensors, 232.0 MB)
  ▶ 📁 13 (9 tensors, 232.0 MB)
  ▶ 📁 14 (9 tensors, 232.0 MB)
  ▶ 📁 15 (9 tensors, 232.0 MB)
▼ 📁 hmt (6 tensors, 96.0 MB)
  ▼ 📁 cross_attn_k (1 tensors, 32.0 MB)
      📄 weight [F32, (2048, 4096), 32.0 MB]
  ▼ 📁 cross_attn_q (1 tensors, 32.0 MB)
      📄 weight [F32, (2048, 4096), 32.0 MB]
  ▼ 📁 initial_memory (1 tensors, 8.0 KB)
      📄 weight [F32, (2048), 8.0 KB]
  ▼ 📁 mem_map (2 tensors, 32.0 MB)
    ▼ 📁 inv (1 tensors, 16.0 MB)
        📄 weight [F32, (2048, 2048), 16.0 MB]
    ▼ 📁 linear (1 tensors, 16.0 MB)
        📄 weight [F32, (2048, 2048), 16.0 MB]
  ▼ 📁 summary_prompt (1 tensors, 8.0 KB)
      📄 weight [F32, (2048), 8.0 KB]
▼ 📁 output (1 tensors, 1002.0 MB)
    📄 weight [F32, (2048, 128256), 1002.0 MB]
▼ 📁 output_norm (1 tensors, 8.0 KB)
    📄 weight [F32, (2048), 8.0 KB]
▼ 📁 token_embd (1 tensors, 1002.0 MB)
    📄 weight [F32, (2048, 128256), 1002.0 MB]

Total Parameters: 1.5B | Selected: 83/83 | Scroll: 20 | Matches: 83



Total Parameters: 1.3B | Selected: 1/58 | Scroll: 0 | Matches: 58
