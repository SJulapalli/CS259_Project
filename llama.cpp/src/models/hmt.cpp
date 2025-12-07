#include "ggml.h"
#include "llama-context.h"
#include "models.h"

llm_build_hmt::llm_build_hmt(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;
    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);

    const float kq_scale =
        hparams.f_attention_scale == 0.0f ? 1.0f / sqrtf(float(n_embd_head)) : hparams.f_attention_scale;

    struct ggml_tensor * hmt_wq = model.hmt_cross_q;
    struct ggml_tensor * hmt_wk = model.hmt_cross_k;
    struct ggml_tensor * hmt_mem;

    // if (params.ctx->hmt_memory_buffer) {
    //     hmt_mem = params.ctx->hmt_memory_buffer;
    // } else {
    //     hmt_mem         = model.hmt_initial_memory;
    //     hmt_mem->buffer = model.hmt_initial_memory->buffer;
    // }

    // Use Context if available, else Model initial memory
    if (params.ctx && params.ctx->hmt_memory_buffer) {
        hmt_mem = params.ctx->hmt_memory_buffer;
    } else {
        hmt_mem = model.hmt_initial_memory;
    }

    ggml_tensor * inpL_raw = build_inp_embd(model.tok_embd);

    ggml_tensor * inpL_dirty = ggml_dup_tensor(ctx0, inpL_raw);
    ggml_set_name(inpL_dirty, "inpL_dirty");

    struct ggml_tensor * copy_raw_to_dirty = ggml_cpy(ctx0, inpL_raw, inpL_dirty);
    ggml_build_forward_expand(gf, copy_raw_to_dirty);

    ggml_tensor * inp_pos  = build_inp_pos();
    auto *        inp_attn = build_attn_inp_kv();

    ggml_tensor * cur = inpL_dirty;

    // hmt_mem = model.hmt_initial_memory;

    // printf("inpL_raw shape: %ld %ld %ld %ld\n", inpL_raw->ne[0], inpL_raw->ne[1], inpL_raw->ne[2], inpL_raw->ne[3]);
    // printf("inp_pos shape: %ld %ld %ld %ld\n", inp_pos->ne[0], inp_pos->ne[1], inp_pos->ne[2], inp_pos->ne[3]);

    // Soft Prompts
    if (model.hmt_summary_prompt) {
        ggml_tensor * prompt = ggml_reshape_2d(ctx0, model.hmt_summary_prompt, model.hmt_summary_prompt->ne[0], 1);
        prompt->buffer       = model.hmt_summary_prompt->buffer;

        int64_t n_tokens = cur->ne[1];

        // Target: Second to last (N-2), or 0 if N < 2
        int64_t target_idx = (n_tokens >= 2) ? (n_tokens - 2) : 0;
        size_t  offset     = target_idx * cur->nb[1];

        struct ggml_tensor * target_token = ggml_view_2d(ctx0, cur, hparams.n_embd, 1, cur->nb[1], offset);

        // Maybe Replace: Overwrite target slot
        struct ggml_tensor * overwrite_op = ggml_cpy(ctx0, prompt, target_token);
        ggml_build_forward_expand(gf, overwrite_op);

        cb(cur, "hmt_inp_prompt_2nd_last", -1);
    }

    // printf("cur_pos shape: %ld %ld %ld %ld\n", cur_pos->ne[0], cur_pos->ne[1], cur_pos->ne[2], cur_pos->ne[3]);

    // Loop 1: Run backbone to get the summary of the current segment.
    for (int il = 0; il < 1; ++il) {
        ggml_tensor * inpSA = cur;

        cur = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        {
            const int64_t n_tokens_layer = cur->ne[1];

            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
            ggml_tensor * Qcur         = build_lora_mm(model.layers[il].wq, cur);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
            }

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
            }

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
            }

            // Reshape using dynamic n_tokens_layer
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens_layer);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens_layer);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens_layer);
            // printf("Qcur shape: %ld %ld %ld %ld\n", Qcur->ne[0], Qcur->ne[1], Qcur->ne[2], Qcur->ne[3]);
            // printf("Kcur shape: %ld %ld %ld %ld\n", Kcur->ne[0], Kcur->ne[1], Kcur->ne[2], Kcur->ne[3]);
            // printf("Vcur shape: %ld %ld %ld %ld\n", Vcur->ne[0], Vcur->ne[1], Vcur->ne[2], Vcur->ne[3]);
            // printf("inp_pos shape: %ld %ld %ld %ld\n", inp_pos->ne[0], inp_pos->ne[1], inp_pos->ne[2], inp_pos->ne[3]);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            if (hparams.use_kq_norm) {
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
            }

            // KV Cache updates here are overwritten in Pass 2.
            cur = build_attn(inp_attn,                        // Input (use inp_attn variable)
                             model.layers[il].wo,             // WO
                             model.layers[il].bo,             // BO
                             Qcur, Kcur, Vcur,                // Q, K, V
                             (struct ggml_tensor *) nullptr,  // Mask
                             (struct ggml_tensor *) nullptr,  // KQV Merged
                             (struct ggml_tensor *) nullptr,  // Pos
                             kq_scale,                        // Scale
                             il                               // Layer ID
            );
            cb(cur, "attn_out", il);
        }

        // FFN
        cur                   = ggml_add(ctx0, cur, inpSA);
        ggml_tensor * ffn_inp = cur;
        cur                   = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);

        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = build_ffn(cur, model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL, model.layers[il].ffn_gate,
                            model.layers[il].ffn_gate_b, NULL, model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                            NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        } else {
            cur =
                build_moe_ffn(cur, model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
                              model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps, nullptr, n_expert,
                              n_expert_used, LLM_FFN_SILU, true, false, 0.0, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cb(cur, "l_out", il);
    }

    // Extract Summary (Last token of Pass 1)
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);

    // View last column: [n_embd, 1]
    // Uses dynamic token count (cur->ne[1])
    struct ggml_tensor * summary = ggml_view_1d(ctx0, cur, hparams.n_embd, (cur->ne[1] - 1) * cur->nb[1]);
    cb(summary, "hmt_summary", -1);

    // printf("Starting Cross Attention\n");

    // Cross Attention
    ggml_tensor * retrieved_context = nullptr;

    if (hmt_wq && hmt_wk && hmt_mem) {
        // Q = Summary * Wq
        // printf("MatMul Q\n");
        struct ggml_tensor * Q = build_lora_mm(hmt_wq, summary);

        // K = Memory * Wk
        // printf("MatMul K\n");
        struct ggml_tensor * K = build_lora_mm(hmt_wk, hmt_mem);

        // V = Memory
        struct ggml_tensor * V = hmt_mem;

        // Scores = Q * K^T
        // printf("MatMul Scores\n");
        struct ggml_tensor * scores = ggml_mul_mat(ctx0, K, Q);
        scores                      = ggml_scale(ctx0, scores, kq_scale);
        scores                      = ggml_soft_max(ctx0, scores);
        cb(scores, "hmt_scores", -1);

        // Context = V * Scores
        // V shape: [2048, 100] (ne[0]=2048)
        // Scores shape: [100, 1] (ne[0]=100)
        // ggml_mul_mat contracts ne[0]. We need V to have ne[0]=100.
        struct ggml_tensor * V_trans = ggml_transpose(ctx0, V);
        struct ggml_tensor * V_cont  = ggml_cont(ctx0, V_trans);

        retrieved_context = ggml_mul_mat(ctx0, V_cont, scores);
        // printf("Retrieved Context Shape: %ld %ld %ld %ld\n", retrieved_context->ne[0], retrieved_context->ne[1],
        //    retrieved_context->ne[2], retrieved_context->ne[3]);
        retrieved_context = ggml_reshape_2d(ctx0, retrieved_context, hparams.n_embd, 1);
        cb(retrieved_context, "hmt_context", -1);
    }

    // printf("Retrieved Context\n");

    // printf("cur shape: %ld %ld %ld %ld\n", cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);

    // Create new tensor and copy original data
    struct ggml_tensor * inpL_pass2 = ggml_new_tensor_2d(ctx0, inpL_raw->type, inpL_raw->ne[0], inpL_raw->ne[1]);
    ggml_set_name(inpL_pass2, "inpL_pass2");

    // Copy the raw input entirely (so we have valid text data)
    struct ggml_tensor * copy_op = ggml_cpy(ctx0, inpL_raw, inpL_pass2);
    ggml_build_forward_expand(gf, copy_op);

    if (retrieved_context) {
        retrieved_context = ggml_cont(ctx0, retrieved_context);

        // RMS norm to force retrieved context to match original llama embeddings
        struct ggml_tensor * ctx_norm = ggml_rms_norm(ctx0, retrieved_context, 1e-5f);

        // Scale down retrieved context to prevent overpowering the text. 0.015 is very subtle, but anything above .03 seems too strong.
        struct ggml_tensor * ctx_scaled = ggml_scale(ctx0, ctx_norm, 0.015f);

        int64_t last_idx = inpL_pass2->ne[1] - 1;
        size_t  offset   = last_idx * inpL_pass2->nb[1];

        // View the Last Token in the destination buffer
        struct ggml_tensor * dest_tail = ggml_view_2d(ctx0, inpL_pass2, hparams.n_embd, 1, inpL_pass2->nb[1], offset);

        // Compute sum
        struct ggml_tensor * mixed_tail = ggml_add(ctx0, dest_tail, ctx_scaled);

        // Write the result back
        struct ggml_tensor * update_op = ggml_cpy(ctx0, mixed_tail, dest_tail);
        ggml_build_forward_expand(gf, update_op);

        cb(inpL_pass2, "hmt_inp_p2_injected", -1);
    }

    cur = inpL_pass2;

    // printf("!!Cur Shape: %ld %ld %ld %ld\n", cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3]);
    // printf("Mem index: %ld\n", params.ctx->hmt_mem_index);
    // Loop 2
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = cur;

        cur = build_norm(cur, model.layers[il].attn_norm, NULL, LLM_NORM_RMS, il);

        {
            const int64_t n_tokens_layer = cur->ne[1];

            ggml_tensor * rope_factors = model.get_rope_factors(cparams, il);
            ggml_tensor * Qcur         = build_lora_mm(model.layers[il].wq, cur);
            if (model.layers[il].bq) {
                Qcur = ggml_add(ctx0, Qcur, model.layers[il].bq);
            }

            ggml_tensor * Kcur = build_lora_mm(model.layers[il].wk, cur);
            if (model.layers[il].bk) {
                Kcur = ggml_add(ctx0, Kcur, model.layers[il].bk);
            }

            ggml_tensor * Vcur = build_lora_mm(model.layers[il].wv, cur);
            if (model.layers[il].bv) {
                Vcur = ggml_add(ctx0, Vcur, model.layers[il].bv);
            }

            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens_layer);
            Kcur = ggml_reshape_3d(ctx0, Kcur, n_embd_head, n_head_kv, n_tokens_layer);
            Vcur = ggml_reshape_3d(ctx0, Vcur, n_embd_head, n_head_kv, n_tokens_layer);

            Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);
            Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, rope_factors, n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                                 ext_factor, attn_factor, beta_fast, beta_slow);

            if (hparams.use_kq_norm) {
                Qcur = ggml_rms_norm(ctx0, Qcur, hparams.f_norm_rms_eps);
                Kcur = ggml_rms_norm(ctx0, Kcur, hparams.f_norm_rms_eps);
            }

            cur = build_attn(inp_attn,                        // Input
                             model.layers[il].wo,             // WO
                             model.layers[il].bo,             // BO
                             Qcur, Kcur, Vcur,                // Q, K, V
                             (struct ggml_tensor *) nullptr,  // Mask
                             (struct ggml_tensor *) nullptr,  // KQV Merged
                             (struct ggml_tensor *) nullptr,  // Pos
                             kq_scale,                        // Scale
                             il                               // Layer ID
            );
            cb(cur, "attn_out_p2", il);
        }

        cur                   = ggml_add(ctx0, cur, inpSA);
        ggml_tensor * ffn_inp = cur;
        cur                   = build_norm(cur, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il);

        if (model.layers[il].ffn_gate_inp == nullptr) {
            cur = build_ffn(cur, model.layers[il].ffn_up, model.layers[il].ffn_up_b, NULL, model.layers[il].ffn_gate,
                            model.layers[il].ffn_gate_b, NULL, model.layers[il].ffn_down, model.layers[il].ffn_down_b,
                            NULL, NULL, LLM_FFN_SILU, LLM_FFN_PAR, il);
        } else {
            cur =
                build_moe_ffn(cur, model.layers[il].ffn_gate_inp, model.layers[il].ffn_up_exps,
                              model.layers[il].ffn_gate_exps, model.layers[il].ffn_down_exps, nullptr, n_expert,
                              n_expert_used, LLM_FFN_SILU, true, false, 0.0, LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX, il);
        }

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);  // output vector mapping
        cb(cur, "l_out_p2", il);
    }

    // Write new summary from Pass 2 back to persistent memory.
    if (params.ctx && params.ctx->hmt_memory_buffer) {
        // Extract new Summary
        struct ggml_tensor * new_summary = ggml_view_1d(ctx0, cur, hparams.n_embd, (cur->ne[1] - 1) * cur->nb[1]);

        // Projection (Linear / Inverse)
        if (model.hmt_mem_map_lin) {
            new_summary = build_lora_mm(model.hmt_mem_map_lin, new_summary);
        }

        // Write to Context Memory
        struct ggml_tensor * update_op = ggml_cpy(ctx0, new_summary, params.ctx->hmt_memory_buffer);
        // int32_t current_idx = params.ctx->hmt_mem_index;

        // struct ggml_tensor * new_summary_slot = ggml_view_1d(ctx0, params.ctx->hmt_memory_buffer, hparams.n_embd,
        //                                                      current_idx * params.ctx->hmt_memory_buffer->nb[1]);
        // new_summary_slot->buffer              = params.ctx->hmt_memory_buffer->buffer;

        // struct ggml_tensor * update_op = ggml_cpy(ctx0, new_summary, new_summary_slot);
        ggml_build_forward_expand(gf, update_op);
    }

    if (false && cur->ne[1] > inpL_pass2->ne[1]) {
        cur = ggml_view_2d(ctx0, cur, hparams.n_embd, inpL_pass2->ne[1], cur->nb[1], cur->nb[1]);
    }

    // Final Output Generation
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
