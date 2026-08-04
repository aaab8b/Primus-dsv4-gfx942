# DeepSeek-V4 SFT on gfx942 (MI308X / CDNA3)

Runs a 4-layer DeepSeek-V4 SFT at **128k context** exercising **all three** V4 attention
branches — dense+SWA, CSA, HCA — on a single 8×MI308X node.

Upstream's V4 support targets gfx950 / CDNA4 and, as shipped, only ever ran at
`seq_length=4096`. Several things that are invisible at 4k become hard blockers at 128k.
This directory carries the fixes and a one-click launcher.

```bash
# inside a rocm/primus:v26.5-pytorch2.12-te2.15 container, from the repo root
bash examples/deepseek-v4/gfx942/run_128k_dense_hca_csa.sh
```

The script is self-contained: it copies Megatron-LM out of the image (or fetches the
submodule), builds the SFT dataset on first run, and resolves every path relative to
itself. The only external requirement is a **local DeepSeek-V4 tokenizer directory** —
point `V4_TOKENIZER` at it, or leave it at `/apps/DeepSeek-V4-Flash`. Only
`tokenizer.json` / `tokenizer_config.json` are read.

## Scope — read this before quoting any number

* **Weights are randomly initialised.** No DeepSeek-V4 Megatron checkpoint exists, and
  Megatron-Bridge registers importers only for V2-Lite/V2/V3. This is *SFT-shaped training
  from scratch*, not fine-tuning. The tokenizer and the prompt/response loss mask are real.
* **4 layers, not 43.** `compress_ratios [0, 0, 4, 128]` was chosen to cover one layer of
  each branch type. The released model is 3 dense / 21 CSA / 20 HCA.
* **`TP=8` is required, not a tuning knob.** TP=1 OOMs at 128k (measured). See below.

## Results

8×MI308X, 4 layers, `[0, 0, 4, 128]`, TP=8, `triton_v2`, 10 steps:

| | |
|---|---|
| loss | 11.935 → 11.777, monotone |
| grad norm | 8.7 – 9.1, stable |
| nan iterations | 0 |
| peak memory | 188.63 GB / 191.98 GB (98.25%) |
| step time | 9.4 s |
| throughput | 85.7 TFLOP/s/GPU |

The 98.25% is not slack: the node must be otherwise idle.

## What had to change

### 1. P14 — shard attention heads across TP

`deepseek_v4_layer_specs.py`, `deepseek_v4_attention.py`, `indexer.py`,
`deepseek_v4_transformer_config.py`

V4 built `linear_q_up_proj` with `gather_output=True` and set
`num_attention_heads_per_partition = num_heads` (no `divide()`, unlike upstream Megatron's
`attention.py`). TP therefore sharded **weights only** — the `[B, S, H, head_dim]` query was
replicated on every rank, 64 KiB/token at V4-Flash width. The source called head-sharded
attention "tracked in P14"; it was never implemented.

Enabled by `v4_shard_attention_heads: true` (default **false**, so existing recipes are
unchanged). Requires `num_attention_heads % tp == 0` and `o_groups % tp == 0`.

The grouped-O projection makes this clean: it is block-diagonal over `o_groups`, and
`linear_o_b` is already row-parallel. At TP=8 with `o_groups=8` each rank owns exactly one
group, computes its partial product, and the row-parallel all-reduce sums them — algebraically
identical to the unsharded path (verified in float64: residual 2.5e-15).

* `q_up`: `gather_output=False`
* `o_a`: plain linear → column-parallel over `o_groups * o_lora_rank`
* `o_b`: `input_is_parallel=True`
* `num_heads`, `attn_sink`, grouped-O's `G`: all localised
* indexer `w_w` / `w_iuq`: sharded over heads, partial score sums all-reduced (the causal
  mask is 0/−inf and survives the sum unchanged)

**Side effect that matters:** at TP=8 the indexer sees 8 local heads instead of 64.

### 2. Fused indexer scoring at the real head count

`v4_attention_kernels/_triton_common/indexer_score.py`, `indexer_score_post.py`

Both Triton indexer kernels gated on `_SUPPORTED_H = (1, 2, 4, 8, 16)`, and their docstrings
describe "V4-Flash production" as `H=8`. The released config and Primus's own
`deepseek_v4_base.yaml` both set `index_n_heads=64`, so **at the real width these kernels
were unreachable** and the indexer silently fell back to the eager einsum — which
materialises `[B, S, H, P]`: 0.5 GiB at 4k, **512 GiB at 128k**.

* `_SUPPORTED_H` extended to include 32 and 64 (verified fwd+bwd against eager; fwd matches
  to 3e-7, bwd to bf16 quantisation noise, same as the already-supported H=8)
* `k_tile` hoisted out of the unrolled head loop in both kernels — it never depended on `h`,
  so H=64 was doing 63 redundant tile loads per block

### 3. int64 offsets in the indexer kernels

Same files. Every offset was int32. `s_offs * P` wraps negative once `S*P` exceeds 2³¹,
which for CSA (`P = S/4`) happens at `S ≈ 92682` — the store then lands out of bounds
*silently*. Bisected: 64k clean, 96k NaN, 128k NaN; matches the predicted threshold. 15
offsets promoted across the forward and backward kernels.

### 4. A dead 16 GiB allocation in the CSA path

`deepseek_v4_attention.py`

The CSA branch built a dense `[S, S]` sliding-window mask and passed it to `_csa_forward`,
whose own docstring says it is *"retained in the signature for back-compat but unused"* —
the function `del`s it on entry and the reference op rebuilds the mask from `swa_window`.
16 GiB at 128k, allocated to be thrown away. Now passes `None`.

### 5. gfx942 LDS budget

`v4_attention_kernels/_triton_v2/dsa_bwd_v4_triton.py`

The sparse-MLA backward is tuned for gfx950's 160 KB LDS; gfx942 has 64 KB and the stock
pipeline staging asks for 73728 B, so the kernel fails to **compile**. Added
`PRIMUS_DSA_BWD_NUM_STAGES` (and `PRIMUS_DSA_BWD_BLOCK_H`) to disable Triton's LDS
multi-buffering. Measured: this is the *only* kernel switch needed — `PRIMUS_HC_TRITON` and
`PRIMUS_DSA_DKV_SAFE` can stay at their defaults.

### 6. Context parallelism for the dense branch

`deepseek_v4_cp.py` (new), `v4_sparse_mla_adapter.py`, `sft/forward_step.py`

Not used by this 128k recipe, but included because the dense branch is index-driven and CP
was cheap to add there: a rank needs only the `d_window` post-RoPE KV rows to its left plus
its global row offset, and the sparse-MLA adapter then validates the window against global
positions while indexing the local `[boundary ++ local]` buffer. Verified bit-exact
(`atol=rtol=0`) against the unsharded path for CP=2/4, and end-to-end CP=1/2/4/8 agree to
6 significant figures. **CSA and HCA have no CP path** — those branches raise
`NotImplementedError` rather than compute silently-wrong results.

### 7. SFT plumbing

`sft/forward_step.py`, `examples/megatron/configs/MI355X/deepseek_v4_flash_4layer-BF16-sft.yaml`

* `mock_data` is fatal under `stage: sft` — the mock-data patch force-installs
  `NullTokenizer`, whose `text_to_ids` is `int(x)` per whitespace token and dies on prose.
  The config uses a real local tokenizer and a local `.jsonl`.
* `train_data_path` must be non-null or the pretrain data-prep hook tries to download
  bookcorpus and demands `HF_TOKEN`.
* `rope_type: rope` — the base preset says `yarn`, but the common-attention path asserts
  `rope`.
* `moe_router_enable_expert_bias: false` — expert bias requires `sigmoid` scoring, which
  conflicts with V4's `sqrtsoftplus`.
* `create_attention_mask_in_dataloader: false` — otherwise a `[S, S]` bool tensor.
* Contiguous CP sharding of the batch and a CP-aware loss reduction.

## Memory, measured

Per-branch peak (fwd+bwd, standalone, TP=8 → H_local=8, `head_dim=512`):

| branch | KiB/token |
|---|---|
| dense (cr=0) | 208 |
| HCA (cr=128) | 372 |
| indexer (CSA scoring) | 203, and **O(S²)** — `scores` is `S × S/4` |

At 128k that is ~150 GiB for four layers, which is where the 188.63 GB goes. Full activation
recompute does not help: these are *intra-layer* peaks.

## Known limits

* **`turbo` backend does not run on gfx942.** Its FlyDSL kernels emit
  `permlane16/32_swap` and `mfma_f32_16x16x32_bf16`, both CDNA4-only. The permlane
  butterfly has a `ds_bpermute` equivalent (ported, in a local patch to `primus_turbo`), but
  the MFMA tile shape does not — CDNA3's `mfma_f32_16x16x16bf16_1k` has half the K depth,
  so 25 call sites plus an inline-asm block would need re-tiling. `gluon*` and `flydsl_v1`
  hard-assert gfx950. **`triton_v2` is the fastest usable backend** (1.8× `triton_v1`).
* **turbo and the fused indexer want opposite TP.** turbo asserts
  `num_heads % 32 == 0` (so TP ≤ 2 at H=64) while the fused indexer needs H_local ≤ 16
  (TP ≥ 4). No TP satisfies both.
* **1M with CSA does not fit on one node.** Under CP the compressed pool must be
  all-gathered, so `P` grows back to the global `S/4` and the indexer's `scores` reaches
  ~139 GiB per rank. The fix is a streaming top-K that never materialises `scores` (what
  upstream gets from TRT-LLM's radix top-K); `torch.topk` needs the whole row.
