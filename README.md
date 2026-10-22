# Multi-Token Prediction (MTP) Speculative Decoding Engine

A hardware-aware speculative decoding engine for accelerating LLM inference.
It drafts multiple future tokens in parallel with a lightweight MTP head, then
verifies them in a single forward pass of the target model.

The engine targets **Nvidia CUDA** in production (validated end-to-end on an
**H100**) and runs a bit-identical eager fallback on **Apple Silicon (MPS)** and
CPU for local development.

---

## Table of contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Modules](#modules)
- [How the code works](#how-the-code-works)
- [Validation](#validation)
- [Benchmarks](#benchmarks)
- [Running it](#running-it)
- [Known trade-offs](#known-trade-offs)
- [Documentation set](#documentation-set)

---

## What it does

Speculative decoding trades the target model's per-token latency for cheaper
draft latency plus a verification pass:

1. A **draft head** (a lightweight MTP module) proposes `depth + 1` candidate tokens.
2. The **target model** verifies all candidates in a single forward pass.
3. Accepted tokens are committed; on the first mismatch the KV cache is rolled
   back and the target's corrected token is appended.

It supports both:

- a **strictly-causal sequential chain** (trunk mode), and
- a **branched speculative tree** (tree mode, default), where the draft proposes
  a tree of candidate paths and the verifier checks all of them against a static
  causal tree mask in one pass.

An **acceptance-rate driveable** "measurable speedup" harness (`bench_speedup.py`)
reports the acceptance rate α and the wall-clock vs. theoretical speedup, so you
can tune the actual lever that matters (a trained MTP head, a cheap draft, depth).

---

## Architecture

```
                 +---------------------------+
   input_ids --> |   SpeculativeDecodingEngine|
                 |   (orchestration)         |
                 +-------------+-------------+
                               |
              +----------------+----------------+
              |                                 |
   +----------v-----------+         +-----------v-----------+
   |   Draft (MTP)        |         |   Verify (target)     |
   |  tree/trunk topology |         |  single-forward pass  |
   |  fused monolith head |         |  rejection sampling   |
   +----------+-----------+         +-----------+-----------+
              |   draft_tokens                   |
              +----------- verification ---------+
                          |  accept / rollback
              +-----------v-----------+
              |   KV cache (paged /  |
              |   radix shared-page) |
              +-----------------------+
```

Key properties:

- **CUDA-first** — `DEVICE` resolves `cuda > mps > cpu`. (An earlier release only
  checked MPS/CPU and silently ran on CPU even with an NVIDIA GPU present.)
- **Bit-identical fallback** — every Triton kernel has an eager PyTorch
  reference that produces identical numbers when Triton is unavailable.
- **CUDA Graph capture** — the draft path is recorded into a
  `torch.cuda.CUDAGraph` so a full draft step replays as one driver call.
- **Tensor parallelism** — row/column-parallel linear layers with NCCL
  collectives that no-op when `tp_size == 1`.
- **TF32 disabled** — `torch.backends.cuda.matmul.allow_tf32 = False`, so
  `tl.dot` and the torch reference share identical fp32 numerics.

---

## Modules

### Core

| File | Purpose |
|------|---------|
| `mtp_kernels.py` | Triton kernels + eager fallbacks: RMSNorm, fused RMSNorm+linear, fused residual+RMSNorm, FlashAttention-style causal attention, in-register bit-mask attention, FP8/INT4 quantized variants, paged attention, tree attention, the fused tree-attention→RMSNorm→linear monolith (eager on CUDA — see Known trade-offs), packed FP8 scales, auto-tuner prewarm. |
| `mtp_engine.py` | PyTorch architecture + orchestration. RMSNorm, attention, MTP heads (shared `lm_head`), `KVManager`, `GraphMemoryPool`, `PagedKVCache`, `QuantScaleState`, `SpeculativeDraftTree`, and `SpeculativeDecodingEngine` (`generate`, `generate_fused`, `generate_optimized`). |
| `mtp_parallel.py` | Tensor parallelism: `RowParallelLinear`, `ColumnParallelLinear`, `ParallelPrimitives`, `SegmentedGraphRunner`. |

### Optimization extensions

| File | Purpose |
|------|---------|
| `mtp_fused_verify.py` | Fused verification kernels: `fused_verify` (single-token linear→argmax→accept) and `fused_multi_verify` (whole-chain linear→argmax→acceptance→rollback index in one pass). |
| `mtp_radix.py` | RadixAttention shared-page KV cache with dedup + `batch_prefix_match` (vectorized/in-kernel prefix match — Triton only when input is on CUDA). |
| `mtp_topk.py` | Top-k tree drafting (`sample_top_k`, `build_tree_mask`, `expand_branches`). |
| `mtp_pipeline.py` | Dual-stream pipelining + async KV eviction + `RollbackCommitFence` + `TripleStreamPipeline` (transfer/draft/verify streams) + `DynamicGraphBatchSlicer` (pre-captured batch graphs). |
| `mtp_perf_hardening.py` | Cross-cutting performance primitives: power-of-2/vocab padding, `PrecomputedConstants`, `PinnedBufferPool` (DMA ingestion), `CachedLinear` (no transpose in forward), greedy `argmax_greedy`/`sample_token`, `FlatPageTable`, `maybe_compile`. |

### Helper scripts

| File | Purpose |
|------|---------|
| `bench_memory.py` | Batch-scaling + radix-context + speculative-vs-autoregressive matrix. |
| `bench_optimizations.py` | Fusion / radix / fence latency benchmarks. |
| `bench_speedup.py` | Acceptance-rate + wall-clock vs. theoretical speculative speedup. |
| `test_perf_hardening.py` | Validation + benchmark of the performance-hardening primitives. |
| `test_engine.py` | Original standalone test harness. |
| `test_new_features.py` | Tests for the optimization extension modules. |
| `test_tree.py`, `test_tree_cycle.py` | Tree drafting / dynamic-lookahead e2e checks. |

---

## How the code works

### The draft chain (sequential / trunk mode)

Each MTP head normalizes its input, projects to vocab logits via the **shared**
`lm_head` (one weight matrix for the whole chain, keeping memory flat), and the
sampled token's embedding is combined (RMSNorm) with the head's output hidden
state and passed through a lightweight transformer layer into the next head.
This is the strictly-causal sequential chain.

### The draft tree (tree mode, default)

`SpeculativeDraftTree` builds a **statically padded breadth-first tree** with a
fixed bounding box (`depth × max_branches`). The tree is flattened into a
contiguous linear array (`depth_offsets`, `branch_offsets`) so kernels parse the
structure with pure `index + branch_id` pointer arithmetic — no dynamic sort, no
host-to-device sync. A 2D causal `tree_mask` marks which ancestor nodes each
node may attend to.

`draft_monolith_body` runs the whole tree through `fused_tree_attn_rmsnorm_linear`
(online softmax → RMSNorm → linear projection, writing only the final projection
to HBM). On CUDA this currently uses the eager reference — see
[Known trade-offs](#known-trade-offs).

### Verification

`SpeculativeDecodingEngine.generate`:

1. Drafts `depth` tokens.
2. Runs the target model once over `cat(last_token, draft_tokens)`.
3. Compares target tokens against draft tokens left-to-right.
4. On the first mismatch, rolls the KV cache back and appends the target's
   corrected token.

`mtp_fused_verify.fused_verify` / `fused_multi_verify` fuse the logit projection,
the argmax/draft comparison, and the acceptance/rollback index into a single
kernel (eliminating `depth` per-token host→device round-trips).

### KV cache

- `PagedKVCache` — logical tokens map to non-contiguous physical pages via a
  static contiguous INT32 `page_table`. `swap_logical_pages` does zero-copy
  swaps; `compact` defragments at checkpoint boundaries.
- `RadixKVCache` — a trie of pages so divergent draft paths share a common
  prefix instead of duplicating it, with geometric (`×2`) page-pool growth.

### CUDA Graph

`compile_draft_graph` allocates a `GraphMemoryPool` (fixed-address input/output
buffers), pre-materializes the tree mask on device, warms up, and captures the
draft body into a `torch.cuda.CUDAGraph`. `draft_speculative_tokens` replays it
with a single `replay()` call.

### Quantization

FP8 (e4m3) weights and INT4 packed weights, with per-channel/per-group scales
stored in fixed `GraphMemoryPool` buffers (`QuantScaleState`), so scales update
in place without allocation. Packed scales are also exposed via
`pack_scales`/`unpack_scale_tensor`.

---

## Validation

All checks run on real CUDA (H100) **and** on MPS/CPU via the eager fallback.

```
$ CUDA_VISIBLE_DEVICES=0 python mtp_engine.py
DEVICE=cuda  HAS_TRITON=True
[kernels] rmsnorm + fused linear: OK
[kernels] fused causal attention: OK
[pipeline] async == sync (bit-identical): OK
[graph] graph == sync (bit-identical): OK
[tp] tp_size=1, collectives are identity no-ops: OK
[tp] RowParallelLinear / ColumnParallelLinear (tp_size=1): OK
[paged-kv] reserve / zero-copy write / swap / compact: OK
[tree] tree active_nodes=32, trunk active_nodes=4: OK
[quant] fused rmsnorm+linear fp8 == fp32 reference: OK
[quant] fused causal attention fp8 == fp32 reference: OK
[quant] fused int4 matmul == dequant reference (bit-identical): OK
[tree] fused tree attention (all-ones mask) == causal attention: OK
[quant] GraphMemoryPool scale buffers + QuantScaleState (fixed-memory): OK
[mode:trunk] async == sync (bit-identical): OK
[mode:tree] async == sync (bit-identical): OK

ALL VALIDATION CHECKS PASSED
```

---

## Benchmarks

Full measured numbers live in [`BENCHMARKS.md`](BENCHMARKS.md). Headlines on the H100:

| Result | Value |
|---|---|
| Greedy `argmax` vs softmax+multinomial | **5.42×** |
| Fused `add+rmsnorm` vs staged | **1.84×** |
| Fused verification acceptance stage | **6.79×** |
| Radix `batch_prefix_match` (4096 requests) | **91×** |
| Radix dense-context dedup (64k tokens) | **7.94×** memory reduction |

The **speculative-decoding wall-clock speedup** is reported honestly at `<1×` for
the *synthetic* 64-dim target + untrained MTP head (α ≈ 0.046). See
[`BENCHMARKS.md` §7](BENCHMARKS.md) for the recipe to flip it positive on a
production target.

---

## Running it

```bash
# one-time dependency install (torch; Triton only on Nvidia)
python -m venv .venv && source .venv/bin/activate
pip install torch

# full validation (CUDA or MPS/CPU)
python mtp_engine.py

# extension-module tests
python test_new_features.py

# original harness
python test_engine.py

# tree / hardening / speedup / memory / optimization benches
python test_tree.py
python test_perf_hardening.py
python bench_speedup.py
python bench_memory.py
python bench_optimizations.py
```

The engine exposes `SpeculativeDecodingEngine` with constructor options:

- `draft_mode="tree"` (default) or `"trunk"`
- `max_branches` (tree branching factor, default 8)
- `async_mode` (currently serialized for determinism — see Known trade-offs)
- `use_fp8` (quantized kernels)
- `tp_group` (torch.distributed process group for tensor parallelism)

```python
from mtp_engine import SequentialMTPModule, KVManager, SpeculativeDecodingEngine

mtp = SequentialMTPModule(hidden_dim=64, vocab_size=256, depth=4)
engine = SpeculativeDecodingEngine(mtp, main_model, KVManager(512),
                                   draft_mode="tree")
out = engine.generate(input_ids, max_new_tokens=8)
```

---

## Known trade-offs

Three paths are currently routed to their proven-correct implementations rather
than their full-Triton form (each was a latent cross-stream / NaN / aliasing
bug only visible on real CUDA):

1. **Async draft/verify overlap** — the draft is serialized with the verify
   (single deterministic loop). True overlap requires a double-buffered
   `GraphMemoryPool`. The CUDA-graph draft replay still provides driver-overhead
   reduction.
2. **`fused_tree_attn_rmsnorm_linear`** (tree monolith) — uses its eager
   reference, because its Triton masked softmax computes `-inf/-inf → NaN` for
   zero-valid-key nodes and propagates non-deterministically. Needs a NaN-safe
   masked-softmax rewrite to go back in-kernel.
3. **`fused_multi_verify` full GEMM** — the acceptance stage is Triton (6.79×);
   the full linear→argmax→accept fusion uses an eager unblocked GEMM. Needs a
   blocked Triton GEMM for the full fused speedup.

These are correctness-preserving fallbacks; all other kernels run in Triton and
are validated bit-identical against their references.

---

## Documentation set

| File | Purpose |
|------|---------|
| `README.md` | This file — architecture, modules, usage. |
| `HARDENING_STATUS.md` | Checklist → implementation audit map (performance hardening). |
| `BENCHMARKS.md` | Measured H100 numbers + interpretation. |
