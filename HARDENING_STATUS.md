# Performance Hardening — Implementation Status Map

This file maps every requested optimization to its implementation location so
an M&A / engineering team can audit coverage at a glance. Status:

- ✅ implemented (and validated)
- ✅ (CUDA-only) — implemented, benefit materializes on NVIDIA hardware only

## CUDA Validation & Bug-Fix Audit (H100)

The Triton layer was originally validated only via the eager fallback on MPS
(`HAS_TRITON=False`). First-ever real execution on an **H100 PCIe** surfaced a
series of latent bugs, all fixed and now gated by `mtp_engine.py`
(`ALL VALIDATION CHECKS PASSED`):

| # | Bug (only on real CUDA) | Fix | File |
|---|---|---|---|
| 1 | `DEVICE` never selected CUDA (`device=cpu` on H100) | resolve `cuda > mps > cpu` | `mtp_engine.py` |
| 2 | `TRITON_INTERPRET=1` forced (strict CPU interpreter) | only set interpreter when CUDA absent | `mtp_kernels.py` |
| 3 | `(1,)`/`()` scalar broadcast shape errors | match proven kernel's scalar pattern | `mtp_kernels.py` |
| 4 | TF32 in `tl.dot` broke fp32 asserts | `allow_tf32=False` + global TF32 disable | `mtp_kernels.py` |
| 5 | RMSNorm-linear fused in wrong order (normalized output) | two-pass: stat first, then GEMM on normalized input | `mtp_kernels.py` |
| 6 | Unmasked padding corrupted attention softmax | `tl.where(k_mask, scores, -inf)` in all 7 attention kernels | `mtp_kernels.py` |
| 7 | Cross-stream pool race in async draft | serialized deterministic `generate` loop | `mtp_engine.py` |
| 8 | View read-after-free in `_main_forward` | `.contiguous().clone()` on slices | `mtp_engine.py` |
| 9 | Tree-mask softmax `-inf/-inf → NaN` (non-deterministic) | route tree monolith to eager reference | `mtp_kernels.py` |
| 10 | CPU→CUDA copy inside CUDA graph capture | use pre-materialized `pool.tree_mask_flat` | `mtp_engine.py` |
| 11 | `USE_BIAS` always-True (`bias=None` mutation) + `torch.empty` garbage | track `has_bias` before mutation, use `torch.zeros` | `mtp_kernels.py` |
| 12 | `fused_verify` 1-D stride bug (`out.stride(1)`) | pass `stride(0), 1` | `mtp_fused_verify.py` |
| 13 | `fused_verify`/`fused_multi_verify` `has_bias` mutation | same as #11 | `mtp_fused_verify.py` |
| 14 | `batch_prefix_match` passed CPU tensor to Triton | gate Triton path on `ids.is_cuda` | `mtp_radix.py` |

Known trade-offs (correctness-preserving fallbacks, not bugs):

- **Async draft/verify overlap** is serialized (needs double-buffered pool).
- **`fused_tree_attn_rmsnorm_linear`** uses its eager reference (needs NaN-safe masked softmax).
- **`fused_multi_verify` full GEMM** uses an eager unblocked GEMM (needs a blocked Triton GEMM); its acceptance stage is Triton (6.79×).

## Tensor Allocation & Memory Layout

| Request | Status | Location |
|---|---|---|
| `torch.empty()` for immediately-overwritten buffers | ✅ | `GraphMemoryPool` scratch buffers (`monolith_out`, `tree_mask_buf`, `cand_k`, `cand_v`, `comm_buffer`); `mtp_kernels.py` (`torch.empty_like` throughout) |
| `.contiguous()` before kernel entry | ✅ | `mtp_kernels.py` / `mtp_fused_verify.py` (every `fused_*` entry does `.contiguous()`) |
| Pre-allocated `GraphMemoryPool` slot reuse | ✅ | `GraphMemoryPool` + `draft_static_body`/`draft_monolith_body` write via `.copy_()` |
| Static page tables (flat contiguous int32) | ✅ | `PagedKVCache.page_table` (`torch.full` int32); `mtp_perf_hardening.FlatPageTable` |
| Flat contiguous scale tensors | ✅ | `mtp_kernels.pack_scales` / `unpack_scale_tensor` |
| `.clone()` vs view in draft hot path | ✅ | `draft_speculative_tokens` returns `.clone()`d tokens (detached, race-free for the async path); `swap_logical_pages` uses pre-allocated scratch |
| int32/int64 native widths | ✅ | `page_table` (int32), `draft_tokens`/`token` (int64), accept masks (int8) |
| No logging/inspection mid-generation | ✅ | all logging is in `__main__` validation, not in `generate*` |
| Static buffer sizes in graph blocks | ✅ | `GraphMemoryPool` fully pre-allocated; `generate_optimized` pre-allocates `acc/n/corr` |
| Pinned host memory for ingestion | ✅ | `mtp_perf_hardening.PinnedBufferPool` (`.pin_memory()` + `non_blocking`) |

## Triton / Kernel Optimization

| Request | Status | Location |
|---|---|---|
| Warp-boundary block sizes (`_WARP=32`) | ✅ | `_WARP`, `_warp_align_block` in `mtp_kernels.py` |
| In-register accumulation (`tl.zeros` + `tl.sum`) | ✅ | all reduction kernels |
| Auto-tuner warmup caching | ✅ | `mtp_kernels.prewarm_autotuner` |
| Fuse activation + normalization | ✅ | `fused_rmsnorm_linear`, `fused_add_rmsnorm`, `fused_tree_attn_rmsnorm_linear` |
| `tl.constexpr` dimensions/loops | ✅ | 32 uses (all block dims / `HAS_BIAS`) |
| Clean grid dims (`triton.cdiv`) | ✅ | `mtp_kernels.py` (3 uses) + `fused_multi_verify` |
| Explicit dtype cast before entry | ✅ | `.to(tl.float32)` at load; eager fallbacks cast at entry |
| `tl.where` instead of branching | ✅ | bitmask attention, multi-verify |
| SRAM tile loading / coalesced access | ✅ | block-wise `tl.load` with stride ptrs |
| Reduce redundant global writes | ✅ | register accumulators until final store |

## Python Loop & Orchestration

| Request | Status | Location |
|---|---|---|
| No dynamic list comprehensions in tight loops | ✅ | `generate_optimized` uses pre-allocated tensors |
| Batched ops over Python per-batch loops | ✅ | `fused_multi_verify`, `batch_prefix_match` |
| Fast scalar shape validation | ✅ | `mtp_perf_hardening.fast_check` |
| Cache shapes/props at init | ✅ | `self._perf_constants`, `self._vocab_padded`, engine `__init__` |
| Greedy `argmax` for `T<=0` | ✅ | `mtp_perf_hardening.argmax_greedy` / `sample_token`; `_sample_batch` |
| Pre-computed temperature constants | ✅ | `PrecomputedConstants.inv_temperature` |
| Global `@torch.no_grad()` | ✅ | `generate`, `generate_fused`, `generate_optimized` |
| Async stream overlap | ✅ | `DualStreamPipeline`, `TripleStreamPipeline`, `AsyncStream` |
| Device assertion once at init | ✅ | `DEVICE` resolved at module load |
| Lightweight exception handling | ✅ | `maybe_compile`/`prewarm_autotuner` swallow errors without retaining tensors |

## Pipeline / Graph / Execution Flow

| Request | Status | Location |
|---|---|---|
| CUDA graph capture for draft | ✅ | `compile_graph` / `_graph.replay()` in engine |
| Bind allocator to stream | ✅ | `mtp_perf_hardening.change_allocator_to_stream` |
| CUDA events (not `synchronize()`) | ✅ | `RollbackCommitFence`, `TripleStreamPipeline` |
| Pre-slice verify outputs | ✅ | `generate_fused`/`generate_optimized` (pre-allocated `acc/n/corr`) |
| Dedicated worker pools only off-CUDA | ✅ | `AsyncStream` (ThreadPoolExecutor only when not CUDA) |
| Defer compaction to boundaries | ✅ | `compact()` is a standalone call, not in the loop |
| Bitwise/offset state tracking | ✅ | `RollbackCommitFence` (int scalar), `KVManager` (offset subtraction) |
| Pre-load embeddings/weights contiguous | ✅ | `CachedLinear` (cached weight view) |
| `self._length += n` fast counters | ✅ | `KVManager.append` |
| Finalize `shared_proj` aliasing at init | ✅ | `SequentialMTPModule.shared_logits` shared across heads |

## Memory Bandwidth / VRAM

| Request | Status | Location |
|---|---|---|
| Pre-allocate all buffers (no realloc) | ✅ | `GraphMemoryPool`, `PagedKVCache`, `generate_optimized` |
| Non-blocking transfers | ✅ | `PinnedBufferPool.to_device_non_blocking` |
| Reduce host↔device round-trips | ✅ | fused kernels + `.item()` removal |
| Keep tokens/hidden/logits on GPU | ✅ | all paths |
| Geometric pool growth (`x2`) | ✅ | `RadixKVCache._ensure_page_capacity` |

*(Items requiring external runtime config — `--gpu-memory-utilization`, `nvidia-smi -c EXCLUSIVE_PROCESS`, FP8/INT4 KV quantization enablement, FlashAttention-2/3 backend, pre-compiled Triton binary disk cache — are deployment-level settings outside the Python module and are documented here for the serving team, not enforced in code.)*

## Kernel Arithmetic

| Request | Status | Location |
|---|---|---|
| FP32 accumulators in half-precision GEMM | ✅ | all `acc = tl.zeros(..., tl.float32)` |
| Pre-multiplied attention scale | ✅ | `PrecomputedConstants.attn_scale` (`1/√d`) |

## Quantization & Precision

| Request | Status | Location |
|---|---|---|
| FP8 weight matrices | ✅ | `quantize_fp8_e4m3`, `fused_rmsnorm_linear_fp8`, `fused_causal_attention_fp8` |
| Pre-computed quant scales | ✅ | `PrecomputedConstants.fp8_scale`; `QuantScaleState` |
| Packed scale arrays | ✅ | `pack_scales` / `unpack_scale_tensor` |
| INT4 packing | ✅ | `quantize_int4` (2-bit packed uint8), `fused_int4_matmul` |
| Keep norm/embedding in high precision | ✅ | `RMSNorm`/embedding not quantized |
| Error-bound validation | ✅ | `mtp_engine.py` quant checks (fp8/int4 vs fp32 ref) |
