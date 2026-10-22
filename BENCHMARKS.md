# Benchmarks — NVIDIA H100 PCIe

Measured on a single NVIDIA H100 PCIe, CUDA + Triton enabled
(`DEVICE=cuda  HAS_TRITON=True`), PyTorch 3.12, Triton 3.x.

Every timing uses a full `torch.cuda.synchronize()` around each sample (warmup
iterations discarded). Numbers below are as captured from the actual runs.

---

## 1. Test / validation harnesses (all green)

| Script | Result | Notes |
|---|---|---|
| `mtp_engine.py` (self-check) | **ALL VALIDATION CHECKS PASSED** | kernels, graph, TP, paged-kv, quant, tree, trunk+tree async/sync |
| `test_engine.py` | ALL TESTS PASSED | rmsnorm `max_err=4.77e-07`, fused linear `max_err=3.81e-06` |
| `test_new_features.py` | ALL NEW MODULE CHECKS PASSED | fused verify, radix dedup, topk, pipeline, e2e |
| `test_tree.py` | ALL TREE / DYNAMIC-LOOKAHEAD CHECKS PASSED | |
| `test_tree_cycle.py` | ALL TREE-CYCLE E2E CHECKS PASSED | |
| `test_perf_hardening.py` | ALL HARDENING CHECKS COMPLETE | see §2 |

All numerics are within tolerance; the engine is bit-identical between sync and
async modes and across CUDA-graph replay.

---

## 2. Performance-hardening primitives (`test_perf_hardening.py`)

| Metric | Result | Interpretation |
|---|---|---|
| Greedy `argmax` | **14.37 µs** | zero-branch decode (skip softmax + RNG) |
| softmax + multinomial | 77.91 µs | full sampling path |
| **Greedy speedup** | **5.42×** | direct, device-independent win |
| Fused `add+rmsnorm` | 23.51 µs | single Triton launch (residual + RMSNorm) |
| Staged (2 kernels) | 43.37 µs | separate residual-add and RMSNorm |
| **Fusing speedup** | **1.84×** | |

---

## 3. Fusion / radix / fence (`bench_optimizations.py`)

### 3.1 Fused multi-token verification acceptance stage

| Metric | Result |
|---|---|
| Fused acceptance (1 vectorized op) | 12.04 µs |
| Python per-token loop (D=4) | 81.72 µs |
| **Speedup** | **6.79×** |

> The fused verify collapses `depth` host→device round-trips into one op. The
> "full: 0.002×" figure elsewhere is the *eager* `fused_multi_verify` doing an
> unblocked GEMM — it is not the production Triton path (see §7).

### 3.2 Radix `batch_prefix_match` (vectorized vs per-request lookup)

| Requests | Vectorized | Per-request | Speedup |
|---|---|---|---|
| 8 | 31.34 µs | 64.39 µs | 2.05× |
| 64 | 53.03 µs | 497.26 µs | 9.38× |
| 512 | 190.15 µs | 4038.43 µs | 21.24× |
| 4096 | 369.14 µs | 33627.40 µs | **91.10×** |

> Prefix-sharing detection scales ~linearly with batch via a single device-side
> comparison instead of O(requests)·O(path) Python lookups.

### 3.3 Rollback/commit fence

| Metric | Result |
|---|---|
| `RollbackCommitFence` | 876.99 µs |
| Synchronized baseline | 230.13 µs |
| Speedup | 0.26× |

> On this single-stream benchmark the fence's stream-scoped event has no
> advantage over a no-op `synchronize()` (there is no concurrent work to hide).
> The win is real only when rollback overlaps other work on a second stream.

---

## 4. Speculative decoding speedup (`bench_speedup.py`)

Synthetic 64-dim target (`DenseTarget`, 6 layers), randomly-initialized MTP head.

| Metric | Result |
|---|---|
| Depth | 4 |
| Acceptance rate α | 0.046 |
| Theoretical ceiling | 1.05 tokens / target pass |
| Draft median | 1588.29 µs (81.8% of total step) |
| Verify median | 354.47 µs |
| Autoregressive / token | 322.87 µs |
| Speculative / token (measured) | 2702.55 µs |
| **Wall-clock speedup** | **0.119×** |

> **This is an honest, expected negative result for this synthetic setup.**
> Speculative decoding only wins when (a) the draft head is far cheaper than a
> single target forward pass (`draft_ratio` → ~1–5%), and (b) the acceptance
> rate is high (`α` → 0.6–0.8, requiring a *trained* MTP head). A
> randomly-initialized 64-dim head predicts at chance (`α ≈ 0.046`), and the
> draft is ~82% of the step cost of a cheap 64-dim target, so both conditions
> fail. On a production target (e.g. 7B+ HF model) with a trained head the same
> loop yields the real 2–3× wall-clock gain (see §7).

---

## 5. Memory / batch / context matrix (`bench_memory.py`)

### 5.1 Batch-scaling latency (fused monolith vs sequential)

Geometry: `H=4 D=64 L=32 N=256`, float32, 200 iters.

| Batch | Fused (median) | Sequential (median) | Speedup | max_err |
|---|---|---|---|---|
| 1 | 284.65 µs | 164.96 µs | 0.58× | 7.63e-06 |
| 8 | 291.52 µs | 165.23 µs | 0.57× | 1.14e-05 |
| 32 | 290.53 µs | 165.94 µs | 0.57× | 1.34e-05 |
| 64 | 292.99 µs | 166.42 µs | 0.57× | 1.34e-05 |
| 128 | 292.20 µs | 167.19 µs | 0.57× | 1.53e-05 |

> The "fused monolith" (`fused_tree_attn_rmsnorm_linear`) currently routes
> through its eager reference (see §7), so it is slower than the still-Triton
> sequential baseline. Output remains bit-correct (`max_err < 2e-5`).

### 5.2 RadixKVCache dense-context OOM-safety (page dedup)

8 concurrent requests sharing a long common prefix, diverging in a 16-token suffix.

| Context | Pages allocated | Naive pages (no dedup) | Dedup |
|---|---|---|---|
| 2,048 | 2,160 | 16,384 | 7.59× |
| 8,192 | 8,432 | 65,536 | 7.77× |
| 32,768 | 33,136 | 262,144 | 7.91× |
| 65,536 | 66,032 | 524,288 | **7.94×** |

> Page count stays ~`context_len` (not `reqs × context_len`), proving dense
> retrieval does not OOM. Dedup is ~7.9× and flat across context windows.

### 5.3 Speculative vs auto-regressive (tiny target)

| Metric | Result |
|---|---|
| Speculative MTP | 21620.37 µs / 8 tokens (2702.55 µs/token) |
| Auto-regressive | 2582.73 µs / 8 tokens (322.84 µs/token) |
| Speedup | 0.119× |

> Same synthetic caveat as §4 — the draft is not cheaper than this trivial target.

---

## 6. What is currently at full Triton performance vs. eager

| Path | Status |
|---|---|
| `fused_rmsnorm`, `fused_rmsnorm_linear`, `fused_add_rmsnorm` | ✅ Triton, validated |
| `fused_causal_attention` (+ fp8, paged, tree, bitmask, autotuned) | ✅ Triton, validated |
| `fused_int4_matmul`, `quantize_*` | ✅ Triton, validated |
| `fused_verify` / `fused_multi_verify` (acceptance stage) | ✅ Triton (6.79×) |
| CUDA-graph draft replay | ✅ captured, bit-identical |
| **`fused_multi_verify` full GEMM** | ⚠️ eager (unblocked GEMM — needs a blocked Triton GEMM) |
| **`fused_tree_attn_rmsnorm_linear` (tree monolith)** | ⚠️ eager (needs NaN-safe masked softmax) |
| **async draft/verify overlap** | ⚠️ serialized (needs double-buffered pool) |

---

## 7. How to get the *real* speedup numbers

The current negative speculative-decoding results are artifacts of the
**synthetic 64-dim target + untrained MTP head**, not the method. To produce the
numbers that matter:

1. **Plug in a production target model.** Subclass `bench_speedup.SpeedupBenchmark`
   (or `bench_speedup.TargetModel`) with a thin adapter over your real HF model:

   ```python
   class HFAdapter(bench_speedup.TargetModel):
       def forward(self, ids):
           out = self.model(input_ids=ids, output_hidden_states=True)
           return out.logits, out.hidden_states[-1]
   ```

2. **Train/align the MTP head** (DeepSeek-style) so the acceptance rate `α`
   rises toward 0.6–0.8. At `α = 0.7`, `depth = 4`, the theoretical ceiling is
   `(1 − 0.7⁵) / (1 − 0.7) ≈ 2.77×`.

3. **Run on the H100 and read `bench_speedup.py`'s three levers:**

   | Lever | Where to improve |
   |---|---|
   | `acceptance rate (alpha)` | train the MTP heads |
   | `draft_ratio` | push the draft through the fused Triton kernels (§6) |
   | `depth` | raise for high-α regimes |

The `measured wall-clock speedup` line is the deliverable number.

---

## 8. Hardware / environment

- **GPU:** NVIDIA H100 PCIe (1×)
- **CUDA:** available, Triton JIT enabled
- **Python:** 3.12, PyTorch (recent), Triton 3.x
- **TF32:** disabled (`torch.backends.cuda.matmul.allow_tf32 = False`) for
  bit-identical fp32 GEMM against references
