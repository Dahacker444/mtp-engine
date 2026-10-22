"""Correct wall-clock speedup measurement for the MTP speculative engine.

The naive benchmark (``bench_memory.py``) times "speculative vs autoregressive"
but never reports the two quantities that actually *determine* the speedup:

  1. Acceptance rate (alpha)        -- how many draft tokens pass verification
  2. Draft-cost ratio               -- draft latency / one target forward pass

This script instruments the engine loop to recover both, then reports:

  * measured wall-clock speedup     (accepteance-time over autoregressive time)
  * theoretical maximum (Leviathan) = (1 - alpha^{depth+1}) / (1 - alpha)
  * acceptance rate, mean accepted tokens / step, draft-cost fraction

Methodology note
----------------
Speculative decoding only wins when the draft head is MUCH cheaper than a
single target forward pass (typical production ratio is 1-10%). On a tiny
synthetic target the draft/verify overhead dominates and you will measure <1x.
The correct way to get a real number is to plug in a production target model
(e.g. a 7B+ huggingface model) where the MTP head is ~1% of the target cost.

To use a real model, subclass the ``TargetModel`` protocol below and pass it in:
    engine = SpeedupBenchmark(depth=4, target=my_hf_adapter())
"""

from __future__ import annotations

import math
import statistics
import time
from typing import Callable, List, Optional, Tuple

import torch

from mtp_engine import (
    DEVICE,
    KVManager,
    SequentialMTPModule,
    SpeculativeDecodingEngine,
    DraftState,
)


def _sync() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


class TargetModel(torch.nn.Module):
    """Protocol for the target (main) model.

    ``forward(ids)`` must return ``(logits, hidden)`` where:
      * ``logits``  shape ``(B, S, vocab)``
      * ``hidden``  shape ``(B, S, hidden_dim)`` (last-layer hidden state)

    The engine derives the MTP hidden_dim from ``hidden``'s last dimension, so
    the draft head's ``hidden_dim`` must match the target's hidden width.
    """

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class SpeedupBenchmark:
    def __init__(
        self,
        hidden_dim: int,
        vocab_size: int,
        depth: int,
        target: TargetModel,
        temperature: float = 0.0,
        draft_mode: str = "trunk",
    ):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.depth = depth
        self.target = target
        self.temperature = temperature
        self.draft_mode = draft_mode

        self.mtp = SequentialMTPModule(
            hidden_dim=hidden_dim, vocab_size=vocab_size, depth=depth
        ).to(DEVICE)
        self.engine = SpeculativeDecodingEngine(
            self.mtp,
            target,
            KVManager(max_seq_len=4096),
            temperature=temperature,
            depth=depth,
            async_mode=False,
            use_cuda_stream=False,
            draft_mode=draft_mode,
        )

        # ---- instrumentation accumulators ----
        self._accepted_counts: List[int] = []
        self._draft_secs: List[float] = []
        self._verify_secs: List[float] = []

    # ------------------------------------------------------------------ #
    # Instrumented generate: mirrors engine.generate but records per-step
    # acceptance and phase timings.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate_instrumented(
        self, input_ids: torch.Tensor, max_new_tokens: int
    ) -> torch.Tensor:
        temp = self.temperature
        input_ids = input_ids.to(DEVICE)
        self.engine.kv._length = input_ids.shape[1]

        _, base_hidden, base_logits = self._target_forward(input_ids)
        state = DraftState(base_hidden, base_logits, temp)
        cur_ids = input_ids
        generated: List[torch.Tensor] = []

        while sum(t.shape[1] for t in generated) < max_new_tokens:
            # ---- draft phase (cheap) ----
            _sync()
            t0 = time.perf_counter()
            draft_logits, draft_tokens = self.engine.draft_speculative_tokens(state)
            _sync()
            self._draft_secs.append(time.perf_counter() - t0)

            batch, depth = draft_tokens.shape
            draft_seq = torch.cat([cur_ids[:, -1:], draft_tokens], dim=1)

            # ---- verify phase (one target forward over depth+1 tokens) ----
            _sync()
            t1 = time.perf_counter()
            verified_logits, verified_hidden, verified_last_logits = self._target_forward(draft_seq)
            main_tokens = self._sample(verified_logits)
            _sync()
            self._verify_secs.append(time.perf_counter() - t1)

            accepted_n, corrected = self.engine._verify(draft_tokens, main_tokens, depth)
            self._accepted_counts.append(accepted_n)

            if corrected:
                new_tokens = torch.cat(
                    [draft_tokens[:, :accepted_n],
                     main_tokens[:, accepted_n + 1: accepted_n + 2]], dim=1,
                )
            else:
                new_tokens = draft_tokens

            if new_tokens.shape[1] == 0:
                break
            generated.append(new_tokens)
            cur_ids = torch.cat([cur_ids, new_tokens], dim=1)
            state = DraftState(verified_hidden, verified_last_logits, temp)

        return torch.cat(generated, dim=1)

    # ------------------------------------------------------------------ #
    # helpers (mirror engine internals without coupling to privates)
    # ------------------------------------------------------------------ #
    def _target_forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, hidden = self.target(ids)
        if logits.dim() == 2:
            logits = logits.unsqueeze(1)
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        return logits, hidden[:, -1, :], logits[:, -1, :]

    def _sample(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature <= 0.0:
            return logits.argmax(dim=-1)
        flat = logits.reshape(-1, logits.shape[-1])
        probs = torch.nn.functional.softmax(flat / self.temperature, dim=-1)
        sampled = torch.multinomial(probs, 1).squeeze(-1)
        return sampled.reshape(*logits.shape[:-1])

    # ------------------------------------------------------------------ #
    # Single-token autoregressive baseline (same target)
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def autoregressive(
        self, input_ids: torch.Tensor, max_new_tokens: int
    ) -> Tuple[torch.Tensor, float]:
        ids = input_ids.to(DEVICE)
        _sync()
        t0 = time.perf_counter()
        for _ in range(max_new_tokens):
            logits, _ = self.target(ids)
            nxt = logits[:, -1].argmax(dim=-1)
            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
        _sync()
        elapsed = time.perf_counter() - t0
        return ids, elapsed

    # ------------------------------------------------------------------ #
    # Report
    # ------------------------------------------------------------------ #
    def report(self, prompt_len: int, max_new_tokens: int, n_ar_trials: int = 5) -> None:
        if not self._accepted_counts:
            print("no instrumented steps recorded -- run generate_instrumented first")
            return

        n_steps = len(self._accepted_counts)
        alpha = statistics.fmean(c / self.depth for c in self._accepted_counts)
        mean_accepted = statistics.fmean(self._accepted_counts)
        draft_med = statistics.median(self._draft_secs)
        verify_med = statistics.median(self._verify_secs)
        draft_ratio = draft_med / (draft_med + verify_med)

        # Theoretical ceiling (Leviathan et al., token-by-token acceptance)
        d = self.depth
        alpha_full = mean_accepted / d if d > 1 else alpha
        if abs(alpha_full - 1.0) < 1e-6:
            theo = float(d + 1)
        else:
            theo = (1.0 - alpha_full ** (d + 1)) / (1.0 - alpha_full)

        spec_per_token = statistics.fmean(self._draft_secs) / max(mean_accepted, 1) \
            + statistics.fmean(self._verify_secs) / max(mean_accepted, 1)

        print("\n[acceptance]")
        print(f"  depth={self.depth}  steps={n_steps}  mean_accepted={mean_accepted:.2f}")
        print(f"  acceptance rate (alpha) = {alpha:.3f}")
        print(f"  theoretical ceiling      = {theo:.2f} tokens / target pass")

        print("\n[phase timing]")
        print(f"  draft  median = {draft_med * 1e6:10.2f}us  ({draft_ratio * 100:.1f}% of total)")
        print(f"  verify median = {verify_med * 1e6:10.2f}us")

        # Autoregressive baseline
        prompt = torch.randint(0, self.vocab_size, (1, prompt_len), device=DEVICE)
        ar_times = []
        for _ in range(n_ar_trials):
            _, elapsed = self.autoregressive(prompt, max_new_tokens)
            ar_times.append(elapsed)
        ar_per_token = statistics.median(ar_times) / max_new_tokens

        print("\n[speedup]")
        print(f"  autoregressive per-token = {ar_per_token * 1e6:10.2f}us")
        print(f"  speculative  per-token (measured) = {spec_per_token * 1e6:10.2f}us")
        if spec_per_token > 0:
            print(f"  measured wall-clock speedup = {ar_per_token / spec_per_token:.3f}x")
        print(f"  (theoretical max {theo:.2f}x is only approached when "
              f"draft_ratio -> 0 and alpha -> 1)")


# --------------------------------------------------------------------------- #
# Minimal demo target (replace this with a real HF model adapter to get real
# numbers; see module docstring).
# --------------------------------------------------------------------------- #
class DenseTarget(TargetModel):
    def __init__(self, hidden_dim: int, vocab_size: int, n_layers: int = 6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embed = torch.nn.Embedding(vocab_size, hidden_dim)
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Sequential(
                    torch.nn.Linear(hidden_dim, hidden_dim * 4),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim * 4, hidden_dim),
                )
                for _ in range(n_layers)
            ]
        )
        self.proj = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(ids)
        for layer in self.layers:
            h = h + layer(h)
        return self.proj(h), h


def main() -> None:
    torch.manual_seed(0)
    hidden_dim, vocab_size, depth = 64, 256, 4
    target = DenseTarget(hidden_dim, vocab_size, n_layers=6).to(DEVICE)

    bench = SpeedupBenchmark(
        hidden_dim=hidden_dim, vocab_size=vocab_size, depth=depth,
        target=target, temperature=0.0, draft_mode="trunk",
    )

    prompt = torch.randint(0, vocab_size, (1, 8), device=DEVICE)
    bench.generate_instrumented(prompt, max_new_tokens=32)
    bench.report(prompt_len=8, max_new_tokens=32)


if __name__ == "__main__":
    main()
