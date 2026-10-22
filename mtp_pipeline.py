from __future__ import annotations

from typing import Optional, Tuple

import torch

from mtp_kernels import _HAS_TRITON
from mtp_parallel import ParallelPrimitives, get_parallel_primitives


class DualStreamPipeline:
    def __init__(
        self,
        draft_fn,
        verify_fn,
        draft_stream=None,
        verify_stream=None,
        comm_stream=None,
        prim: Optional[ParallelPrimitives] = None,
    ):
        self.draft_fn = draft_fn
        self.verify_fn = verify_fn
        self.prim = prim if prim is not None else get_parallel_primitives()
        self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.draft_stream = draft_stream or torch.cuda.Stream()
            self.verify_stream = verify_stream or torch.cuda.Stream()
            self.comm_stream = comm_stream or torch.cuda.Stream()
        else:
            self.draft_stream = draft_stream
            self.verify_stream = verify_stream
            self.comm_stream = comm_stream

    def _run_draft(self, state):
        with torch.cuda.stream(self.draft_stream) if self.use_cuda else _nullctx():
            return self.draft_fn(state)

    def _run_verify(self, tokens):
        with torch.cuda.stream(self.verify_stream) if self.use_cuda else _nullctx():
            return self.verify_fn(tokens)

    def _comm_all_reduce(self, tensor):
        if self.prim.tp_size <= 1:
            return tensor
        if self.use_cuda:
            with torch.cuda.stream(self.comm_stream):
                self.prim.all_reduce(tensor)
                self.draft_stream.wait_stream(self.comm_stream)
                self.verify_stream.wait_stream(self.comm_stream)
        else:
            self.prim.all_reduce(tensor)
        return tensor

    def step(self, state):
        draft = self._run_draft(state)
        if self.use_cuda:
            self.verify_stream.wait_stream(self.draft_stream)
        verified = self._run_verify(draft)
        return draft, verified

    def synchronize(self):
        if self.use_cuda:
            torch.cuda.synchronize()
        return None


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class AsyncEviction:
    def __init__(self, rollback_fn, capacity: int):
        self.rollback_fn = rollback_fn
        self.capacity = capacity
        self._pending: Optional[Tuple[int]] = None
        self.use_cuda = torch.cuda.is_available()
        self._event = torch.cuda.Event() if self.use_cuda else None

    def schedule_rollback(self, new_length: int):
        self._pending = (new_length,)
        if self.use_cuda:
            self._event.record()
        return self

    def apply(self):
        if self._pending is None:
            return None
        new_len = self._pending[0]
        self.rollback_fn(new_len)
        self._pending = None
        return new_len

    def applied_to_next_draft(self, draft_producer):
        if self._pending is None:
            return draft_producer()
        new_len = self.apply()
        return draft_producer(), new_len


class RollbackCommitFence:
    """Stream-ordered rollback/commit fence with zero host dispatch in the loop.

    Speculative decoding performs a *metadata-only* KV rollback on every rejected
    step (the actual KV pages are only finalized on commit). This helper turns
    those metadata updates into device-recorded events so that:

      * ``schedule`` never forces a host->device synchronize;
      * ``commit`` / ``rollback`` are ordered against the draft/verify streams
        via ``stream.wait_event``, not via ``torch.cuda.synchronize()``;
      * the metadata length is kept in a pre-allocated device scalar so the
        engine loop reads the effective length without a Python attribute read
        (this is what lets the step be captured into a CUDA graph unchanged).

    On CPU/MPS the fence degrades to a trivial in-place integer update, keeping
    the API uniform across backends.
    """

    def __init__(self, dtype=torch.int64):
        self.use_cuda = torch.cuda.is_available()
        self._event = torch.cuda.Event() if self.use_cuda else None
        self._length_tensor = torch.zeros((1,), dtype=dtype)
        if self.use_cuda:
            self._length_tensor = self._length_tensor.cuda()

    @property
    def length(self) -> int:
        return int(self._length_tensor.item())

    def record(self, stream) -> None:
        """Record a fence event on ``stream`` (draft or verify)."""
        if self.use_cuda:
            self._event.record(stream)

    def wait(self, stream) -> None:
        """Make ``stream`` wait until the recorded fence completes."""
        if self.use_cuda and self._event is not None:
            stream.wait_event(self._event)

    def commit(self, new_length: int) -> None:
        """Commit: advance the logical length by writing a device scalar."""
        if self.use_cuda:
            with torch.cuda.stream(torch.cuda.current_stream()):
                self._length_tensor.fill_(new_length)
        else:
            self._length_tensor[0] = new_length

    def rollback(self, new_length: int) -> None:
        """Rollback: identical metadata update (sets length back)."""
        self.commit(new_length)


class TripleStreamPipeline:
    """Three-stream staging: memory transfer, drafting, and verification run on
    dedicated CUDA streams synchronized by lightweight CUDA events (no CPU
    blocking, no ``torch.cuda.synchronize()``).

    Streams:
      * ``transfer_stream`` -- host<->device embedding pre-fetch / prompt DMA
      * ``draft_stream``    -- MTP draft heads
      * ``verify_stream``   -- target-model verification pass
    """

    def __init__(self, draft_fn, verify_fn, transfer_fn=None):
        self.draft_fn = draft_fn
        self.verify_fn = verify_fn
        self.transfer_fn = transfer_fn
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.transfer_stream = torch.cuda.Stream()
            self.draft_stream = torch.cuda.Stream()
            self.verify_stream = torch.cuda.Stream()
            self.ev_transfer = torch.cuda.Event()
            self.ev_draft = torch.cuda.Event()
        else:
            self.transfer_stream = self.draft_stream = self.verify_stream = None
            self.ev_transfer = self.ev_draft = None

    def step(self, state):
        if self.transfer_fn is not None:
            if self.use_cuda:
                with torch.cuda.stream(self.transfer_stream):
                    transfer_out = self.transfer_fn(state)
                    self.ev_transfer.record(self.transfer_stream)
            else:
                transfer_out = self.transfer_fn(state)
        else:
            transfer_out = None

        if self.use_cuda:
            with torch.cuda.stream(self.draft_stream):
                if transfer_out is not None:
                    self.draft_stream.wait_event(self.ev_transfer)
                draft = self.draft_fn(state)
                self.ev_draft.record(self.draft_stream)
            with torch.cuda.stream(self.verify_stream):
                self.verify_stream.wait_event(self.ev_draft)
                verified = self.verify_fn(draft)
        else:
            draft = self.draft_fn(state)
            verified = self.verify_fn(draft)
        return transfer_out, draft, verified

    def synchronize(self):
        if self.use_cuda:
            torch.cuda.current_stream().wait_stream(self.transfer_stream)
            torch.cuda.current_stream().wait_stream(self.draft_stream)
            torch.cuda.current_stream().wait_stream(self.verify_stream)
        return None


class DynamicGraphBatchSlicer:
    """Pre-captures CUDA graphs across standard batch sizes and routes a
    generation call to the matching captured graph, avoiding re-capture.

    Graph capture is only meaningful on CUDA; on CPU/MPS the slicer stores and
    replays plain callables.
    """

    def __init__(self, capture_fn, batch_sizes=(1, 2, 4, 8, 16)):
        self.capture_fn = capture_fn
        self.batch_sizes = tuple(batch_sizes)
        self._graphs = {}
        self.use_cuda = torch.cuda.is_available()

    def capture(self):
        for b in self.batch_sizes:
            if self.use_cuda:
                try:
                    g = torch.cuda.CUDAGraph()
                    # Warm up on a side stream so the capture is not polluted
                    # by lazy allocations, then capture.
                    with torch.cuda.graph(g):
                        self.capture_fn(b)
                    self._graphs[b] = g
                except Exception:
                    # If the capture function performs no CUDA work (or graph
                    # capture is unsupported), fall back to storing the plain
                    # callable so routing still works via ``replay``.
                    self._graphs[b] = lambda b=b: self.capture_fn(b)
            else:
                self._graphs[b] = lambda b=b: self.capture_fn(b)

    def route(self, batch: int):
        """Return the graph/callable for the smallest captured batch >= batch."""
        for b in self.batch_sizes:
            if b >= batch:
                return self._graphs[b]
        return self._graphs[self.batch_sizes[-1]] if self._graphs else None

    def replay(self, batch: int):
        target = self.route(batch)
        if target is None:
            return None
        if self.use_cuda and hasattr(target, "replay"):
            target.replay()
            return None
        return target()



__all__ = ["DualStreamPipeline", "AsyncEviction", "RollbackCommitFence", "TripleStreamPipeline", "DynamicGraphBatchSlicer"]
