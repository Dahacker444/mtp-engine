from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ParallelPrimitives:
    def __init__(self, group: Optional[object] = None):
        self._group = group
        self._dist = self._init_dist()

    @staticmethod
    def _init_dist():
        if not torch.distributed.is_available():
            return None
        if not torch.distributed.is_initialized():
            return None
        return torch.distributed

    @property
    def dist(self):
        return self._dist

    @property
    def initialized(self) -> bool:
        return self._dist is not None

    @property
    def world_size(self) -> int:
        if self.initialized:
            return int(self._dist.get_world_size(self._group or None))
        return 1

    @property
    def rank(self) -> int:
        if self.initialized:
            return int(self._dist.get_rank(self._group or None))
        return 0

    @property
    def tp_size(self) -> int:
        return self.world_size

    def all_reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.tp_size <= 1 or not self.initialized:
            return x
        self._dist.all_reduce(x, group=self._group)
        return x

    def all_gather(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        if self.tp_size <= 1 or not self.initialized:
            return x
        gathered = [torch.empty_like(x) for _ in range(self.tp_size)]
        self._dist.all_gather(gathered, x, group=self._group)
        return torch.cat(gathered, dim=dim)

    def scatter(self, x: torch.Tensor, dim: int, shard_size: int) -> torch.Tensor:
        if self.tp_size <= 1:
            return x
        chunks = torch.chunk(x, self.tp_size, dim=dim)
        return chunks[self.rank].contiguous()


_PRIMITIVES: Optional[ParallelPrimitives] = None


def get_parallel_primitives(group: Optional[object] = None) -> ParallelPrimitives:
    global _PRIMITIVES
    if _PRIMITIVES is None:
        _PRIMITIVES = ParallelPrimitives(group)
    return _PRIMITIVES


def _reset_parallel_primitives() -> None:
    global _PRIMITIVES
    _PRIMITIVES = None


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        prim: Optional[ParallelPrimitives] = None,
    ):
        super().__init__()
        self.prim = prim if prim is not None else get_parallel_primitives()
        tp = self.prim.tp_size
        assert out_features % tp == 0
        self.in_features = in_features
        self.out_features = out_features
        self.out_features_per_rank = out_features // tp
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.empty(self.out_features_per_rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.out_features_per_rank))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output:
            return self.prim.all_gather(y, dim=-1)
        return self.prim.all_reduce(y)


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        gather_output: bool = False,
        prim: Optional[ParallelPrimitives] = None,
    ):
        super().__init__()
        self.prim = prim if prim is not None else get_parallel_primitives()
        tp = self.prim.tp_size
        assert in_features % tp == 0
        self.in_features = in_features
        self.in_features_per_rank = in_features // tp
        self.out_features = out_features
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.empty(out_features, self.in_features_per_rank))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features_per_rank)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_shard = self.prim.scatter(x, dim=-1, shard_size=self.in_features_per_rank)
        y = F.linear(x_shard, self.weight, self.bias)
        if self.gather_output:
            return self.prim.all_gather(y, dim=-1)
        return y


@dataclass
class GraphSegment:
    graph: object
    collective: Optional[str]
    collective_dim: int = -1


class SegmentedGraphRunner:
    _fence_buffers = {}

    def __init__(self, prim: ParallelPrimitives):
        self.prim = prim
        self.segments = []

    def add_segment(self, segment: GraphSegment) -> None:
        self.segments.append(segment)

    def replay(self) -> None:
        for segment in self.segments:
            if segment.graph is not None and hasattr(segment.graph, "replay"):
                segment.graph.replay()
            elif callable(segment.graph):
                segment.graph()
            if segment.collective == "all_reduce":
                if self.prim.tp_size > 1:
                    buf = self._fence_buffers.get(id(segment))
                    if buf is not None:
                        self.prim.all_reduce(buf)
            elif segment.collective == "all_gather":
                buf = self._fence_buffers.get(id(segment))
                if buf is not None:
                    gathered = self.prim.all_gather(buf, dim=segment.collective_dim)
                    self._fence_buffers[id(segment)] = gathered

    def bind_fence_buffer(self, segment: GraphSegment, tensor: torch.Tensor) -> None:
        SegmentedGraphRunner._fence_buffers[id(segment)] = tensor


__all__ = [
    "ParallelPrimitives",
    "get_parallel_primitives",
    "RowParallelLinear",
    "ColumnParallelLinear",
    "GraphSegment",
    "SegmentedGraphRunner",
]
