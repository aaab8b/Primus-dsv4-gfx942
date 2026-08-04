###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Context-parallel support for the DeepSeek-V4 dense / SWA attention branch.

V4 attention materialises the query tensor at FULL head width on every tensor-parallel rank
(``deepseek_v4_attention.py`` sets ``num_attention_heads_per_partition = num_heads`` with no
``divide()``), so TP shards weights but not attention activations. Context parallelism is the
only lever that shrinks the per-rank sequence, and therefore the only way past the long-context
memory wall: measured on 8x MI308X, an unmodified 4-layer V4-Flash needs ~396 GB at 128k
against a 192 GiB card, while 32k fits in 145 GB.

For the dense (``compress_ratio == 0``) branch the CP contract is small, because that branch is
already index-driven -- causality and the sliding window live entirely in the index matrix the
sparse-MLA adapter builds, not in the kernel. A CP rank therefore needs exactly two things:

  1. the ``d_window`` post-RoPE KV rows immediately left of its shard, so a query near the shard
     start can still see its full window (:class:`LeftBoundaryExchange`), and
  2. its global row offset, so the adapter can validate the window against global positions
     while indexing into the local ``[boundary ++ local]`` buffer
     (``cp_dwindow`` / ``cp_global_start`` in ``v4_sparse_mla_adapter``).

Exchanging post-RoPE KV rather than pre-projection hidden states is deliberate: the neighbour
has already applied RoPE with the correct global positions, and it moves ``d_window`` rows
instead of a full hidden block.

Ported from NVIDIA/Megatron-LM PR #5087 (`csa_cp_utils.py`), whose CP path is THD-only; this
is the BSHD-shaped equivalent for Primus's V4 attention.
"""

from typing import Optional

import torch
import torch.distributed as dist


class LeftBoundaryExchange(torch.autograd.Function):
    """Receive the previous CP rank's trailing ``d_window`` rows; scatter grads back.

    Forward is one batched isend/irecv step around the CP ring. Backward returns the boundary
    gradient to the rank that actually owns those rows, so they accumulate where the parameters
    that produced them live.
    """

    @staticmethod
    def forward(ctx, tensor: torch.Tensor, d_window: int, cp_group):
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        ctx.cp_group = cp_group
        ctx.d_window = d_window
        ctx.input_shape = tensor.shape
        if tensor.shape[0] < d_window:
            raise RuntimeError(
                "DeepSeek-V4 CP boundary exchange needs local rows >= d_window: "
                f"local_rows={tensor.shape[0]}, d_window={d_window}. Reduce "
                "context_parallel_size or the sliding window."
            )
        boundary = tensor.new_zeros((d_window,) + tuple(tensor.shape[1:]))

        ops = []
        if cp_rank > 0:
            ops.append(
                dist.P2POp(dist.irecv, boundary, dist.get_global_rank(cp_group, cp_rank - 1), cp_group)
            )
        if cp_rank + 1 < cp_size:
            send_tail = tensor[-d_window:].contiguous()
            ops.append(
                dist.P2POp(dist.isend, send_tail, dist.get_global_rank(cp_group, cp_rank + 1), cp_group)
            )
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        return boundary

    @staticmethod
    def backward(ctx, grad_boundary: torch.Tensor):
        cp_group = ctx.cp_group
        cp_size = cp_group.size()
        cp_rank = cp_group.rank()
        grad_input = grad_boundary.new_zeros(ctx.input_shape)

        ops = []
        recv_grad = None
        if cp_rank > 0:
            ops.append(
                dist.P2POp(
                    dist.isend, grad_boundary.contiguous(),
                    dist.get_global_rank(cp_group, cp_rank - 1), cp_group,
                )
            )
        if cp_rank + 1 < cp_size:
            recv_grad = grad_boundary.new_empty(grad_boundary.shape)
            ops.append(
                dist.P2POp(dist.irecv, recv_grad, dist.get_global_rank(cp_group, cp_rank + 1), cp_group)
            )
        if ops:
            for req in dist.batch_isend_irecv(ops):
                req.wait()
        if recv_grad is not None:
            grad_input[-ctx.d_window :] = recv_grad
        return grad_input, None, None


def get_cp_group():
    """The context-parallel process group, or None when CP is off / torch.distributed is not up."""
    if not dist.is_available() or not dist.is_initialized():
        return None
    try:
        from megatron.core import parallel_state
    except ImportError:
        return None
    try:
        group = parallel_state.get_context_parallel_group()
    except (AssertionError, RuntimeError):
        return None
    if group is None or group.size() <= 1:
        return None
    return group


def exchange_boundary_kv(kv_bshd: torch.Tensor, d_window: int, cp_group) -> torch.Tensor:
    """Boundary KV for a ``[B, S, 1, head_dim]`` post-RoPE latent.

    Returns ``[B, d_window, 1, head_dim]``. Rank 0 gets zeros, which the adapter's
    global-position validity mask then excludes -- no separate special case is needed.
    """
    B, S, G, Dh = kv_bshd.shape
    if B != 1:
        raise RuntimeError(f"DeepSeek-V4 CP currently assumes micro_batch_size=1, got B={B}.")
    flat = kv_bshd.reshape(S, G * Dh)
    boundary = LeftBoundaryExchange.apply(flat, int(d_window), cp_group)
    return boundary.reshape(1, int(d_window), G, Dh)


__all__ = ["LeftBoundaryExchange", "get_cp_group", "exchange_boundary_kv"]
