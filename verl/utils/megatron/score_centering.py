# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Top-k weighted score centering (arXiv:2609.20807).

Sampler probabilities are never renormalized over the transported head. The
unobserved tail is modeled as rho * p. Coefficients are detached; the center's
gradient flows through the trainer log probabilities only.
"""

from typing import Any

import torch
import torch.distributed as dist

from verl.trainer.ppo.score_centering import score_centering_residual, score_centering_weight_fn


class _SelectedLogits(torch.autograd.Function):
    """Project selected vocabulary rows without retaining [tokens, k, hidden].

    The hidden gradient is TP-local, like the fused CE kernel. The caller's
    sequence-parallel gather or copy-to-TP mapping reduces it in backward.
    """

    @staticmethod
    def forward(
        ctx: Any,
        hidden: torch.Tensor,
        weight: torch.Tensor,
        ids: torch.Tensor,
        temperature: float,
        group: Any,
        chunk_size: int,
    ) -> torch.Tensor:
        rank = dist.get_rank(group) if group is not None else 0
        flat = hidden.reshape(-1, hidden.shape[-1])
        local_ids = ids.long() - rank * weight.shape[0]
        owned = (local_ids >= 0) & (local_ids < weight.shape[0])
        local_ids = local_ids.clamp(0, weight.shape[0] - 1)
        dtype = torch.float64 if hidden.dtype == torch.float64 else torch.float32
        output = torch.empty(ids.shape, dtype=dtype, device=hidden.device)
        for start in range(0, flat.shape[0], chunk_size):
            end = start + chunk_size
            selected = weight[local_ids[start:end]].to(dtype)
            output[start:end] = (selected * flat[start:end, None, :].to(dtype)).sum(-1)
        output *= owned
        output /= temperature
        if group is not None and dist.get_world_size(group) > 1:
            dist.all_reduce(output, group=group)
        ctx.save_for_backward(hidden, weight, local_ids, owned)
        ctx.temperature, ctx.chunk_size = temperature, chunk_size
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad: torch.Tensor) -> tuple[Any, ...]:
        hidden, weight, ids, owned = ctx.saved_tensors
        flat = hidden.reshape(-1, hidden.shape[-1])
        dh = torch.empty_like(flat)
        # Repeated vocabulary IDs must accumulate in FP32, not model precision.
        dw = torch.zeros(weight.shape, dtype=grad.dtype, device=weight.device)
        for start in range(0, flat.shape[0], ctx.chunk_size):
            end = start + ctx.chunk_size
            g = grad[start:end] * owned[start:end] / ctx.temperature
            selected = weight[ids[start:end]].to(grad.dtype)
            dh[start:end] = (g[..., None] * selected).sum(1).to(hidden.dtype)
            updates = g[..., None] * flat[start:end, None, :].to(grad.dtype)
            dw.index_add_(0, ids[start:end].reshape(-1), updates.reshape(-1, weight.shape[-1]))
        return dh.reshape_as(hidden), dw.to(weight.dtype), None, None, None, None


def compute_fused_score_centering(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    sampled_log_probs: torch.Tensor,
    ids: torch.Tensor,
    log_q: torch.Tensor,
    temperature: float,
    correction: Any,
    group: Any = None,
    chunk_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Reuse fused CE's normalization: log p(k) = z_k - z_y + log p(y).

    Heads and labels must already follow the hidden states' local CP layout.
    Only selected projections are materialized; the full vocabulary softmax
    and its gradient remain in the existing fused linear CE kernel.
    """
    if chunk_size <= 0:
        raise ValueError("Score centering chunk_size must be positive")
    ids = ids.reshape(-1, ids.shape[-1])
    log_q = log_q.reshape_as(ids)
    selected_ids = torch.cat((ids, labels.reshape(-1, 1)), dim=-1)
    selected = _SelectedLogits.apply(hidden, weight, selected_ids, temperature, group, chunk_size)
    head = selected[:, :-1] - selected[:, -1:] + sampled_log_probs.reshape(-1, 1)
    weight_fn = score_centering_weight_fn(correction.get("rollout_is"), correction.get("rollout_is_threshold", 2.0))
    residual, q_mass, p_mass = score_centering_residual(head, log_q.to(head.dtype), weight_fn)
    return {
        "sc_correction": (residual * head).sum(-1).reshape_as(sampled_log_probs),
        "sc_sampler_head_mass": q_mass.reshape_as(sampled_log_probs),
        "sc_train_head_mass": p_mass.reshape_as(sampled_log_probs),
    }


class _VocabParallelPolicyOutputs(torch.autograd.Function):
    """Share one model-precision logits buffer across SC, logprob and entropy.

    Only token chunks are promoted to fp32. Backward recomputes probabilities
    and combines all output gradients before allocating the single logits grad.
    """

    @staticmethod
    def forward(
        ctx: Any,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ids: torch.Tensor,
        log_q: torch.Tensor,
        temperature: torch.Tensor,
        threshold: str | float | None,
        group: Any,
        chunk_size: int,
        entropy_enabled: bool,
    ) -> tuple[torch.Tensor, ...]:
        size = dist.get_world_size(group) if group is not None else 1
        rank = dist.get_rank(group) if group is not None else 0
        vocab = logits.shape[-1]
        flat = logits.reshape(-1, vocab)
        dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
        ids = ids.reshape(flat.shape[0], -1).long()
        log_q = log_q.reshape_as(ids)
        local_ids = ids - rank * vocab
        owned = (local_ids >= 0) & (local_ids < vocab)
        local_ids = local_ids.clamp(0, vocab - 1)
        local_labels = labels.reshape(-1).long() - rank * vocab
        label_owned = (local_labels >= 0) & (local_labels < vocab)
        local_labels = local_labels.clamp(0, vocab - 1)
        temperature = temperature.reshape(-1).to(dtype)
        log_z = torch.empty(flat.shape[0], dtype=dtype, device=logits.device)
        residual = torch.empty(ids.shape, dtype=dtype, device=logits.device)
        center, sampled, entropy, q_mass, p_mass = (torch.empty_like(log_z) for _ in range(5))
        weight_fn = score_centering_weight_fn(None if threshold is None else "token", threshold or 2.0)
        for start in range(0, flat.shape[0], chunk_size):
            end = start + chunk_size
            x = flat[start:end].to(dtype) / temperature[start:end, None]
            maximum = x.max(-1).values
            if size > 1:
                dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
            exp_x = (x - maximum[:, None]).exp()
            denominator = exp_x.sum(-1)
            if size > 1:
                dist.all_reduce(denominator, group=group)
            z = maximum + denominator.log()
            head = (x.gather(-1, local_ids[start:end]) - z[:, None]) * owned[start:end]
            selected = (x.gather(-1, local_labels[start:end, None]).squeeze(-1) - z) * label_owned[start:end]
            if size > 1:
                dist.all_reduce(head, group=group)
                dist.all_reduce(selected, group=group)
            b, q_mass[start:end], p_mass[start:end] = score_centering_residual(
                head,
                log_q[start:end].to(dtype),
                weight_fn,
            )
            residual[start:end], log_z[start:end] = b, z
            center[start:end], sampled[start:end] = (b * head).sum(-1), selected
            if entropy_enabled:
                mean_x = (exp_x / denominator[:, None] * x).sum(-1)
                if size > 1:
                    dist.all_reduce(mean_x, group=group)
                entropy[start:end] = z - mean_x
            else:
                entropy[start:end] = 0
        ctx.save_for_backward(
            logits, local_ids, owned, local_labels, label_owned, temperature, log_z, residual, entropy
        )
        ctx.chunk_size = chunk_size
        ctx.entropy_enabled = entropy_enabled
        outputs = tuple(value.reshape(logits.shape[:-1]) for value in (center, sampled, entropy, q_mass, p_mass))
        ctx.mark_non_differentiable(outputs[3], outputs[4])
        return outputs

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(
        ctx: Any,
        grad_center: torch.Tensor,
        grad_sampled: torch.Tensor,
        grad_entropy: torch.Tensor,
        grad_q_mass: torch.Tensor,
        grad_p_mass: torch.Tensor,
    ) -> tuple[Any, ...]:
        logits, ids, owned, labels, label_owned, temperature, log_z, residual, entropy = ctx.saved_tensors
        flat = logits.reshape(-1, logits.shape[-1])
        output = torch.empty_like(flat)
        gc, gs, gh = (value.reshape(-1) for value in (grad_center, grad_sampled, grad_entropy))
        for start in range(0, flat.shape[0], ctx.chunk_size):
            end = start + ctx.chunk_size
            log_p = flat[start:end].to(log_z.dtype) / temperature[start:end, None] - log_z[start:end, None]
            p = log_p.exp()
            b = residual[start:end] * gc[start:end, None]
            dx = -p * (b.sum(-1, keepdim=True) + gs[start:end, None])
            dx.scatter_add_(-1, ids[start:end], b * owned[start:end])
            dx.scatter_add_(-1, labels[start:end, None], (gs[start:end] * label_owned[start:end])[:, None])
            if ctx.entropy_enabled:
                dx -= p * (log_p + entropy[start:end, None]) * gh[start:end, None]
            output[start:end] = (dx / temperature[start:end, None]).to(logits.dtype)
        return output.reshape_as(logits), None, None, None, None, None, None, None, None


def compute_policy_outputs_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ids: torch.Tensor,
    log_q: torch.Tensor,
    temperature: torch.Tensor,
    correction: Any,
    group: Any = None,
    calculate_entropy: bool = False,
    chunk_size: int = 128,
) -> dict[str, torch.Tensor]:
    """Compute SC and the sampled term from the same fp32 chunk normalization."""
    if chunk_size <= 0:
        raise ValueError("Score centering chunk_size must be positive")
    if logits.shape[:-1] != ids.shape[:-1] or ids.shape != log_q.shape:
        raise ValueError("Score centering logits and sampler head shapes must align")
    if labels.shape != logits.shape[:-1] or temperature.shape != labels.shape:
        raise ValueError("Score centering labels and temperatures must align with logits")
    threshold = correction.get("rollout_is_threshold", 2.0) if correction.get("rollout_is") is not None else None
    center, sampled, entropy, q_mass, p_mass = _VocabParallelPolicyOutputs.apply(
        logits,
        labels,
        ids,
        log_q,
        temperature,
        threshold,
        group,
        chunk_size,
        calculate_entropy,
    )
    result = {
        "sc_correction": center,
        "log_probs": sampled,
        "sc_sampler_head_mass": q_mass,
        "sc_train_head_mass": p_mass,
    }
    if calculate_entropy:
        result["entropy"] = entropy
    return result
