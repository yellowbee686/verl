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
"""Fused SC algebra and gradients; CUDA cases exercise the real CE kernel."""

from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist

from verl.trainer.ppo.score_centering import score_centering_residual, score_centering_weight_fn
from verl.utils.megatron.score_centering import _SelectedLogits, compute_fused_score_centering


def _check_fused_sc(threshold: str | float | None, chunk: int, dtype: torch.dtype, cuda: bool = False) -> None:
    torch.manual_seed(87)
    device = "cuda" if cuda else "cpu"
    hidden = (torch.randn(7, 1, 128, device=device, dtype=dtype) * 0.1).requires_grad_()
    weight = (torch.randn(256, 128, device=device, dtype=dtype) * 0.1).requires_grad_()
    labels = torch.tensor([255, 1, 200, 14, 9, 6, 7], device=device)
    q = torch.randn(7, 256, device=device).softmax(-1)
    ids = q.topk(4).indices.to(torch.int32)
    log_q = q.gather(-1, ids.long()).log()
    # Cover an empty prompt/padding head and repeated placeholder IDs.
    log_q[0] = -torch.inf
    ids[1, 2:] = 0
    log_q[1, 2:] = -torch.inf
    temperature = 0.7
    calc_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    reference_hidden = hidden.detach().to(calc_dtype).requires_grad_()
    reference_weight = weight.detach().to(calc_dtype).requires_grad_()
    reference_lp = (reference_hidden.reshape(7, 128) @ reference_weight.T / temperature).log_softmax(-1)
    if cuda:
        from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

        sampled, entropy = linear_cross_entropy(hidden, weight, labels, temperature, "none", None)
    else:
        lp = (hidden.reshape(7, 128).to(calc_dtype) @ weight.to(calc_dtype).T / temperature).log_softmax(-1)
        sampled = lp.gather(-1, labels[:, None]).squeeze(-1)
        entropy = -(lp.exp() * lp).sum(-1)
    correction = {"rollout_is": None if threshold is None else "token", "rollout_is_threshold": threshold}
    actual = compute_fused_score_centering(
        hidden, weight, labels, sampled, ids, log_q, temperature, correction, chunk_size=chunk
    )
    reference_head = reference_lp.gather(-1, ids.long())
    residual, q_mass, p_mass = score_centering_residual(
        reference_head, log_q.to(calc_dtype), score_centering_weight_fn(correction["rollout_is"], threshold)
    )
    reference_center = (residual * reference_head).sum(-1)
    tolerance = 2e-3 if cuda else 2e-6
    torch.testing.assert_close(actual["sc_correction"], reference_center, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual["sc_sampler_head_mass"], q_mass, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(actual["sc_train_head_mass"], p_mass, atol=tolerance, rtol=tolerance)
    advantage = torch.linspace(-1, 1, 7, device=device, dtype=calc_dtype)
    sampled_q = q.gather(-1, labels[:, None]).squeeze(-1)
    sampled_ref = reference_lp.gather(-1, labels[:, None]).squeeze(-1)
    sample_weight = score_centering_weight_fn(correction["rollout_is"], threshold)(
        sampled_ref.detach().exp() / sampled_q
    )
    actual_loss = (advantage * (-sample_weight * sampled + actual["sc_correction"]) - 0.03 * entropy).sum()
    expected_loss = (
        advantage * (-sample_weight * sampled_ref + reference_center)
        + 0.03 * (reference_lp.exp() * reference_lp).sum(-1)
    ).sum()
    actual_grads = torch.autograd.grad(actual_loss, (hidden, weight))
    expected_grads = torch.autograd.grad(expected_loss, (reference_hidden, reference_weight))
    for got, expected in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(
            got.float(),
            expected.float(),
            atol=3e-3 if dtype == torch.bfloat16 else 2e-6,
            rtol=3e-2 if dtype == torch.bfloat16 else 2e-5,
        )


@pytest.mark.parametrize("threshold", [None, 2.0, "0.5_2.0"])
@pytest.mark.parametrize("chunk", [1, 3, 128])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_fused_sc_matches_dense_hidden_and_weight_gradients(
    threshold: str | float | None, chunk: int, dtype: torch.dtype
) -> None:
    _check_fused_sc(threshold, chunk, dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires the CUDA fused linear CE kernel")
@pytest.mark.parametrize("threshold", [None, 2.0, "0.5_2.0"])
def test_cuda_fused_sc_matches_dense(threshold: str | float | None) -> None:
    _check_fused_sc(threshold, 3, torch.bfloat16, cuda=True)


class _TPSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor) -> torch.Tensor:
        result = value.clone()
        dist.all_reduce(result)
        return result

    @staticmethod
    def backward(ctx: Any, grad: torch.Tensor) -> torch.Tensor:
        return grad


def _tp_projection_worker(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(8)
        hidden = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
        full_weight = torch.randn(12, 3, dtype=torch.float64, requires_grad=True)
        weight = full_weight.detach()[rank * 6 : (rank + 1) * 6].clone().requires_grad_()
        ids = torch.tensor([[0, 6, 6], [11, 4, 0], [5, 5, 10], [7, 9, 3], [1, 8, 11]])
        actual = _SelectedLogits.apply(hidden, weight, ids, 0.7, dist.group.WORLD, 2)
        expected = (hidden @ full_weight.T / 0.7).gather(-1, ids)
        torch.testing.assert_close(actual, expected)
        scale = torch.randn_like(actual)
        dh, dw = torch.autograd.grad((actual * scale).sum(), (hidden, weight))
        reference_dh, reference_dw = torch.autograd.grad((expected * scale).sum(), (hidden, full_weight))
        dist.all_reduce(dh)
        torch.testing.assert_close(dh, reference_dh)
        torch.testing.assert_close(dw, reference_dw[rank * 6 : (rank + 1) * 6])
        labels = torch.tensor([0, 11, 5, 7, 8])
        q = torch.randn(5, 12, dtype=torch.float64).softmax(-1)
        ids = q.topk(3).indices
        log_q = q.gather(-1, ids).log()
        for threshold in (None, 2.0, "0.5_2.0"):
            local_logits = hidden @ weight.T / 0.7
            local_labels = labels - rank * 6
            owned = (local_labels >= 0) & (local_labels < 6)
            selected = local_logits.gather(-1, local_labels.clamp(0, 5)[:, None]).squeeze(-1) * owned
            # CPU equivalent of CE's global normalizer with TP-local backward.
            sampled = _TPSum.apply(selected) - _TPSum.apply(local_logits.exp().sum(-1)).log()
            correction = {"rollout_is": None if threshold is None else "token", "rollout_is_threshold": threshold}
            actual = compute_fused_score_centering(
                hidden, weight, labels, sampled, ids, log_q, 0.7, correction, dist.group.WORLD, 2
            )
            lp = (hidden @ full_weight.T / 0.7).log_softmax(-1)
            head = lp.gather(-1, ids)
            b, _, _ = score_centering_residual(
                head, log_q, score_centering_weight_fn(correction["rollout_is"], threshold)
            )
            center = (b * head).sum(-1)
            torch.testing.assert_close(actual["sc_correction"], center)
            dh, dw = torch.autograd.grad((actual["sc_correction"] - sampled).sum(), (hidden, weight))
            ref_dh, ref_dw = torch.autograd.grad(
                (center - lp.gather(-1, labels[:, None]).squeeze(-1)).sum(), (hidden, full_weight)
            )
            dist.all_reduce(dh)
            torch.testing.assert_close(dh, ref_dh)
            torch.testing.assert_close(dw, ref_dw[rank * 6 : (rank + 1) * 6])
    finally:
        dist.destroy_process_group()


def test_tp_sparse_projection_gradients(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(_tp_projection_worker, args=(str(tmp_path / "tp"),), nprocs=2, join=True)
