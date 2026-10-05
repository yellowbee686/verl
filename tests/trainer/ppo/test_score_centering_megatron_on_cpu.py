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
"""Behavioral gradient, tensor-parallel and configuration checks for Megatron SC."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf

from verl.trainer.ppo.score_centering import (
    score_centering_correction,
    score_centering_ppo_loss,
    score_centering_residual,
    score_centering_weight_fn,
)
from verl.utils.config import _validate_score_centering_config
from verl.utils.megatron.score_centering import (
    compute_policy_outputs_from_logits,
)


def effective_mass(log_p: torch.Tensor, log_q: torch.Tensor, threshold: str | float | None) -> torch.Tensor:
    """Return q*f(p/q), including verl's sampled-ratio numerical safety bounds."""
    if threshold is None:
        return log_q.exp()
    from verl.trainer.ppo.rollout_corr_helper import SAFETY_BOUND, _parse_rollout_is_threshold

    upper, lower = _parse_rollout_is_threshold(threshold)
    ratio = (log_p - log_q).clamp(-SAFETY_BOUND, SAFETY_BOUND).exp()
    weights = ratio.clamp_max(upper) if lower is None else torch.where((ratio >= lower) & (ratio <= upper), ratio, 0.0)
    return log_q.exp() * weights


def topk_residual(log_p: torch.Tensor, log_q: torch.Tensor, threshold: str | float | None) -> torch.Tensor:
    """Compute detached head coefficients after analytically canceling the tail."""
    weight_fn = score_centering_weight_fn(None if threshold is None else "token", threshold or 2.0)
    return score_centering_residual(log_p, log_q, weight_fn)[0]


def topk_center(log_p: torch.Tensor, log_q: torch.Tensor, threshold: str | float | None) -> torch.Tensor:
    """Dense reference on trainer logprobs gathered at the sampler's top-k IDs."""
    return (topk_residual(log_p, log_q, threshold) * log_p).sum(-1)


def vocab_parallel_center(
    logits: torch.Tensor,
    ids: torch.Tensor,
    log_q: torch.Tensor,
    threshold: str | float | None = None,
    group: object = None,
    chunk_size: int = 128,
) -> torch.Tensor:
    return compute_policy_outputs_from_logits(
        logits,
        torch.zeros(logits.shape[:-1], dtype=torch.long),
        ids,
        log_q,
        torch.ones(logits.shape[:-1], dtype=logits.dtype),
        {"rollout_is": None if threshold is None else "token", "rollout_is_threshold": threshold},
        group,
        chunk_size=chunk_size,
    )["sc_correction"]


@pytest.mark.parametrize("threshold", [None, 2.0, "0.5_2.0"])
@pytest.mark.parametrize("chunk", [1, 3, 128])
def test_chunked_center_matches_modeled_full_vocab_gradient(threshold: str | float | None, chunk: int) -> None:
    torch.manual_seed(23)
    x = torch.randn(5, 11, dtype=torch.float64, requires_grad=True)
    q = torch.randn_like(x).softmax(-1)
    ids = q.topk(3).indices
    log_p = x.log_softmax(-1)
    head_q = q.gather(-1, ids)
    tail_mask = torch.ones_like(q, dtype=torch.bool).scatter_(-1, ids, False)
    rho = q.masked_fill(~tail_mask, 0).sum(-1, keepdim=True) / log_p.exp().detach().masked_fill(~tail_mask, 0).sum(
        -1, keepdim=True
    )
    modeled_q = torch.where(tail_mask, rho * log_p.exp().detach(), q)
    full = (effective_mass(log_p.detach(), modeled_q.log(), threshold).detach() * log_p).sum(-1)
    actual = vocab_parallel_center(x, ids, head_q.log(), threshold, chunk_size=chunk)
    reference = topk_center(log_p.gather(-1, ids), head_q.log(), threshold)
    torch.testing.assert_close(actual, reference)
    weights = torch.linspace(-1, 2, 5)
    actual_grad = torch.autograd.grad((actual * weights).sum(), x, retain_graph=True)[0]
    full_grad = torch.autograd.grad((full * weights).sum(), x)[0]
    torch.testing.assert_close(actual_grad, full_grad, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("threshold", [None, 2.0, "0.5_2.0"])
def test_constant_advantage_has_zero_expected_gradient(threshold: str | float | None) -> None:
    x = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64).log().requires_grad_()
    q = torch.tensor([0.1, 0.4, 0.5], dtype=torch.float64)
    log_p = x.log_softmax(-1)
    mass = effective_mass(log_p.detach(), q.log(), threshold)
    center = (mass * log_p).sum()
    expected_loss = -(mass * log_p - q * center).sum()
    torch.testing.assert_close(torch.autograd.grad(expected_loss, x)[0], torch.zeros_like(x), atol=1e-12, rtol=0)


def test_mis_zero_sample_weight_keeps_center_gradient() -> None:
    x = torch.tensor([0.6, 0.3, 0.1], dtype=torch.float64).log().requires_grad_()
    q = torch.tensor([0.1, 0.4, 0.5], dtype=torch.float64)
    mass = effective_mass(x.log_softmax(-1).detach(), q.log(), "0.5_2.0")
    assert mass[0] == 0
    center = (mass * x.log_softmax(-1)).sum()
    assert torch.autograd.grad(center, x)[0].norm() > 0.01


def config() -> DictConfig:
    cfg = OmegaConf.create(
        {
            "trainer": {"use_v1": True},
            "algorithm": {
                "rollout_correction": {
                    "bypass_mode": True,
                    "loss_type": "reinforce",
                    "rollout_is": "token",
                    "rollout_is_threshold": 2.0,
                    "rollout_is_batch_normalize": False,
                    "rollout_rs": None,
                    "score_centering": True,
                }
            },
            "actor_rollout_ref": {
                "actor": {"strategy": "megatron", "policy_loss": {"loss_mode": "bypass_mode"}},
                "model": {"use_fused_kernels": False},
                "rollout": {
                    "name": "vllm",
                    "topk_log_probs": 32,
                    "calculate_log_probs": True,
                    "logprobs_mode": "processed_logprobs",
                },
            },
        }
    )

    cfg.actor_rollout_ref.actor.policy_loss.rollout_correction = deepcopy(cfg.algorithm.rollout_correction)
    return cfg


def test_config_validation_is_read_only_and_disabled_is_unchanged() -> None:
    cfg = config()
    _validate_score_centering_config(cfg)
    assert cfg.actor_rollout_ref.rollout.topk_log_probs == 32
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == "bypass_mode"
    assert cfg.actor_rollout_ref.actor.policy_loss.rollout_correction == cfg.algorithm.rollout_correction
    cfg.algorithm.rollout_correction.score_centering = False
    cfg.actor_rollout_ref.actor.policy_loss.rollout_correction.score_centering = False
    before = deepcopy(cfg)
    _validate_score_centering_config(cfg)
    assert cfg == before


@pytest.mark.parametrize(
    "path,value",
    [
        ("algorithm.rollout_correction.bypass_mode", False),
        ("algorithm.rollout_correction.loss_type", "ppo_clip"),
        ("algorithm.rollout_correction.rollout_is", "sequence"),
        ("algorithm.rollout_correction.rollout_rs", "seq_mean_k3"),
        ("algorithm.rollout_correction.rollout_is_batch_normalize", True),
        ("distillation.enabled", True),
    ],
)
def test_unsupported_configuration_fails_before_training(path: str, value: object) -> None:
    cfg = config()
    OmegaConf.update(cfg, path, value)
    with pytest.raises(ValueError):
        _validate_score_centering_config(cfg)


def test_fused_megatron_score_centering_configuration() -> None:
    cfg = config()
    cfg.actor_rollout_ref.model.use_fused_kernels = True
    cfg.actor_rollout_ref.actor.use_fused_kernels = True
    _validate_score_centering_config(cfg)


@pytest.mark.parametrize("threshold", [None, 2.0, "0.5_2.0"])
def test_worker_loss_matches_weighted_score_gradient(threshold: str | float | None) -> None:
    from tensordict import TensorDict

    from verl.trainer.config.algorithm import RolloutCorrectionConfig
    from verl.utils import tensordict_utils as tu
    from verl.workers.config.actor import ActorConfig, PolicyLossConfig

    torch.manual_seed(14)
    logits = torch.randn(5, 7, dtype=torch.float64, requires_grad=True)
    log_p = logits.log_softmax(-1)
    q = torch.randn_like(logits).softmax(-1)
    labels = torch.tensor([2, 3, 4, 5, 0])
    ids = torch.arange(7).expand(5, -1)
    center = vocab_parallel_center(logits, ids, q.log(), threshold, chunk_size=2)
    sampled = log_p.gather(-1, labels[:, None]).squeeze(-1)
    sampled_q = q.gather(-1, labels[:, None]).squeeze(-1)
    advantages = torch.tensor([[1.5, -2.0, 0.7]], dtype=torch.float64)
    mask = torch.tensor([[1, 0, 1]])
    data = TensorDict(
        {
            "prompts": torch.tensor([[1, 2]]),
            "responses": torch.tensor([[3, 4, 5]]),
            "attention_mask": torch.ones(1, 5, dtype=torch.long),
            "response_mask": mask,
            "old_log_probs": sampled_q[1:4].log().unsqueeze(0),
            "advantages": advantages,
        },
        batch_size=1,
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=2, global_batch_size=1)
    correction = RolloutCorrectionConfig(
        bypass_mode=True,
        loss_type="reinforce",
        rollout_is=None if threshold is None else "token",
        rollout_is_threshold=threshold if threshold is not None else 2.0,
        score_centering=True,
    )
    actor = ActorConfig(
        strategy="megatron",
        rollout_n=1,
        ppo_micro_batch_size_per_gpu=1,
        policy_loss=PolicyLossConfig(loss_mode="bypass_mode", rollout_correction=correction),
    )
    loss, metrics = score_centering_ppo_loss(
        actor,
        {
            "log_probs": sampled,
            "sc_correction": center,
            "sc_sampler_head_mass": q.sum(-1),
            "sc_train_head_mass": log_p.exp().sum(-1).detach(),
        },
        data,
    )
    assert "actor/sc_correction" in metrics
    mass = effective_mass(log_p.detach(), q.log(), threshold)
    weights = mass.gather(-1, labels[:, None]).squeeze(-1) / sampled_q
    expected = (
        -(advantages.squeeze(0) * mask.squeeze(0) * (weights[1:4] * sampled[1:4] - (mass * log_p).sum(-1)[1:4])).sum()
        / 2
    )
    torch.testing.assert_close(
        torch.autograd.grad(loss, logits, retain_graph=True)[0],
        torch.autograd.grad(expected, logits)[0],
        atol=1e-12,
        rtol=1e-12,
    )


def _tp_worker(rank: int, rendezvous: str) -> None:
    import torch.distributed as dist

    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        torch.manual_seed(29)
        full_logits = torch.randn(5, 12, dtype=torch.float64, requires_grad=True)
        q = torch.randn_like(full_logits).softmax(-1)
        ids = q.topk(4).indices
        lq = q.gather(-1, ids).log()
        shard = full_logits.detach()[:, rank * 6 : (rank + 1) * 6].clone().requires_grad_()
        for threshold in (None, 2.0, "0.5_2.0"):
            actual = vocab_parallel_center(shard, ids, lq, threshold, dist.group.WORLD, chunk_size=2)
            reference = topk_center(full_logits.log_softmax(-1).gather(-1, ids), lq, threshold)
            torch.testing.assert_close(actual, reference, atol=1e-12, rtol=1e-12)
            local_grad = torch.autograd.grad(actual.sum(), shard)[0]
            full_grad = torch.autograd.grad(reference.sum(), full_logits)[0]
            torch.testing.assert_close(local_grad, full_grad[:, rank * 6 : (rank + 1) * 6], atol=1e-12, rtol=1e-12)
            _check_joint_outputs(full_logits, q, threshold, shard=shard, group=dist.group.WORLD, rank=rank)
    finally:
        dist.destroy_process_group()


def _check_joint_outputs(
    logits: torch.Tensor,
    q: torch.Tensor,
    threshold: str | float | None,
    shard: torch.Tensor | None = None,
    group: object = None,
    rank: int = 0,
    entropy: bool = True,
) -> None:
    ids = q.topk(4).indices
    log_q = q.gather(-1, ids).log()
    temperature = torch.linspace(0.6, 1.4, logits.shape[0], dtype=torch.float64)
    labels = torch.arange(logits.shape[0]) * 2
    correction = {
        "rollout_is": None if threshold is None else "token",
        "rollout_is_threshold": threshold,
        "score_centering": True,
    }
    local = logits if shard is None else shard
    outputs = compute_policy_outputs_from_logits(
        local,
        labels,
        ids,
        log_q,
        temperature,
        correction,
        group,
        calculate_entropy=entropy,
        chunk_size=2,
    )
    reference_logits = logits.detach().to(torch.float64 if logits.dtype == torch.float64 else torch.float32)
    reference_logits.requires_grad_()
    lp = (reference_logits / temperature.to(reference_logits.dtype)[:, None]).log_softmax(-1)
    expected = {
        "sc_correction": topk_center(lp.gather(-1, ids), log_q.to(lp.dtype), threshold),
        "log_probs": lp.gather(-1, labels[:, None]).squeeze(-1),
    }
    if entropy:
        expected["entropy"] = -(lp.exp() * lp).sum(-1)
    actual_loss, expected_loss = 0, 0
    tolerance = 1e-12 if logits.dtype == torch.float64 else 1e-5
    for i, key in enumerate(expected):
        torch.testing.assert_close(outputs[key], expected[key], atol=tolerance, rtol=tolerance)
        weights = torch.linspace(-1, 2 + i, logits.shape[0], dtype=outputs[key].dtype)
        actual_loss += (outputs[key] * weights).sum()
        expected_loss += (expected[key] * weights).sum()
    actual_grad = torch.autograd.grad(actual_loss, local)[0]
    expected_grad = torch.autograd.grad(expected_loss, reference_logits)[0]
    if shard is not None:
        expected_grad = expected_grad[:, rank * shard.shape[-1] : (rank + 1) * shard.shape[-1]]
    torch.testing.assert_close(actual_grad, expected_grad.to(local.dtype), atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("threshold", [None, 2.0, "0.5_2.0"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
@pytest.mark.parametrize("entropy", [True, False])
def test_joint_outputs_match_dense_policy_gradient(
    threshold: str | float | None,
    dtype: torch.dtype,
    entropy: bool,
) -> None:
    torch.manual_seed(75)
    logits = torch.randn(5, 12, dtype=dtype, requires_grad=True)
    q = torch.randn(5, 12, dtype=torch.float64).softmax(-1)
    _check_joint_outputs(logits, q, threshold, entropy=entropy)


def test_two_real_cpu_tp_ranks_match_unsharded_gradient(tmp_path: Path) -> None:
    torch.multiprocessing.spawn(_tp_worker, args=(str(tmp_path / "tp_init"),), nprocs=2, join=True)


def test_hydra_config_survives_worker_materialization() -> None:
    from hydra import compose, initialize_config_dir

    from verl.utils.config import omega_conf_to_dataclass

    config_dir = Path(__file__).resolve().parents[3] / "verl/trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(
            config_name="ppo_trainer",
            overrides=[
                "model_engine=megatron",
                "algorithm.rollout_correction.bypass_mode=true",
                "algorithm.rollout_correction.loss_type=reinforce",
                "algorithm.rollout_correction.rollout_is=token",
                "algorithm.rollout_correction.score_centering=true",
                "actor_rollout_ref.rollout.topk_log_probs=128",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.calculate_log_probs=true",
                "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
                "actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode",
                "++actor_rollout_ref.actor.policy_loss.rollout_correction=${algorithm.rollout_correction}",
            ],
        )
    _validate_score_centering_config(cfg)
    actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    x = torch.tensor([[1.0, 0.0, -1.0]], requires_grad=True)
    ids = torch.tensor([[0, 1]])
    log_q = torch.tensor([[0.4, 0.2]]).log()
    result = score_centering_ppo_loss(
        actor,
        student_logits=x,
        score_centering_inputs=(ids, log_q),
        score_centering_labels=torch.tensor([0]),
        score_centering_temperature=torch.ones(1),
    )
    expected = topk_center(x.log_softmax(-1).gather(-1, ids), log_q, 2.0)
    torch.testing.assert_close(result["sc_correction"], expected)
    result["sc_correction"].sum().backward()
    assert x.grad.isfinite().all()


@pytest.mark.parametrize("threshold", [None, 2.0, "0.5_2.0"])
def test_megatron_matches_fsdp_contract_with_int32_heads(threshold: str | float | None) -> None:
    torch.manual_seed(93)
    logits = torch.randn(2, 3, 11, requires_grad=True)
    original = logits.detach().clone()
    sampler = torch.randn_like(logits).log_softmax(-1)
    ids = sampler.topk(4).indices.to(torch.int32)
    head_q = sampler.gather(-1, ids.long())
    temperatures = torch.tensor([[0.7, 1.1, 1.4], [0.9, 1.0, 1.2]])
    labels = torch.tensor([[1, 5, 7], [0, 3, 10]], dtype=torch.int32)
    output = compute_policy_outputs_from_logits(
        logits,
        labels,
        ids,
        head_q,
        temperatures,
        {"rollout_is": None if threshold is None else "token", "rollout_is_threshold": threshold},
        calculate_entropy=True,
        chunk_size=2,
    )
    log_p = (logits / temperatures.unsqueeze(-1)).log_softmax(-1)
    expected, q_mass, p_mass = score_centering_correction(
        log_p.gather(-1, ids.long()),
        head_q,
        score_centering_weight_fn(None if threshold is None else "token", threshold or 2.0),
    )
    for key, ref in (("sc_correction", expected), ("sc_sampler_head_mass", q_mass), ("sc_train_head_mass", p_mass)):
        torch.testing.assert_close(output[key], ref, atol=1e-6, rtol=1e-5)
    assert not output["sc_sampler_head_mass"].requires_grad
    assert not output["sc_train_head_mass"].requires_grad
    gradient = torch.autograd.grad(output["sc_correction"].sum(), logits, retain_graph=True)[0]
    reference_gradient = torch.autograd.grad(expected.sum(), logits)[0]
    torch.testing.assert_close(gradient, reference_gradient, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(logits, original)


@pytest.mark.parametrize(
    "path,value",
    [
        ("trainer.use_v1", False),
        ("actor_rollout_ref.actor.policy_loss.loss_mode", "gspo"),
        ("actor_rollout_ref.actor.policy_loss.rollout_correction.rollout_rs", "seq_mean_k3"),
        ("actor_rollout_ref.actor.calculate_sum_pi_squared", True),
        ("actor_rollout_ref.actor.megatron.dynamic_context_parallel", True),
        ("actor_rollout_ref.actor.megatron.pad_to_length", True),
        ("actor_rollout_ref.model.mtp.enable", True),
        ("actor_rollout_ref.rollout.engine_kwargs.vllm.speculative_config", {"method": "ngram"}),
        ("actor_rollout_ref.rollout.top_p", 0.95),
        ("actor_rollout_ref.rollout.top_k", 50),
        ("actor_rollout_ref.actor.megatron.score_centering_chunk_size", 0),
    ],
)
def test_megatron_rejects_incompatible_explicit_config(path: str, value: object) -> None:
    cfg = config()
    OmegaConf.update(cfg, path, value)
    before = deepcopy(cfg)
    with pytest.raises(ValueError):
        _validate_score_centering_config(cfg)
    assert cfg == before


def test_bshd_context_parallelism_keeps_the_existing_engine_restriction() -> None:
    cfg = config()
    cfg.actor_rollout_ref.actor.megatron = {"use_remove_padding": False, "context_parallel_size": 2}
    with pytest.raises(ValueError, match="BSHD"):
        _validate_score_centering_config(cfg)
    cfg.actor_rollout_ref.actor.megatron.context_parallel_size = 1
    _validate_score_centering_config(cfg)
