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
"""Exercise real forward preprocessing with CPU-only model/parallel-state doubles."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch


def _load_forward(monkeypatch: pytest.MonkeyPatch, cp_size: int, cp_rank: int) -> ModuleType:
    root = Path(__file__).resolve().parents[2]
    mpu = ModuleType("megatron.core.parallel_state")
    mpu.get_context_parallel_world_size = lambda: cp_size
    mpu.get_context_parallel_rank = lambda: cp_rank
    mpu.get_tensor_model_parallel_world_size = lambda: 1
    mpu.get_context_parallel_group = lambda: None
    core = ModuleType("megatron.core")
    core.parallel_state = mpu
    packed = ModuleType("megatron.core.packed_seq_params")

    class PackedSeqParams:
        def __init__(self, **kwargs: Any) -> None:
            self.__dict__.update(kwargs)

    packed.PackedSeqParams = PackedSeqParams
    megatron_utils = ModuleType("verl.utils.megatron_utils")
    megatron_utils.unwrap_model = lambda model: model
    for name, module in {
        "megatron": ModuleType("megatron"),
        "megatron.core": core,
        "megatron.core.parallel_state": mpu,
        "megatron.core.packed_seq_params": packed,
        "verl.utils.megatron_utils": megatron_utils,
        "_sc_forward_test": ModuleType("_sc_forward_test"),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    for name in ("util", "model_forward"):
        qualified = f"_sc_forward_test.{name}"
        spec = importlib.util.spec_from_file_location(qualified, root / "verl/models/mcore" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, qualified, module)
        spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("data_format,cp_size", [("thd", 1), ("thd", 2), ("thd", 4), ("bshd", 1)])
@pytest.mark.parametrize("score_centering", [False, True])
def test_forward_aligns_sampler_heads_with_prediction_rows(
    monkeypatch: pytest.MonkeyPatch, data_format: str, cp_size: int, score_centering: bool
) -> None:
    rows = [torch.arange(1, 8), torch.arange(11, 16)]
    ids = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
    # Every head encodes its prediction row, independently of the shifted target label.
    heads = torch.nested.as_nested_tensor([torch.stack((row, row + 20), -1) for row in rows], layout=torch.jagged)
    log_q = torch.nested.as_nested_tensor([(-row[:, None] / 20).expand(-1, 2) for row in rows], layout=torch.jagged)
    temperatures = torch.nested.as_nested_tensor([torch.ones_like(row).float() for row in rows], layout=torch.jagged)
    seen = 0
    for rank in range(cp_size):
        forward = _load_forward(monkeypatch, cp_size, rank)
        # Communication is outside this CPU test: inspect the rank-local logits processor inputs.
        for name in ("postprocess_thd_engine", "postprocess_bshd_engine"):
            monkeypatch.setattr(forward, name, lambda output, *args, **kwargs: output)

        class Model:
            pre_process = True
            post_process = True
            config = SimpleNamespace(fp8=None, bf16=True, fp16=False)

            def __call__(self, **kwargs: Any) -> torch.Tensor:
                assert kwargs.get("fp32_output") is (False if score_centering else None)
                self.tokens = kwargs["input_ids"]
                return self.tokens.float().unsqueeze(-1)

        model = Model()

        def processor(
            logits: torch.Tensor, label: torch.Tensor, temperature: torch.Tensor, _model: Model = model, **extra: Any
        ) -> dict:
            nonlocal seen
            valid = _model.tokens.ne(0)
            assert logits.shape[:-1] == label.shape == temperature.shape
            if score_centering:
                topk = extra["rollout_topk_ids"]
                torch.testing.assert_close(topk[..., 0][valid], _model.tokens[valid])
                torch.testing.assert_close(topk[..., 1][valid], _model.tokens[valid] + 20)
                torch.testing.assert_close(extra["rollout_topk_log_probs"][..., 0][valid], -_model.tokens[valid] / 20)
            else:
                assert not extra
            # Final tokens have no next-token target; verify only effective prediction positions.
            predicts = valid.clone()
            for row in rows:
                predicts &= _model.tokens.ne(row[-1])
            torch.testing.assert_close(label[predicts], _model.tokens[predicts] + 1)
            seen += int(predicts.sum())
            return {"log_probs": logits.squeeze(-1)}

        args = {"label": ids, "temperature": temperatures}
        if score_centering:
            args.update(rollout_topk_ids=heads, rollout_topk_log_probs=log_q)
        forward.gptmodel_forward_model_engine(
            model,
            ids,
            {},
            logits_processor=processor,
            logits_processor_args=args,
            data_format=data_format,
            keep_logits_in_model_dtype=score_centering,
        )
    assert seen == sum(len(row) - 1 for row in rows)
