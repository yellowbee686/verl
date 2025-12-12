# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import copy
import logging
import os
import time
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@register("single_turn_agent")
class SingleTurnAgentLoop(AgentLoopBase):
    """Naive agent loop that only do single turn chat completion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length
        self.apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        debug_enabled = os.getenv("VERL_ROLLOUT_DEBUG", "0").lower() in ("1", "true", "yes", "y", "on")
        debug_verbose = os.getenv("VERL_ROLLOUT_DEBUG_VERBOSE", "0").lower() in ("1", "true", "yes", "y", "on")

        t_start = time.perf_counter()
        messages = list(kwargs["raw_prompt"])
        image_data = copy.deepcopy((kwargs.get("multi_modal_data") or {}).get("image", None))
        t_after_copy = time.perf_counter()

        metrics = {}
        request_id = uuid4().hex

        t_template_done = t_after_copy
        t_processor_done = t_after_copy

        # Use processor if available for multimodal support
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            t_template_done = time.perf_counter()
            model_inputs = self.processor(text=[raw_prompt], images=image_data, return_tensors="pt")
            prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
            t_processor_done = time.perf_counter()
        else:
            prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=True, **self.apply_chat_template_kwargs
                ),
            )
            t_template_done = time.perf_counter()
            t_processor_done = t_template_done

        t_before_vllm = time.perf_counter()
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=request_id, prompt_ids=prompt_ids, sampling_params=sampling_params, image_data=image_data
            )
        t_after_vllm = time.perf_counter()

        if debug_enabled:
            if image_data is None:
                num_images = 0
            elif isinstance(image_data, list):
                num_images = len(image_data)
            else:
                num_images = 1

            t_copy_ms = (t_after_copy - t_start) * 1000.0
            t_template_ms = (t_template_done - t_after_copy) * 1000.0
            t_processor_ms = (t_processor_done - t_template_done) * 1000.0
            t_vllm_ms = (t_after_vllm - t_before_vllm) * 1000.0
            t_total_ms = (t_after_vllm - t_start) * 1000.0

            if debug_verbose:
                logger.info(
                    "[rollout_debug][single_turn] request_id=%s prompt_tokens=%s images=%s "
                    "t_copy_ms=%.1f t_template_ms=%.1f t_processor_ms=%.1f t_vllm_ms=%.1f t_total_ms=%.1f "
                    "sampling=%s",
                    request_id,
                    len(prompt_ids),
                    num_images,
                    t_copy_ms,
                    t_template_ms,
                    t_processor_ms,
                    t_vllm_ms,
                    t_total_ms,
                    {k: sampling_params.get(k) for k in ("temperature", "top_p", "top_k", "logprobs")},
                )
            else:
                logger.info(
                    "[rollout_debug][single_turn] request_id=%s prompt_tokens=%s images=%s "
                    "t_prepare_ms=%.1f t_vllm_ms=%.1f t_total_ms=%.1f",
                    request_id,
                    len(prompt_ids),
                    num_images,
                    (t_before_vllm - t_start) * 1000.0,
                    t_vllm_ms,
                    t_total_ms,
                )

        response_mask = [1] * len(output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            routed_experts=(
                output.routed_experts[: len(prompt_ids) + self.response_length]
                if output.routed_experts is not None
                else None
            ),
            multi_modal_data={"image": image_data} if image_data is not None else {},
            num_turns=2,
            metrics=metrics,
        )
        return output
