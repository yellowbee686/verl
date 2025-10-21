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

import asyncio
import logging
import os
from abc import ABC, abstractmethod

from omegaconf import DictConfig
from transformers import AutoTokenizer

from verl import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class RewardLoopManagerBase(ABC):
    _class_initialized = False

    def __init__(self, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize agent loop.

        Args:
            config (DictConfig): YAML config.
            tokenizer (AutoTokenizer): Tokenizer for tokenize messages.
        """
        self.config = config
        self.tokenizer = tokenizer
        self.loop = asyncio.get_running_loop()
        self.init_class(config, tokenizer)

    @classmethod
    def init_class(cls, config: DictConfig, tokenizer: AutoTokenizer):
        """Initialize class state shared across all instances."""
        if cls._class_initialized:
            return
        cls._class_initialized = True

    @abstractmethod
    async def run_single(self, data: DataProto):
        raise NotImplementedError
