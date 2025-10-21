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

# tokenizer.apply_chat_template is not working properly for gpt-oss model.
# Because the chat template requires tool call messages to parse tool response messages
# so we need to format the tool response manually.
def format_gpt_oss_tool_response_manually(tool_response: str, tool_call_name: str) -> str:
    """Format tool response for gpt-oss model.
    Args:
        tool_response: Tool response string
        tool_call_name: Name of the tool that was called

    Returns:
        Formatted tool response string
    """
    return f"<|start|>functions.{tool_call_name} to=assistant<|channel|>commentary<|message|>{tool_response}<|end|>"


def add_generation_prompt_for_gpt_oss(message_content: str) -> str:
    """Add generation prompt for gpt-oss model.
    Args:
        message_content: Message content string

    Returns:
        Message content string with generation prompt
    """
    return message_content + "<|start|>assistant"
