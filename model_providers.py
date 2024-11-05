from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import base64
from openai import OpenAI
from anthropic import Anthropic
import os
from puzzle_types import Puzzle, ModelConfig

class ModelProvider(ABC):
    @abstractmethod
    def generate_response(self, puzzle: Puzzle, model_config: ModelConfig) -> str:
        pass

class OpenAIProvider(ModelProvider):
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate_response(self, puzzle: Puzzle, model_config: ModelConfig) -> str:
        messages = [{"role": "system", "content": model_config.system_prompt}]
        
        if puzzle.image_path:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": puzzle.text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._encode_image(puzzle.image_path)}"
                        }
                    }
                ]
            })
        else:
            messages.append({"role": "user", "content": puzzle.text})

        response = self.client.chat.completions.create(
            model=model_config.name,
            messages=messages,
            max_tokens=model_config.max_tokens,
            **model_config.additional_params or {}
        )
        print(f"Response for {model_config.name}: {response.choices[0].message.content}")
        return response.choices[0].message.content

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

class AnthropicProvider(ModelProvider):
    def __init__(self, api_key: str = None):
        self.client = Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate_response(self, puzzle: Puzzle, model_config: ModelConfig) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": puzzle.text
                    }
                ]
            }
        ]

        if puzzle.image_path:
            messages[0]["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": self._encode_image(puzzle.image_path)
                }
            })

        response = self.client.messages.create(
            model=model_config.name,
            messages=messages,
            max_tokens=model_config.max_tokens,
            **model_config.additional_params or {}
        )
        return response.content[0].text

    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8") 