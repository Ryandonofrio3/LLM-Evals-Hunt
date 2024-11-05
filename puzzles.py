from typing import Optional, List, Dict
import base64
from openai import OpenAI
from dotenv import load_dotenv
import os
import re
from enum import Enum
from puzzle_types import Puzzle, ModelConfig
from model_providers import ModelProvider, OpenAIProvider, AnthropicProvider

load_dotenv()

MASTER_SYSTEM_PROMPT = "You are a professional puzzle solver. You are given a puzzle and you need to solve it. The final answer will always be a SINGLE word. When you have a final answer output it surrounded by <answer> tags."


class Model(Enum):
    GPT4o_MINI = ModelConfig(name="gpt-4o-mini", provider="openai", system_prompt=MASTER_SYSTEM_PROMPT)
    GPT4_Turbo = ModelConfig(name="gpt-4-turbo", provider="openai", system_prompt=MASTER_SYSTEM_PROMPT)
    GPT4o = ModelConfig(name="gpt-4o", provider="openai", system_prompt=MASTER_SYSTEM_PROMPT)
    CLAUDE3 = ModelConfig(name="claude-3-opus-20240229", provider="anthropic", system_prompt=MASTER_SYSTEM_PROMPT)


class PuzzleSolver:
    def __init__(self):
        self.providers = {
            "openai": OpenAIProvider(),
            "anthropic": AnthropicProvider()
        }

    def solve_puzzle(self, puzzle: Puzzle, model: Model) -> Dict:
        try:
            model_config = model.value
            provider = self.providers[model_config.provider]
            response = provider.generate_response(puzzle, model_config)
            extracted_answer = self._extract_answer(response)
            is_correct = extracted_answer.lower() == puzzle.answer.lower()
            
            return {
                "puzzle_id": puzzle.id,
                "model": model_config.name,
                "provider": model_config.provider,
                "raw_response": response,
                "extracted_answer": extracted_answer,
                "correct_answer": puzzle.answer,
                "is_correct": is_correct,
                "error": None
            }
        except Exception as e:
            return {
                "puzzle_id": puzzle.id,
                "model": model_config.name,
                "provider": model_config.provider,
                "raw_response": None,
                "extracted_answer": None,
                "correct_answer": puzzle.answer,
                "is_correct": False,
                "error": str(e)
            }

    def _extract_answer(self, response: str) -> str:
        match = re.search(r'<answer>(.*?)</answer>', response)
        if match is None:
            raise ValueError("No answer found in response")
        return match.group(1)
