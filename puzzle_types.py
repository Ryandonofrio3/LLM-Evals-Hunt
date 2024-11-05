from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class Puzzle:
    id: int
    title: str
    text: str
    image_path: Optional[str]
    answer: str

    def __post_init__(self):
        if self.id < 0:
            raise ValueError("ID must be positive")
        if not self.title.strip():
            raise ValueError("Title cannot be empty")
        if not self.text.strip():
            raise ValueError("Text cannot be empty")
        if not self.answer.strip():
            raise ValueError("Answer cannot be empty")
        if self.image_path and not os.path.exists(self.image_path):
            raise ValueError(f"Image file not found: {self.image_path}")

@dataclass
class ModelConfig:
    name: str
    provider: str
    system_prompt: str
    max_tokens: int = 300
    temperature: float = 0.7
    additional_params: dict = None 