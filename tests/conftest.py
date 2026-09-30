"""Test doubles: a deterministic embedder and a scripted LLM. No network, no model download."""

import hashlib
import re

import numpy as np
import pytest

from app.knowledge import KnowledgeBase
from app.llm import Message, Reply, ToolCall


class HashEmbedder:
    """Bag-of-words hashed into a normalized vector: shared words give high cosine similarity."""

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), 256))
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                out[i, int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


class ScriptedLLM:
    """Plays back a fixed list of tool calls, one assistant turn each, and records what it saw."""

    def __init__(self, turns: list[list[ToolCall]]):
        self.turns, self.seen = list(turns), []

    async def chat(self, system, messages, tools) -> Reply:
        self.seen.append(list(messages))
        return Reply(Message("assistant", None, self.turns.pop(0)), {"input_tokens": 10, "output_tokens": 5})


@pytest.fixture(scope="session")
def kb() -> KnowledgeBase:
    return KnowledgeBase.from_file(HashEmbedder())
