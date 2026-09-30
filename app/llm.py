"""One tiny, provider-neutral chat interface with tool calling. Two adapters.

The agent only knows `LLM.chat(system, messages, tools) -> Reply`. Swapping
OpenAI <-> Bedrock is a config change, which is how you do fallbacks,
cost/latency A/B tests, and data-residency requirements without touching
agent code.
"""

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import boto3
from botocore.config import Config
from openai import AsyncOpenAI
from pydantic import BaseModel


@dataclass
class Tool:
    name: str
    description: str
    args: type[BaseModel]  # Pydantic model = JSON schema for the LLM + validation for us

    @property
    def schema(self) -> dict:
        return self.args.model_json_schema()


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    raw: Any = None  # provider-native assistant turn, replayed verbatim (keeps reasoning blocks intact)


@dataclass
class Reply:
    message: Message
    usage: dict = field(default_factory=dict)


class LLM(Protocol):
    async def chat(self, system: str, messages: list[Message], tools: list[Tool]) -> Reply: ...


class OpenAILLM:
    def __init__(self, model: str, api_key: str):
        # The SDK retries 429/5xx with backoff; we just bound the total time.
        self.client = AsyncOpenAI(api_key=api_key or None, max_retries=3, timeout=30)
        self.model = model

    async def chat(self, system: str, messages: list[Message], tools: list[Tool]) -> Reply:
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, *map(self._to_openai, messages)],
            tools=[
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.schema}}
                for t in tools
            ],
        )
        msg = resp.choices[0].message
        calls = [ToolCall(c.id, c.function.name, json.loads(c.function.arguments)) for c in msg.tool_calls or []]
        usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
        raw = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            raw["tool_calls"] = [c.model_dump() for c in msg.tool_calls]
        return Reply(Message("assistant", msg.content, calls, raw=raw), usage)

    @staticmethod
    def _to_openai(m: Message) -> dict:
        if m.role == "tool":
            return {"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content}
        if m.role == "assistant" and m.raw:
            return m.raw
        return {"role": m.role, "content": m.content}


class BedrockLLM:
    """Bedrock Converse API: one request shape for every model family on Bedrock."""

    def __init__(self, model: str, region: str):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,
            config=Config(retries={"mode": "adaptive", "max_attempts": 5}, read_timeout=60),
        )
        self.model = model

    async def chat(self, system: str, messages: list[Message], tools: list[Tool]) -> Reply:
        resp = await asyncio.to_thread(  # boto3 is sync; keep the event loop free
            self.client.converse,
            modelId=self.model,
            system=[{"text": system}],
            messages=self._to_converse(messages),
            toolConfig={
                "tools": [
                    {"toolSpec": {"name": t.name, "description": t.description, "inputSchema": {"json": t.schema}}}
                    for t in tools
                ]
            },
            inferenceConfig={"maxTokens": 4096},
        )
        content = resp["output"]["message"]["content"]
        text = "".join(b["text"] for b in content if "text" in b) or None
        calls = [ToolCall(b["toolUse"]["toolUseId"], b["toolUse"]["name"], b["toolUse"]["input"]) for b in content if "toolUse" in b]
        usage = {"input_tokens": resp["usage"]["inputTokens"], "output_tokens": resp["usage"]["outputTokens"]}
        return Reply(Message("assistant", text, calls, raw=content), usage)

    @staticmethod
    def _to_converse(messages: list[Message]) -> list[dict]:
        out: list[dict] = []
        for m in messages:
            if m.role == "tool":
                role, blocks = "user", [{"toolResult": {"toolUseId": m.tool_call_id, "content": [{"text": m.content}]}}]
            elif m.role == "assistant":
                role, blocks = "assistant", m.raw or [{"text": m.content or ""}]
            else:
                role, blocks = "user", [{"text": m.content}]
            # Converse requires strictly alternating roles: parallel tool results share one user turn.
            if out and out[-1]["role"] == role:
                out[-1]["content"].extend(blocks)
            else:
                out.append({"role": role, "content": list(blocks)})
        return out
