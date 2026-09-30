"""Composition root: builds each dependency once per process (i.e. per Lambda container).

Everything is lazy, so the API Lambda never loads the embedding model and a cold
start only pays for what that function actually uses.
"""

from functools import cached_property, lru_cache

import boto3

from app.agent import Agent
from app.config import Settings, get_settings
from app.integrations import GladlyClient, ShopifyClient
from app.knowledge import BGEEmbedder, KnowledgeBase
from app.llm import LLM, BedrockLLM, OpenAILLM
from app.store import DynamoStore, MemoryStore, Store


class Services:
    def __init__(self, settings: Settings):
        self.settings = settings

    @cached_property
    def store(self) -> Store:
        s = self.settings
        return MemoryStore() if s.env == "local" else DynamoStore(s.table_name, s.aws_region)

    @cached_property
    def sqs(self):
        return boto3.client("sqs", region_name=self.settings.aws_region)

    @cached_property
    def llm(self) -> LLM:
        s = self.settings
        if s.llm_provider == "bedrock":
            return BedrockLLM(s.bedrock_model, s.aws_region)
        return OpenAILLM(s.openai_model, s.openai_api_key.get_secret_value())

    @cached_property
    def kb(self) -> KnowledgeBase:
        return KnowledgeBase.from_file(BGEEmbedder(self.settings.embedding_model, self.settings.embedding_cache_dir))

    @cached_property
    def gladly(self) -> GladlyClient | None:
        s = self.settings
        if not s.gladly_base_url:
            return None
        return GladlyClient(s.gladly_base_url, s.gladly_api_user, s.gladly_api_token.get_secret_value())

    @cached_property
    def shopify(self) -> ShopifyClient:
        return ShopifyClient(self.settings.shopify_shop, self.settings.shopify_token.get_secret_value())

    @cached_property
    def agent(self) -> Agent:
        s = self.settings
        return Agent(self.llm, self.kb, self.shopify, s.min_retrieval_score, s.min_confidence, s.max_agent_steps)


@lru_cache
def get_services() -> Services:
    return Services(get_settings())
