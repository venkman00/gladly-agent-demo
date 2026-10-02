"""Settings: env vars for config, AWS Secrets Manager for secrets.

Resolved once per Lambda container (cold start) and cached, so warm invocations
never pay the Secrets Manager round-trip.
"""

import json
from functools import lru_cache
from typing import Literal

import boto3
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: Literal["local", "aws"] = "local"
    aws_region: str = "us-east-1"
    table_name: str = "support-tickets"
    queue_url: str = ""
    secret_id: str = ""  # Secrets Manager secret holding the fields marked (secret) below

    # LLM
    llm_provider: Literal["openai", "bedrock"] = "openai"
    openai_model: str = "gpt-4.1-mini"
    bedrock_model: str = "anthropic.claude-opus-5-5"  # use the model/inference-profile ID from your Bedrock console
    openai_api_key: SecretStr = SecretStr("")  # (secret)

    # Retrieval + guardrails
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: str | None = None
    # BGE cosine scores are compressed: on our KB, relevant ~0.72+, uncovered topic ~0.60, nonsense ~0.36.
    # Calibrate on your own data; don't copy thresholds between embedding models.
    min_retrieval_score: float = 0.65
    min_confidence: float = 0.7
    max_agent_steps: int = 5

    # Gladly
    gladly_base_url: str = ""  # e.g. https://acme.gladly.com
    gladly_api_user: str = ""  # agent email the API token belongs to
    gladly_api_token: SecretStr = SecretStr("")  # (secret)
    gladly_webhook_token: SecretStr = SecretStr("")  # (secret) sent by Gladly in the X-Webhook-Token header

    # Shopify (order lookups via GraphQL)
    shopify_shop: str = ""  # e.g. acme.myshopify.com
    shopify_token: SecretStr = SecretStr("")  # (secret)

    # Dashboard auth (in production: SSO/SAML at the edge, this token for service-to-service)
    api_token: SecretStr = SecretStr("dev-token")  # (secret)


def _load_secret(secret_id: str, region: str) -> dict:
    client = boto3.client("secretsmanager", region_name=region)
    return json.loads(client.get_secret_value(SecretId=secret_id)["SecretString"])


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.secret_id:
        secrets = _load_secret(settings.secret_id, settings.aws_region)
        # Re-validate so secret values are coerced to the declared types (SecretStr etc.).
        settings = Settings(**{**settings.model_dump(), **secrets})
    return settings
