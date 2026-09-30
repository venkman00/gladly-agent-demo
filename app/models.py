"""Pydantic models: the contract at every boundary (webhooks, queue, LLM output, API)."""

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class GladlyModel(BaseModel):
    """Gladly speaks camelCase; we speak snake_case. Unknown fields are ignored."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, extra="ignore")


class WebhookContent(GladlyModel):
    customer_id: str
    conversation_id: str
    conversation_item_id: str


class GladlyWebhook(GladlyModel):
    id: str
    type: str  # e.g. CONVERSATION_ITEM/CREATED
    created_at: datetime
    content: WebhookContent


class Job(BaseModel):
    """SQS message body. Small and self-describing; the worker fetches the rest."""

    ticket_id: str
    conversation_id: str
    customer_id: str
    message: str | None = None  # set for manual/demo tickets; otherwise fetched from Gladly


# Literal (not Enum) keeps the JSON schema flat: no $defs/$ref for LLM tool schemas.
Intent = Literal["order_status", "return_refund", "shipping", "account", "other"]


class Draft(BaseModel):
    """What the agent must return. Its JSON schema doubles as the `respond` tool schema."""

    intent: Intent
    reply: str = Field(description="Customer-facing reply. Friendly, concise, no internal notes.")
    confidence: float = Field(ge=0, le=1, description="How sure you are the reply is correct and complete.")
    citations: list[str] = Field(default_factory=list, description="IDs of knowledge-base articles used.")
    escalate: bool = Field(description="True if a human must handle this (angry, legal, unclear, out of policy).")
    reason: str = Field(description="One sentence for the human agent explaining the draft or escalation.")


TicketStatus = Literal["queued", "drafted", "escalated", "approved", "rejected", "ignored"]


class Ticket(BaseModel):
    id: str
    conversation_id: str
    customer_id: str
    message: str | None = None
    status: TicketStatus = "queued"
    draft: Draft | None = None
    latency_ms: int | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class NewTicket(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    customer_id: str = "demo-customer"


class Review(BaseModel):
    approved: bool
