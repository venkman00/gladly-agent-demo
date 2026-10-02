import json

import httpx
import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from app.api import app
from app.config import Settings
from app.integrations import GladlyClient
from app.services import Services, get_services

TOKEN = "test-webhook-token"
AUTH = {"x-webhook-token": TOKEN}


@pytest.fixture
def client(monkeypatch):
    svc = Services(Settings(env="local", gladly_webhook_token=TOKEN, api_token="t", _env_file=None))
    app.dependency_overrides[get_services] = lambda: svc
    # These tests cover the edge; test_agent covers the brain. Record jobs instead of running them.
    svc.jobs = []
    monkeypatch.setattr(api_module, "enqueue", lambda job, svc, bg: _record(svc, job))
    yield TestClient(app), svc
    app.dependency_overrides.clear()


async def _record(svc, job):
    svc.jobs.append(job)


# Shape from Gladly's webhook docs: payloads carry IDs only, the text is fetched by the worker.
WEBHOOK = json.dumps({
    "id": "evt-1", "type": "CONTACT/MESSAGE_RECEIVED", "timestamp": "2026-09-30T12:00:00Z",
    "initiator": {"id": "cust-1", "type": "CUSTOMER"},
    "content": {"channel": "EMAIL", "contactId": "contact-1", "conversationId": "conv-1", "customerId": "cust-1",
                "messageId": "item-1", "context": {}},
}).encode()


def test_webhook_is_queued_once(client):
    http, svc = client
    first = http.post("/webhooks/gladly", content=WEBHOOK, headers=AUTH)
    retry = http.post("/webhooks/gladly", content=WEBHOOK, headers=AUTH)
    assert first.status_code == 202 and first.json()["status"] == "queued"
    assert retry.json()["status"] == "duplicate"
    assert svc.store.items["item-1"].status == "queued"
    assert [(j.ticket_id, j.conversation_id) for j in svc.jobs] == [("item-1", "conv-1")]


def test_webhook_without_valid_token_rejected(client):
    http, svc = client
    assert http.post("/webhooks/gladly", content=WEBHOOK).status_code == 401
    assert http.post("/webhooks/gladly", content=WEBHOOK, headers={"x-webhook-token": "wrong"}).status_code == 401
    assert svc.jobs == []


def test_webhook_rejected_when_no_token_configured(client):
    http, svc = client
    svc.settings.gladly_webhook_token = Settings(_env_file=None).gladly_webhook_token  # empty: fail closed
    assert http.post("/webhooks/gladly", content=WEBHOOK, headers={"x-webhook-token": ""}).status_code == 401


def test_ping_gets_200_and_other_events_are_ignored(client):
    """Gladly sends PING when a webhook is created and only saves the webhook if it gets a 200."""
    http, svc = client
    ping = http.post("/webhooks/gladly", content=b'{"type": "PING"}', headers=AUTH)
    assert ping.status_code == 200
    other = http.post("/webhooks/gladly", content=b'{"type": "CONVERSATION/CLOSED", "content": {}}', headers=AUTH)
    assert other.status_code == 202 and other.json()["status"] == "ignored"
    bad = http.post("/webhooks/gladly", content=b'{"type": "CONTACT/MESSAGE_RECEIVED", "content": {}}', headers=AUTH)
    assert bad.status_code == 422
    assert svc.jobs == []


async def test_gladly_fetches_item_and_keeps_only_customer_messages():
    items = {
        "item-1": {"initiator": {"type": "CUSTOMER"}, "content": {"type": "EMAIL", "subject": "Order", "content": "Where is it?"}},
        "item-2": {"initiator": {"type": "AGENT"}, "content": {"type": "CHAT_MESSAGE", "content": "On its way!"}},
    }
    paths = []

    def handler(req: httpx.Request) -> httpx.Response:
        paths.append(req.url.path)
        return httpx.Response(200, json=items[req.url.path.rsplit("/", 1)[-1]])

    gladly = GladlyClient("https://acme.gladly.com", "bot@acme.com", "t")
    gladly.http = httpx.AsyncClient(base_url="https://acme.gladly.com/api/v1", transport=httpx.MockTransport(handler))
    assert await gladly.get_customer_message("item-1") == "Order\nWhere is it?"
    assert await gladly.get_customer_message("item-2") is None
    assert paths == ["/api/v1/conversation-items/item-1", "/api/v1/conversation-items/item-2"]


def test_dashboard_api_requires_token(client):
    http, _ = client
    assert http.get("/api/tickets").status_code == 401
    assert http.get("/api/tickets", headers={"authorization": "Bearer t"}).status_code == 200
