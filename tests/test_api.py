import hashlib
import json
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import app.api as api_module
from app.api import app
from app.config import Settings
from app.integrations import gladly_canonical_hash, gladly_signature, verify_gladly_request
from app.services import Services, get_services

KEY = b"test-signing-key"


def test_signature_matches_gladly_docs_vector():
    """Worked example from Gladly's request-signing docs."""
    canonical = gladly_canonical_hash(
        "POST", "/api/v2/customer/lookup", {},
        {"accept": "application/json", "content-type": "application/json",
         "gladly-correlation-id": "vXmSEPjVSWCaCMzvjufxZg", "gladly-time": "20190213T214016Z",
         "x-b3-traceid": "bd799210f8d549609a08ccef8ee7f166"},
        ["accept", "content-type", "gladly-correlation-id", "gladly-time", "x-b3-traceid"],
        "f187462a1d8e09bc86ea4b4ff8c022e5e4ed23ae783b3b1b5baee4b8d69e02ca",
    )
    assert canonical == "f96c13077adb3c06df1fa5fda8a6f32d7067735f63aa58d47e45fd6429d3cad3"
    assert gladly_signature(b"test-apikey-1", "20190213T214016Z", canonical) == (
        "4c633fca4914f51df04c9ec40f4545d66d653e771c6634e33eed52a242bc278c"
    )


def signed_headers(body: bytes, sent_at: datetime | None = None) -> dict:
    """Sign a request the way Gladly would."""
    path = "/webhooks/gladly"
    headers = {"content-type": "application/json", "gladly-time": (sent_at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")}
    signed = sorted(headers)
    canonical = gladly_canonical_hash("POST", path, {}, headers, signed, hashlib.sha256(body).hexdigest())
    sig = gladly_signature(KEY, headers["gladly-time"], canonical)
    return {**headers, "gladly-authorization": f"SigningAlgorithm=hmac-sha256, SignedHeaders={';'.join(signed)}, Signature={sig}"}


@pytest.fixture
def client(monkeypatch):
    svc = Services(Settings(env="local", gladly_signing_key=KEY.decode(), api_token="t", _env_file=None))
    app.dependency_overrides[get_services] = lambda: svc
    # These tests cover the edge; test_agent covers the brain. Record jobs instead of running them.
    svc.jobs = []
    monkeypatch.setattr(api_module, "enqueue", lambda job, svc, bg: _record(svc, job))
    yield TestClient(app), svc
    app.dependency_overrides.clear()


async def _record(svc, job):
    svc.jobs.append(job)


WEBHOOK = json.dumps({
    "id": "evt-1", "type": "CONVERSATION_ITEM/CREATED", "createdAt": "2026-09-30T12:00:00Z",
    "content": {"customerId": "cust-1", "conversationId": "conv-1", "conversationItemId": "item-1"},
}).encode()


def test_signed_webhook_is_queued_once(client):
    http, svc = client
    first = http.post("/webhooks/gladly", content=WEBHOOK, headers=signed_headers(WEBHOOK))
    retry = http.post("/webhooks/gladly", content=WEBHOOK, headers=signed_headers(WEBHOOK))
    assert first.status_code == 202 and first.json()["status"] == "queued"
    assert retry.json()["status"] == "duplicate"
    assert svc.store.items["item-1"].status == "queued"
    assert [j.ticket_id for j in svc.jobs] == ["item-1"]


def test_tampered_or_unsigned_webhook_rejected(client):
    http, _ = client
    tampered = WEBHOOK.replace(b"item-1", b"item-2")
    assert http.post("/webhooks/gladly", content=tampered, headers=signed_headers(WEBHOOK)).status_code == 401
    assert http.post("/webhooks/gladly", content=WEBHOOK).status_code == 401


def test_stale_timestamp_rejected():
    body = b"{}"
    assert verify_gladly_request(KEY, "POST", "/webhooks/gladly", {}, signed_headers(body), body)
    headers = signed_headers(body, sent_at=datetime.now(UTC) - timedelta(minutes=10))  # validly signed, but replayed
    assert not verify_gladly_request(KEY, "POST", "/webhooks/gladly", {}, headers, body)


def test_dashboard_api_requires_token(client):
    http, _ = client
    assert http.get("/api/tickets").status_code == 401
    assert http.get("/api/tickets", headers={"authorization": "Bearer t"}).status_code == 200
