"""SaaS integrations: Gladly (REST + request signing) and Shopify (GraphQL).

Patterns worth copying: one shared async client per integration, explicit
timeouts, retry only what's retryable (429/5xx/network) with jittered backoff,
honor Retry-After, and treat GraphQL `errors` as failures even on HTTP 200.
"""

import asyncio
import hashlib
import hmac
import random
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import httpx

RETRYABLE = {429, 500, 502, 503, 504}


async def request(client: httpx.AsyncClient, method: str, url: str, attempts: int = 4, **kw) -> httpx.Response:
    for attempt in range(attempts):
        try:
            resp = await client.request(method, url, **kw)
            if resp.status_code not in RETRYABLE or attempt == attempts - 1:
                return resp.raise_for_status()
            delay = float(resp.headers.get("retry-after", 0)) or 0.5 * 2**attempt
        except httpx.TransportError:
            if attempt == attempts - 1:
                raise
            delay = 0.5 * 2**attempt
        await asyncio.sleep(min(delay, 10) + random.uniform(0, 0.25))
    raise AssertionError("unreachable")


# ---------------------------------------------------------------- Gladly ---


class GladlyClient:
    """Gladly REST API: Basic auth with <agent email>:<API token>."""

    def __init__(self, base_url: str, user: str, token: str):
        self.http = httpx.AsyncClient(base_url=f"{base_url}/api/v1", auth=(user, token), timeout=10)

    async def get_customer_message(self, conversation_id: str, item_id: str) -> str | None:
        """Text of a customer-sent item, or None for anything else (agent replies, notes, events).

        Filtering matters: ITEM/CREATED also fires for the notes *we* post, and
        without this check the bot would reply to itself in a loop.
        """
        resp = await request(self.http, "GET", f"/conversations/{conversation_id}/items/{item_id}")
        item = resp.json()
        if item.get("initiator", {}).get("type") != "CUSTOMER":
            return None
        content = item.get("content", {})
        # Chat/SMS items carry `content`, emails carry `subject` + `content`.
        text = "\n".join(filter(None, [content.get("subject"), content.get("content") or content.get("body")]))
        return text or None

    async def add_note(self, conversation_id: str, body: str) -> None:
        """Drafts go in as internal notes: the human agent reviews, edits, sends."""
        await request(self.http, "POST", f"/conversations/{conversation_id}/notes", json={"content": {"body": body}})

    async def aclose(self) -> None:
        await self.http.aclose()


def gladly_canonical_hash(
    method: str, path: str, query: Mapping[str, str], headers: Mapping[str, str], signed: list[str], body_sha256: str
) -> str:
    """SHA-256 of Gladly's normalized request (a SigV4-style canonical request)."""
    signed = sorted(h.lower() for h in signed)
    lower = {k.lower(): v for k, v in headers.items()}
    normalized = "\n".join([
        method.upper(),
        path,
        "&".join(f"{k.lower()}={v}" for k, v in sorted(query.items())),
        "".join(f"{h}:{lower.get(h, '').strip()}\n" for h in signed),
        ";".join(signed),
        body_sha256,
    ])
    return hashlib.sha256(normalized.encode()).hexdigest()


def gladly_signature(key: bytes, timestamp: str, canonical_hash: str) -> str:
    """Salt the key with the date (YYYYMMDD), then sign algorithm + time + canonical hash."""
    salted = hmac.new(key, timestamp[:8].encode(), hashlib.sha256).digest()
    to_sign = f"hmac-sha256\n{timestamp}\n{canonical_hash}"
    return hmac.new(salted, to_sign.encode(), hashlib.sha256).hexdigest()


def verify_gladly_request(
    key: bytes, method: str, path: str, query: Mapping[str, str], headers: Mapping[str, str], body: bytes,
    max_skew: timedelta = timedelta(minutes=5),
) -> bool:
    """Verify `Gladly-Authorization: SigningAlgorithm=hmac-sha256, SignedHeaders=a;b, Signature=hex`."""
    lower = {k.lower(): v for k, v in headers.items()}
    try:
        parts = dict(p.strip().split("=", 1) for p in lower["gladly-authorization"].split(","))
        signed, provided, timestamp = parts["SignedHeaders"].split(";"), parts["Signature"], lower["gladly-time"]
        sent_at = datetime.strptime(timestamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
    except (KeyError, ValueError):
        return False
    if "gladly-time" not in signed or abs(datetime.now(UTC) - sent_at) > max_skew:
        return False  # replay protection: the timestamp must be signed and fresh
    canonical = gladly_canonical_hash(method, path, query, lower, signed, hashlib.sha256(body).hexdigest())
    return hmac.compare_digest(gladly_signature(key, timestamp, canonical), provided)  # constant-time


# --------------------------------------------------------------- Shopify ---

ORDER_QUERY = """
query OrderByName($q: String!) {
  orders(first: 1, query: $q) {
    nodes {
      name
      createdAt
      displayFinancialStatus
      displayFulfillmentStatus
      totalPriceSet { shopMoney { amount currencyCode } }
    }
  }
}
"""

# Used when no shop is configured so the demo runs end-to-end without credentials.
DEMO_ORDERS = {
    "1001": {"name": "#1001", "displayFulfillmentStatus": "FULFILLED", "tracking": "1Z999AA10123456784", "carrier": "UPS"},
    "1002": {"name": "#1002", "displayFulfillmentStatus": "UNFULFILLED", "note": "Backordered, ships in 5-7 days"},
}


class ShopifyClient:
    def __init__(self, shop: str, token: str, api_version: str = "2025-07"):
        self.shop = shop
        self.http = httpx.AsyncClient(
            base_url=f"https://{shop}/admin/api/{api_version}",
            headers={"X-Shopify-Access-Token": token},
            timeout=10,
        )

    async def find_order(self, order_number: str) -> dict | None:
        number = order_number.lstrip("#")
        if not self.shop:
            return DEMO_ORDERS.get(number)
        resp = await request(self.http, "POST", "/graphql.json", json={"query": ORDER_QUERY, "variables": {"q": f"name:#{number}"}})
        payload = resp.json()
        if payload.get("errors"):  # GraphQL reports failures in the body, not the status code
            raise RuntimeError(f"Shopify GraphQL error: {payload['errors']}")
        nodes = payload["data"]["orders"]["nodes"]
        return nodes[0] if nodes else None

    async def aclose(self) -> None:
        await self.http.aclose()
