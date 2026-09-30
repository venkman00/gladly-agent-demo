"""HTTP edge: Gladly webhook in, dashboard API out. Deployed as a Lambda via Mangum.

The webhook does the minimum (verify, validate, dedupe, enqueue) and returns 202
in milliseconds. The slow LLM work happens in the worker, so Gladly never times out
waiting on us and traffic spikes queue up instead of failing.
"""

import asyncio
import secrets
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import ValidationError

from app.integrations import verify_gladly_request
from app.models import GladlyWebhook, Job, NewTicket, Review, Ticket
from app.observability import configure_logging, record_event
from app.services import Services, get_services
from app.worker import process_job

configure_logging()
app = FastAPI(title="Support Copilot")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ auth ---


async def require_token(authorization: str = Header(default=""), svc: Services = Depends(get_services)) -> None:
    """Bearer token for the dashboard API. In production this sits behind SSO (SAML/OIDC at
    the load balancer or an auth middleware), and the token is for service-to-service calls."""
    expected = f"Bearer {svc.settings.api_token.get_secret_value()}"
    if not secrets.compare_digest(authorization.encode(), expected.encode()):
        raise HTTPException(401, "Invalid token")


async def verified_gladly_body(request: Request, svc: Services = Depends(get_services)) -> bytes:
    """Reject anything not signed by Gladly *before* parsing it."""
    body = await request.body()
    key = svc.settings.gladly_signing_key.get_secret_value().encode()
    if not key or not verify_gladly_request(
        key, request.method, request.url.path, dict(request.query_params), request.headers, body
    ):
        raise HTTPException(401, "Invalid Gladly signature")
    return body


async def enqueue(job: Job, svc: Services, background: BackgroundTasks) -> None:
    if svc.settings.env == "local":
        background.add_task(process_job, job, svc)  # no AWS needed for local dev
    else:
        await asyncio.to_thread(svc.sqs.send_message, QueueUrl=svc.settings.queue_url, MessageBody=job.model_dump_json())


# ---------------------------------------------------------------- routes ---


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


@app.post("/webhooks/gladly", status_code=202)
async def gladly_webhook(
    background: BackgroundTasks, body: bytes = Depends(verified_gladly_body), svc: Services = Depends(get_services)
) -> dict:
    try:
        event = GladlyWebhook.model_validate_json(body)
    except ValidationError as e:
        raise HTTPException(422, e.errors(include_url=False)) from e
    if event.type != "CONVERSATION_ITEM/CREATED":
        return {"status": "ignored"}

    c = event.content
    ticket = Ticket(id=c.conversation_item_id, conversation_id=c.conversation_id, customer_id=c.customer_id)
    if not await svc.store.create(ticket):  # at-least-once delivery: dedupe on the item id
        return {"status": "duplicate", "ticket_id": ticket.id}
    await enqueue(Job(ticket_id=ticket.id, conversation_id=c.conversation_id, customer_id=c.customer_id), svc, background)
    return {"status": "queued", "ticket_id": ticket.id}


api = APIRouter(prefix="/api", dependencies=[Depends(require_token)])


@api.post("/tickets", status_code=202)
async def create_ticket(req: NewTicket, background: BackgroundTasks, svc: Services = Depends(get_services)) -> Ticket:
    """Manual entry point, used by the dashboard, demos and load tests."""
    ticket = Ticket(id=str(uuid.uuid4()), conversation_id="manual", customer_id=req.customer_id, message=req.message)
    await svc.store.create(ticket)
    await enqueue(Job(ticket_id=ticket.id, conversation_id="manual", customer_id=req.customer_id, message=req.message), svc, background)
    return ticket


@api.get("/tickets")
async def list_tickets(svc: Services = Depends(get_services)) -> list[Ticket]:
    return await svc.store.recent()


@api.post("/tickets/{ticket_id}/review")
async def review_ticket(ticket_id: str, review: Review, svc: Services = Depends(get_services)) -> Ticket:
    """Human verdict on a draft. These labels are your eval set: track approval rate per intent over time."""
    ticket = await svc.store.get(ticket_id)
    if not ticket or not ticket.draft:
        raise HTTPException(404, "No draft for this ticket")
    ticket = ticket.model_copy(update={"status": "approved" if review.approved else "rejected"})
    await svc.store.save(ticket)
    record_event("SupportDraftReview", {"ticketId": ticket_id, "intent": ticket.draft.intent, "approved": review.approved,
                                        "confidence": ticket.draft.confidence})
    return ticket


app.include_router(api)
handler = Mangum(app, lifespan="off")
