"""SQS -> Lambda worker. Runs the agent and writes the draft back to Gladly.

Reliability in four lines of config and code:
- SQS delivers at least once   -> the ticket status check makes reprocessing a no-op
- a batch has 5 messages       -> report *partial* failures so only the bad ones retry
- a message keeps failing      -> after 3 receives, the redrive policy moves it to the DLQ
- warm invocations             -> one long-lived event loop, so HTTP connection pools are reused
"""

import asyncio
import logging
import time

from app.models import Draft, Job
from app.observability import configure_logging, record_event
from app.services import Services, get_services

configure_logging()
log = logging.getLogger(__name__)
_loop = asyncio.new_event_loop()


async def process_job(job: Job, svc: Services) -> None:
    ticket = await svc.store.get(job.ticket_id)
    if ticket is None or ticket.status != "queued":
        log.info("skip already-processed ticket", extra={"fields": {"ticket_id": job.ticket_id}})
        return

    started = time.perf_counter()
    message = job.message or await svc.gladly.get_customer_message(job.ticket_id)
    if message is None:
        await svc.store.save(ticket.model_copy(update={"status": "ignored"}))
        return
    result = await svc.agent.run(message)
    draft = result.draft

    # Post the note before marking the ticket done: a retry can then duplicate a note, but never lose one.
    if svc.gladly and job.message is None:  # only Gladly-sourced tickets get a note back
        await svc.gladly.add_note(job.conversation_id, render_note(draft))

    latency_ms = int((time.perf_counter() - started) * 1000)
    status = "escalated" if draft.escalate else "drafted"
    await svc.store.save(ticket.model_copy(update={"message": message, "draft": draft, "status": status, "latency_ms": latency_ms}))

    record_event("SupportDraft", {
        "ticketId": ticket.id,
        "intent": draft.intent,
        "status": status,
        "confidence": draft.confidence,
        "escalate": draft.escalate,
        "citations": len(draft.citations),
        "steps": result.steps,
        "latencyMs": latency_ms,
        "llmProvider": svc.settings.llm_provider,
        **result.usage,
    })


def render_note(draft: Draft) -> str:
    header = "🚩 AI suggests ESCALATION" if draft.escalate else f"🤖 AI draft (confidence {draft.confidence:.0%})"
    sources = f"\nSources: {', '.join(draft.citations)}" if draft.citations else ""
    return f"{header}\nIntent: {draft.intent}\nWhy: {draft.reason}\n\n{draft.reply}{sources}"


async def _process_record(record: dict, svc: Services) -> None:
    # Parse inside the task so one malformed body fails alone, not the whole batch.
    await process_job(Job.model_validate_json(record["body"]), svc)


def handler(event: dict, context) -> dict:
    svc = get_services()
    records = event["Records"]
    results = _loop.run_until_complete(asyncio.gather(*(_process_record(r, svc) for r in records), return_exceptions=True))

    failures = []
    for record, outcome in zip(records, results):
        if isinstance(outcome, Exception):
            log.error("job failed", exc_info=outcome, extra={"fields": {"messageId": record["messageId"]}})
            failures.append({"itemIdentifier": record["messageId"]})
    return {"batchItemFailures": failures}  # requires ReportBatchItemFailures on the event source
