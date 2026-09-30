# Support Copilot

An AI copilot for customer support. A customer writes in on Gladly. Within seconds the human agent sees a drafted reply as an internal note. The draft is grounded in the help center and live order data, has a confidence score and sources, and is flagged for escalation when the AI shouldn't be trusted.

About 1,000 lines of Python and 140 of TypeScript. Built for the **20% of the work that delivers 80% of the value**.

```
Gladly ──webhook──▶ API Lambda (FastAPI) ──▶ SQS ──▶ Worker Lambda ──▶ Gladly note
         signed      verify · dedupe · 202     │       │  agent loop:
                            │                  DLQ     │   search_kb (BGE embeddings)
                            ▼                          │   lookup_order (Shopify GraphQL)
                        DynamoDB ◀─────────────────────┘   respond (Pydantic-validated)
                            ▲                              + deterministic guardrails
               React dashboard (review → approve/reject = eval labels)
```

## Where each skill lives

| Skill | Where | The idea to explain |
|---|---|---|
| FastAPI, async, Pydantic | `app/api.py`, `app/models.py` | Validate at every boundary. `async` for I/O, `to_thread` for sync SDKs and CPU work |
| Pytest | `tests/` | 9 tests on the risky parts: signing, idempotency, guardrails, self-correction. Fakes, no network |
| Lambda + SQS | `app/worker.py`, `template.yaml` | Webhook returns 202 fast. Worker retries, partial batch failures, DLQ |
| DynamoDB | `app/store.py` | Conditional write = idempotency under at-least-once delivery |
| Secrets Manager | `app/config.py` | Load once per cold start, cache, never in env vars or code |
| OpenAI SDK + Bedrock | `app/llm.py` | One tiny interface, two adapters. Provider is a config flag |
| Prompt engineering | `app/agent.py` `SYSTEM_PROMPT` | Grounding rules, escalation criteria, untrusted input in tags |
| LLM integration / agents | `app/agent.py` | Bounded tool loop. Final answer *is* a tool call validated by Pydantic |
| BGE embeddings / similarity | `app/knowledge.py` | Normalized vectors → cosine = dot product. Query-side instruction prefix |
| REST / GraphQL integrations | `app/integrations.py` | Timeouts, retry only 429/5xx, Retry-After. GraphQL errors arrive on HTTP 200 |
| Gladly | `app/integrations.py`, `app/api.py` | Request signing (verified against Gladly's docs vector), notes API, loop prevention |
| React + TypeScript | `web/src/` | Typed API client mirroring the Pydantic models. Human-in-the-loop review |
| New Relic NRQL | `app/observability.py` + below | One wide event per ticket answers most production questions |
| K6 | `load/k6.js` | Load-test the edge, with a p95 threshold as a CI gate |
| Auth middleware | `app/api.py` `require_token` | Constant-time compare. SSO/SAML at the edge in production |

## Design decisions (the "minimum effective dose")

1. **Drafts, not auto-replies.** The AI writes an internal note and a human sends the reply. This one decision removes most of the risk, so the system can be shipped and learn from real traffic on day one.
2. **Async by queue, not by thread.** The webhook does verify → validate → dedupe → enqueue and returns 202. LLM latency and outages never cause Gladly timeouts. Spikes queue instead of failing.
3. **Structured output via a `respond` tool.** The `Draft` Pydantic model is the tool schema *and* the validator. If the model sends bad output, the validation errors go back to it so it can fix itself. This works the same on OpenAI and Bedrock.
4. **Guardrails in code, not in the prompt.** Escalate when confidence is low or the answer isn't grounded. Drop citations the model never retrieved. Prompts are suggestions; code is policy.
5. **Idempotency everywhere.** Gladly and SQS both deliver at least once. A conditional put on the item ID plus a status check in the worker makes a duplicate a no-op.
6. **Calibrate thresholds on your own data.** BGE scores are compressed. On this KB, relevant queries score ~0.72+, an uncovered topic ("gift cards") ~0.60, nonsense ~0.36. So the grounding cutoff is 0.65, not a copied default.
7. **Human reviews are your eval set.** Approve/reject emits `SupportDraftReview`. Track approval rate per intent and use it to decide what to auto-send someday.

**Deliberately left out** (add when the data says so): vector database (a numpy matrix handles hundreds of articles), streaming, WebSockets (polling is fine), auto-send, multi-agent orchestration, LangChain, a GSI for listing tickets, SAML implementation (belongs at the load balancer/IdP).

**Real-world gotchas handled:** Gladly's `CONVERSATION_ITEM/CREATED` also fires for the bot's own notes, so only `initiator.type == CUSTOMER` items are processed (otherwise the bot replies to itself forever). The Bedrock Converse API needs strictly alternating roles, so parallel tool results are merged into one turn. Signature checks run *before* parsing, and stale timestamps are rejected to stop replays.

## Run it locally

Local mode needs no AWS: in-memory store, inline processing, and demo orders `#1001`/`#1002`.

```bash
cp .env.example .env                   # add OPENAI_API_KEY (or set LLM_PROVIDER=bedrock)
uv venv && uv pip install -e ".[dev,embeddings]"
.venv/bin/uvicorn app.api:app --reload # :8000 (first ticket loads the embedding model)
cd web && npm install && npm run dev   # :5173
.venv/bin/pytest -q                    # 9 tests, <1s, no network
```

Try: *"Where is my order #1002?"*, *"How long do refunds take?"*, *"Do you sell gift cards?"* (should escalate: not in KB), *"This is the third time I'm asking, I'm filing a chargeback"* (should escalate: angry/legal).

## Deploy

```bash
sam build && sam deploy --guided       # builds the image, creates queue/DLQ/table/secret/Lambdas
aws secretsmanager put-secret-value --secret-id <AppSecret> --secret-string file://secrets.json
```

In Gladly: point a webhook for `CONVERSATION_ITEM/CREATED` at the `WebhookUrl` output, and use an API token from a dedicated bot agent user. For Bedrock, enable model access in your region and set `BedrockModel` to the model or inference-profile ID shown in the console.

**Before production, verify against your Gladly org:** the webhook payload fields, whether webhooks are signed with the same `Gladly-Authorization` scheme (Gladly documents it for Lookup Adaptors; `test_signature_matches_gladly_docs_vector` proves the algorithm), and the notes request body.

## NRQL

```sql
-- Escalation rate and draft quality per intent
SELECT percentage(count(*), WHERE escalate IS true) AS 'escalated', average(confidence)
FROM SupportDraft FACET intent SINCE 1 day ago

-- Latency and cost drivers
SELECT percentile(latencyMs, 50, 95), average(steps), sum(input_tokens + output_tokens)
FROM SupportDraft FACET llmProvider TIMESERIES

-- Did humans agree with the AI? (the metric that matters)
SELECT percentage(count(*), WHERE approved IS true) FROM SupportDraftReview FACET intent SINCE 1 week ago
```

Alert on: DLQ depth > 0, p95 `latencyMs` > 20s, and approval rate dropping week over week.

## Video outline (≈12 min)

1. **Problem → architecture** (1 min): the diagram above. Why drafts instead of auto-replies.
2. **Edge** (2 min): `api.py` webhook. Signature before parse, the replay window, dedupe, 202. Show the tests.
3. **Brain** (4 min): `agent.py`. The loop, tools, `respond` as structured output, guardrails. Then `knowledge.py`: BGE, cosine, and the threshold calibration story.
4. **Providers** (1 min): `llm.py`. Switch `LLM_PROVIDER` live.
5. **Reliability** (2 min): `worker.py` + `template.yaml`. At-least-once delivery, partial batch failures, DLQ, `MaximumConcurrency`, the self-reply loop bug.
6. **Operate** (2 min): the dashboard demo, then NRQL queries, then the k6 run with thresholds.
