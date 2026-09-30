from app.agent import Agent
from app.integrations import ShopifyClient
from app.llm import ToolCall
from tests.conftest import ScriptedLLM


def make_agent(llm, kb) -> Agent:
    # 0.2 is calibrated for HashEmbedder; real BGE scores run higher (see Settings.min_retrieval_score).
    return Agent(llm, kb, ShopifyClient(shop="", token=""), min_retrieval_score=0.2, min_confidence=0.7)


def respond(**overrides) -> ToolCall:
    args = {"intent": "return_refund", "reply": "You can return it within 30 days.", "confidence": 0.9,
            "citations": ["KB-101"], "escalate": False, "reason": "Covered by return policy."}
    return ToolCall("r1", "respond", {**args, **overrides})


async def test_grounded_answer_passes_guardrails(kb):
    llm = ScriptedLLM([[ToolCall("s1", "search_kb", {"query": "return policy days"})], [respond()]])
    result = await make_agent(llm, kb).run("What's your return policy?")
    assert not result.draft.escalate
    assert result.draft.citations == ["KB-101"]
    assert result.steps == 2


async def test_ungrounded_answer_is_escalated_and_fake_citation_dropped(kb):
    # The model answers without searching and cites an article it never retrieved.
    llm = ScriptedLLM([[respond(citations=["KB-999"])]])
    draft = (await make_agent(llm, kb).run("Can I return this?")).draft
    assert draft.escalate
    assert draft.citations == []
    assert "not grounded" in draft.reason


async def test_invalid_final_answer_is_fed_back_for_self_correction(kb):
    llm = ScriptedLLM([
        [respond(confidence=7)],  # out of range -> ValidationError goes back to the model
        [ToolCall("s1", "search_kb", {"query": "return policy"})],
        [respond()],
    ])
    result = await make_agent(llm, kb).run("Return policy?")
    tool_error = llm.seen[1][-1]
    assert tool_error.role == "tool" and "Invalid arguments" in tool_error.content
    assert not result.draft.escalate


async def test_order_lookup_tool(kb):
    llm = ScriptedLLM([
        [ToolCall("o1", "lookup_order", {"order_number": "#1001"})],
        [respond(intent="order_status", citations=[], reply="Order #1001 shipped via UPS.")],
    ])
    result = await make_agent(llm, kb).run("Where is order #1001?")
    assert "UPS" in llm.seen[1][-1].content  # tool result reached the model
    assert not result.draft.escalate  # order status doesn't need KB grounding
