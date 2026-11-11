import logging

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from htx_ai_technical.llm_logging import LLMCallLogger


def fake_llm(message: AIMessage) -> GenericFakeChatModel:
    return GenericFakeChatModel(messages=iter([message]), callbacks=[LLMCallLogger(1.0, 10.0)])


def test_logs_tokens_cost_and_metadata(caplog):
    message = AIMessage(
        content="ok",
        usage_metadata={
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "output_token_details": {"reasoning": 50},
        },
        response_metadata={"finish_reason": "STOP"},
    )
    with caplog.at_level(logging.INFO, logger="htx_ai_technical.llm_logging"):
        fake_llm(message).invoke("hi", config={"metadata": {"field_group": "TopUps"}})

    line = caplog.records[0].getMessage()
    for part in ["input_tokens=1000", "output_tokens=200", "reasoning_tokens=50", "cost_usd=0.003000", "finish_reason=STOP", "field_group=TopUps", "latency_s="]:
        assert part in line


def test_logs_failed_call(caplog):
    llm = GenericFakeChatModel(messages=iter([]), callbacks=[LLMCallLogger(1.0, 10.0)])
    with caplog.at_level(logging.INFO, logger="htx_ai_technical.llm_logging"), pytest.raises(StopIteration):
        llm.invoke("hi")

    record = caplog.records[0]
    assert record.levelno == logging.WARNING
    assert "error=StopIteration" in record.getMessage()
