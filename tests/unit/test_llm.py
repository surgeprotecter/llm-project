from __future__ import annotations

from langchain_core.messages import AIMessage

from travel_agent.config import Settings
from travel_agent.llm import ModelGateway


def test_deepseek_short_tasks_disable_thinking_and_trace_retry(monkeypatch) -> None:
    options = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            options.update(kwargs)
            self.attempts = 0

        def bind(self, **kwargs):
            options.update(kwargs)
            return self

        def bind_tools(self, tools, **kwargs):
            return self

        def invoke(self, messages):
            self.attempts += 1
            if self.attempts == 1:
                raise TimeoutError("simulated provider timeout")
            return AIMessage(content='{"action":"use_tool","tool_name":"web_search"}')

        async def ainvoke(self, messages):
            return self.invoke(messages)

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChatOpenAI)
    gateway = ModelGateway(Settings(api_key="test-key", require_real_model=True))
    gateway.capture_io = True
    decision = gateway.structured_extract("Return JSON", "Pick a tool", stage="plan")

    # extra_body 内同时包含 thinking 与 max_tokens（实现已将 max_tokens 放入 extra_body 透传）
    assert options["extra_body"]["thinking"] == {"type": "disabled"}
    assert options["extra_body"]["max_tokens"] == 1024
    assert options["max_retries"] == 0

    assert decision == {"action": "use_tool", "tool_name": "web_search"}
    assert gateway.call_count == 2
    assert [(item["stage"], item["attempt"], item["status"]) for item in gateway.attempt_trace] == [
        ("plan", 1, "TimeoutError"),
        ("plan", 2, "success"),
    ]
    assert gateway.attempt_trace[0]["input_messages"] == [
        {"role": "system", "content": "Return JSON"},
        {"role": "human", "content": "Pick a tool"},
    ]
    assert gateway.attempt_trace[1]["output"] == '{"action":"use_tool","tool_name":"web_search"}'