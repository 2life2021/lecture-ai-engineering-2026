"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

import pytest
from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent, run_with_approval
from app import tools

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel(call_tools=sorted(BASE_TOOLS))
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_make_quiz_is_registered_and_called():
    model = TestModel(call_tools=["make_quiz"])
    with agent.override(model=model):
        result = agent.run_sync("第2回の復習クイズを1問ください")

    offered = {tool.name for tool in model.last_model_request_parameters.function_tools}
    called = {part.tool_name for part in _parts(result, ToolCallPart)}
    assert "make_quiz" in offered
    assert "make_quiz" in called


def test_save_learning_log_requires_approval_before_writing(monkeypatch, tmp_path):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)
    model = TestModel(call_tools=["save_learning_log"])

    with agent.override(model=model):
        pending = agent.run_sync("学習記録を保存してください")

    assert isinstance(pending.output, DeferredToolRequests)
    assert [call.tool_name for call in pending.output.approvals] == ["save_learning_log"]
    assert not list(log_dir.glob("*.txt"))

    result = agent.run_sync(
        message_history=pending.all_messages(),
        deferred_tool_results=pending.output.build_results(approve_all=True),
        model=TestModel(call_tools=[]),
    )

    files = list(log_dir.glob("*.txt"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8")
    assert isinstance(result.output, str)


@pytest.mark.parametrize(("response", "expected_files"), [("y", 1), ("n", 0)])
def test_run_with_approval_obeys_confirmation(
    monkeypatch, tmp_path, response, expected_files
):
    log_dir = tmp_path / "learning_log"
    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", log_dir)
    model = TestModel(call_tools=["save_learning_log"])
    prompts = []

    def confirm(prompt):
        prompts.append(prompt)
        model.call_tools = []
        return response

    monkeypatch.setattr("builtins.input", confirm)
    result = run_with_approval(agent, "学習記録を保存してください", model)

    assert isinstance(result.output, str)
    assert prompts == ["この操作を実行しますか？ [y/N]: "]
    assert len(list(log_dir.glob("*.txt"))) == expected_files


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"
