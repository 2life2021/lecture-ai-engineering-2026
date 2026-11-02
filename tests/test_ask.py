from types import SimpleNamespace

from app import ask


def test_ask_passes_temperature_and_prints_usage(monkeypatch, capsys):
    response = SimpleNamespace(
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7),
        text="answer",
    )
    request = {}
    client = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kwargs: request.update(kwargs) or response
        )
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("question", temperature=0.4) == "answer"

    assert request["config"].temperature == 0.4
    assert "入力: 12 / 出力: 7" in capsys.readouterr().out


def test_interactive_reuses_chat_session(monkeypatch, capsys):
    questions = iter(["first", "follow-up", "/exit"])
    sent_questions = []
    response = SimpleNamespace(usage_metadata=None, text="answer")
    chat = SimpleNamespace(send_message=lambda question: sent_questions.append(question) or response)
    create_calls = []
    client = SimpleNamespace(
        chats=SimpleNamespace(
            create=lambda **kwargs: create_calls.append(kwargs) or chat
        )
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("builtins.input", lambda _: next(questions))

    ask.interactive(temperature=0.8)

    assert sent_questions == ["first", "follow-up"]
    assert len(create_calls) == 1
    assert create_calls[0]["config"].temperature == 0.8
    assert "入力: 不明 / 出力: 不明" in capsys.readouterr().out