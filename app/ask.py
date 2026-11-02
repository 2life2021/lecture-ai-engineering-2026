"""第1回: LLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --interactive --temperature 0.7

通常実行は単発の質問、--interactive は会話履歴を保った対話。
講座に固有のことを聞くと、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors, types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def _generation_config(temperature: float | None) -> types.GenerateContentConfig | None:
    if temperature is None:
        return None
    return types.GenerateContentConfig(temperature=temperature)


def _print_usage(response) -> None:
    usage = response.usage_metadata
    input_tokens = usage.prompt_token_count if usage is not None else None
    output_tokens = usage.candidates_token_count if usage is not None else None
    input_display = input_tokens if input_tokens is not None else "不明"
    output_display = output_tokens if output_tokens is not None else "不明"
    print(f"[トークン数] 入力: {input_display} / 出力: {output_display}")


def ask(question: str, temperature: float | None = None) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=_generation_config(temperature),
    )
    _print_usage(response)

    return response.text


def interactive(temperature: float | None = None) -> None:
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(
        model=MODEL,
        config=_generation_config(temperature),
    )
    print("対話モードです。終了するには /exit または /quit を入力してください。")

    while True:
        try:
            question = input("あなた> ").strip()
        except EOFError:
            break
        if question.lower() in {"/exit", "/quit"}:
            break
        if not question:
            continue

        response = chat.send_message(question)
        _print_usage(response)
        print(response.text or "")


def main() -> None:
    parser = argparse.ArgumentParser(description="LLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--interactive", "-i", action="store_true", help="会話履歴を保って続けて質問する")
    parser.add_argument("--temperature", type=float, help="生成 temperature")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.interactive:
            interactive(temperature=args.temperature)
        else:
            print(f"[質問] {args.question}\n")
            print(ask(args.question, temperature=args.temperature))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
