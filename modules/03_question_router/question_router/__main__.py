from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .router import (
    DEFAULT_INPUT_DIR,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_ROUTER_MODEL,
    route_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Route a question from input/ to output/.")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Input folder with question.txt")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output folder for router result")
    parser.add_argument("--question-file", default="question.txt", help="Question txt filename")
    parser.add_argument("--answer-file", default="answer.txt", help="Readable answer txt filename")
    parser.add_argument("--doc-type", default=None, help="Optional upstream doc_type")
    parser.add_argument("--session-id", default="local", help="Session id for observability")
    parser.add_argument(
        "--use-api-router",
        action="store_true",
        help="Use OpenAI-compatible small LLM router; falls back to local routing if unavailable",
    )
    parser.add_argument("--router-model", default=DEFAULT_ROUTER_MODEL)
    parser.add_argument("--router-base-url", default=DEFAULT_OPENAI_BASE_URL)
    args = parser.parse_args()

    result = route_directory(
        input_dir=args.input,
        output_dir=args.output,
        question_filename=args.question_file,
        answer_filename=args.answer_file,
        doc_type=args.doc_type,
        session_id=args.session_id,
        use_api_router=args.use_api_router,
        router_model=args.router_model,
        router_base_url=args.router_base_url,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
