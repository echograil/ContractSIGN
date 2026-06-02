from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .generator import (
    DEFAULT_INPUT_DIR,
    DEFAULT_LARGE_MODEL,
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SMALL_MODEL,
    generate_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a grounded answer from retrieval results.")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Input folder")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output folder")
    parser.add_argument("--question-file", default="question.txt")
    parser.add_argument("--router-file", default="router_output.json")
    parser.add_argument("--retrieval-file", default="retrieval_results.json")
    parser.add_argument("--answer-file", default="answer.txt")
    parser.add_argument("--session-id", default="local")
    parser.add_argument(
        "--use-api-generator",
        action="store_true",
        help="Use OpenAI-compatible generation; falls back to local extractive generation if unavailable",
    )
    parser.add_argument("--small-model", default=DEFAULT_SMALL_MODEL)
    parser.add_argument("--large-model", default=DEFAULT_LARGE_MODEL)
    parser.add_argument("--generator-base-url", default=DEFAULT_OPENAI_BASE_URL)
    args = parser.parse_args()

    result = generate_directory(
        input_dir=args.input,
        output_dir=args.output,
        question_filename=args.question_file,
        router_filename=args.router_file,
        retrieval_filename=args.retrieval_file,
        answer_filename=args.answer_file,
        session_id=args.session_id,
        use_api_generator=args.use_api_generator,
        small_model=args.small_model,
        large_model=args.large_model,
        generator_base_url=args.generator_base_url,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
