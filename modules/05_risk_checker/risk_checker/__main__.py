from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .checker import (
    DEFAULT_ANSWER_FILENAME,
    DEFAULT_GENERATOR_OUTPUT_FILENAME,
    DEFAULT_INPUT_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_QUESTION_FILENAME,
    DEFAULT_RETRIEVAL_FILENAME,
    DEFAULT_ROUTER_FILENAME,
    DEFAULT_RULES_FILENAME,
    check_directory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check generated contract answers for compliance risk.")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Input folder")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output folder")
    parser.add_argument("--question-file", default=DEFAULT_QUESTION_FILENAME)
    parser.add_argument("--router-file", default=DEFAULT_ROUTER_FILENAME)
    parser.add_argument("--retrieval-file", default=DEFAULT_RETRIEVAL_FILENAME)
    parser.add_argument("--generator-file", default=DEFAULT_GENERATOR_OUTPUT_FILENAME)
    parser.add_argument("--answer-file", default=DEFAULT_ANSWER_FILENAME)
    parser.add_argument("--rules-file", default=DEFAULT_RULES_FILENAME)
    parser.add_argument("--session-id", default="local")
    args = parser.parse_args()

    result = check_directory(
        input_dir=args.input,
        output_dir=args.output,
        question_filename=args.question_file,
        router_filename=args.router_file,
        retrieval_filename=args.retrieval_file,
        generator_filename=args.generator_file,
        answer_filename=args.answer_file,
        rules_filename=args.rules_file,
        session_id=args.session_id,
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
