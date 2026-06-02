from __future__ import annotations

import argparse
import json

from .retrieval import retrieve_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Run folder-based retrieval smoke test.")
    parser.add_argument("--input", default="input", help="Input folder with chunks.json and question.txt")
    parser.add_argument("--output", default="output", help="Output folder for answer.txt")
    parser.add_argument("--question-file", default="question.txt", help="Question txt filename in input folder")
    parser.add_argument("--answer-file", default="answer.txt", help="Answer txt filename in output folder")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to return")
    parser.add_argument(
        "--use-api-embeddings",
        action="store_true",
        help="Use OpenAI-compatible embedding API; falls back to local hashing if unavailable",
    )
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-base-url", default="https://aihubmix.com/v1")
    args = parser.parse_args()

    result = retrieve_directory(
        input_dir=args.input,
        output_dir=args.output,
        question_filename=args.question_file,
        answer_filename=args.answer_file,
        top_k=args.top_k,
        use_api_embeddings=args.use_api_embeddings,
        embedding_model=args.embedding_model,
        embedding_base_url=args.embedding_base_url,
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
