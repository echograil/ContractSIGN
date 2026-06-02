from __future__ import annotations

import argparse
from dataclasses import asdict
import json

from .ingestion import DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, ingest_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch ingest PDFs from input/ to output/.")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Input directory for PDF files.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output directory for JSON results.")
    parser.add_argument("--hint-doc-type", default=None, help="Optional doc_type hint.")
    args = parser.parse_args()

    results = ingest_directory(
        input_dir=args.input,
        output_dir=args.output,
        hint_doc_type=args.hint_doc_type,
    )
    print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
