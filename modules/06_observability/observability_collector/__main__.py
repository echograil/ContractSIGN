from __future__ import annotations

import argparse
import json

from .collector import DEFAULT_OUTPUT_DIR, export_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ContractSIGN observability data for offline eval.")
    parser.add_argument("--db", default=None, help="SQLite database path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Output folder")
    parser.add_argument("--span-type", default="generator")
    parser.add_argument("--no-anonymize", action="store_true", help="Keep raw text in export")
    args = parser.parse_args()

    result = export_directory(
        db_path=args.db,
        output_dir=args.output,
        span_type=args.span_type,
        anonymize=not args.no_anonymize,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
