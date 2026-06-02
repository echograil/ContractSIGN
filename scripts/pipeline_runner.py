from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Callable, Iterable
from uuid import uuid4


RELEASE_ROOT = Path(__file__).resolve().parents[1]
MODULES_ROOT = RELEASE_ROOT / "modules"
WORKSPACE_ROOT = RELEASE_ROOT / "workspace"

MODULE_PATHS = {
    "ingestion": MODULES_ROOT / "01_document_ingestion",
    "retrieval": MODULES_ROOT / "02_retrieval",
    "router": MODULES_ROOT / "03_question_router",
    "generator": MODULES_ROOT / "04_answer_generator",
    "risk_checker": MODULES_ROOT / "05_risk_checker",
    "observability": MODULES_ROOT / "06_observability",
}

for module_path in MODULE_PATHS.values():
    sys.path.insert(0, str(module_path))

from document_ingestion import ingest_directory
from retrieval_module import retrieve_directory
from question_router import route_directory
from answer_generator import generate_directory
from risk_checker import check_directory
from observability_collector import ObservabilityCollector, export_directory


@dataclass(frozen=True)
class StepResult:
    id: str
    label: str
    status: str
    latency_ms: float
    input_dir: str
    output_dir: str
    files_out: list[str]
    summary: dict[str, object]
    error: str | None = None


def run_pipeline(
    pdf_path: str | Path,
    question: str,
    *,
    session_id: str | None = None,
    use_api_router: bool = False,
    use_api_embeddings: bool = False,
    use_api_generator: bool = False,
) -> dict[str, object]:
    load_env_file(RELEASE_ROOT / ".env")
    session_id = session_id or f"run-{uuid4().hex[:10]}"
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    for module_path in MODULE_PATHS.values():
        ensure_io_dirs(module_path)
        clean_dir(module_path / "input")
        clean_dir(module_path / "output")

    db_path = MODULE_PATHS["observability"] / "output" / "observability.sqlite3"
    collector = ObservabilityCollector(db_path=db_path)
    started = time.perf_counter()
    completed = False
    final_answer: str | None = None
    steps: list[StepResult] = []
    pipeline_error: Exception | None = None

    try:
        copy_file(pdf_path, MODULE_PATHS["ingestion"] / "input" / pdf_path.name)
        write_question(question, MODULE_PATHS["router"] / "input")
        write_question(question, MODULE_PATHS["retrieval"] / "input")

        steps.append(run_step(
            "ingestion",
            "01 Document Ingestion",
            lambda: ingest_directory(
                input_dir=MODULE_PATHS["ingestion"] / "input",
                output_dir=MODULE_PATHS["ingestion"] / "output",
            ),
        ))
        copy_matching(MODULE_PATHS["ingestion"] / "output", MODULE_PATHS["retrieval"] / "input", ["chunks.json", "*.chunks.json"])

        steps.append(recorded_step(
            collector,
            "router",
            "03 Question Router",
            session_id,
            {"question": question},
            lambda: route_directory(
                input_dir=MODULE_PATHS["router"] / "input",
                output_dir=MODULE_PATHS["router"] / "output",
                session_id=session_id,
                use_api_router=use_api_router,
            ),
            output_file=MODULE_PATHS["router"] / "output" / "router_output.json",
            schema_version="router_v0.2",
        ))
        copy_file(MODULE_PATHS["router"] / "output" / "router_output.json", MODULES_ROOT / "04_answer_generator" / "input" / "router_output.json")
        copy_file(MODULE_PATHS["router"] / "output" / "router_output.json", MODULES_ROOT / "05_risk_checker" / "input" / "router_output.json")

        steps.append(recorded_step(
            collector,
            "retrieval",
            "02 Retrieval",
            session_id,
            {"question": question, "chunks_file": "chunks.json"},
            lambda: retrieve_directory(
                input_dir=MODULE_PATHS["retrieval"] / "input",
                output_dir=MODULE_PATHS["retrieval"] / "output",
                use_api_embeddings=use_api_embeddings,
            ),
            output_file=MODULE_PATHS["retrieval"] / "output" / "retrieval_results.json",
            schema_version="retrieval_v0.2",
        ))
        copy_file(MODULE_PATHS["retrieval"] / "output" / "retrieval_results.json", MODULES_ROOT / "04_answer_generator" / "input" / "retrieval_results.json")
        copy_file(MODULE_PATHS["retrieval"] / "output" / "retrieval_results.json", MODULES_ROOT / "05_risk_checker" / "input" / "retrieval_results.json")

        write_question(question, MODULE_PATHS["generator"] / "input")
        steps.append(recorded_step(
            collector,
            "generator",
            "04 Answer Generator",
            session_id,
            {"question": question},
            lambda: generate_directory(
                input_dir=MODULE_PATHS["generator"] / "input",
                output_dir=MODULE_PATHS["generator"] / "output",
                session_id=session_id,
                use_api_generator=use_api_generator,
            ),
            output_file=MODULE_PATHS["generator"] / "output" / "generator_output.json",
            schema_version="generator_v0.2",
        ))
        copy_file(MODULE_PATHS["generator"] / "output" / "generator_output.json", MODULES_ROOT / "05_risk_checker" / "input" / "generator_output.json")

        write_question(question, MODULE_PATHS["risk_checker"] / "input")
        steps.append(recorded_step(
            collector,
            "risk_checker",
            "05 Risk & Evidence",
            session_id,
            {"question": question},
            lambda: check_directory(
                input_dir=MODULE_PATHS["risk_checker"] / "input",
                output_dir=MODULE_PATHS["risk_checker"] / "output",
                rules_filename=str(MODULE_PATHS["risk_checker"] / "risk_rules_v0.2.yaml"),
                session_id=session_id,
            ),
            output_file=MODULE_PATHS["risk_checker"] / "output" / "risk_checker_output.json",
            schema_version="risk_checker_v0.2",
        ))

        final_answer = read_text(MODULE_PATHS["risk_checker"] / "output" / "answer.txt")
        completed = True
    except Exception as exc:
        pipeline_error = exc
    finally:
        total_latency_ms = (time.perf_counter() - started) * 1000
        collector.record_trace(
            session_id=session_id,
            question=question,
            final_answer=final_answer,
            total_latency_ms=total_latency_ms,
            completed=completed,
        )
        collector.flush()
        steps.append(run_step(
            "observability",
            "06 Observability Export",
            lambda: export_directory(
                db_path=db_path,
                output_dir=MODULE_PATHS["observability"] / "output",
                span_type="generator",
                anonymize=True,
            ),
        ))
        collector.close()
    if pipeline_error is not None:
        raise pipeline_error
    return build_result(session_id, question, final_answer, steps, collector, started, completed)


def recorded_step(
    collector: ObservabilityCollector,
    span_type: str,
    label: str,
    session_id: str,
    input_snapshot: dict,
    action: Callable[[], object],
    *,
    output_file: Path,
    schema_version: str,
) -> StepResult:
    started = time.perf_counter()
    try:
        result = action()
        latency_ms = (time.perf_counter() - started) * 1000
        output_snapshot = read_json(output_file, {})
        collector.record_span(
            session_id=session_id,
            span_type=span_type,
            input=input_snapshot,
            output=output_snapshot if isinstance(output_snapshot, dict) else {"items": output_snapshot},
            latency_ms=latency_ms,
            error=None,
            schema_version=schema_version,
        )
        return make_step_result(span_type, label, "ok", latency_ms, result=result)
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        collector.record_span(
            session_id=session_id,
            span_type=span_type,
            input=input_snapshot,
            output={},
            latency_ms=latency_ms,
            error=str(exc),
            schema_version=schema_version,
        )
        return make_step_result(span_type, label, "error", latency_ms, error=str(exc))


def run_step(step_id: str, label: str, action: Callable[[], object]) -> StepResult:
    started = time.perf_counter()
    try:
        result = action()
        latency_ms = (time.perf_counter() - started) * 1000
        return make_step_result(step_id, label, "ok", latency_ms, result=result)
    except Exception as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return make_step_result(step_id, label, "error", latency_ms, error=str(exc))


def make_step_result(
    step_id: str,
    label: str,
    status: str,
    latency_ms: float,
    *,
    result: object | None = None,
    error: str | None = None,
) -> StepResult:
    module_path = module_path_for_step(step_id)
    return StepResult(
        id=step_id,
        label=label,
        status=status,
        latency_ms=round(latency_ms, 2),
        input_dir=str(module_path / "input"),
        output_dir=str(module_path / "output"),
        files_out=list_relative_files(module_path / "output"),
        summary=summary_payload(result),
        error=error,
    )


def build_result(
    session_id: str,
    question: str,
    final_answer: str | None,
    steps: list[StepResult],
    collector: ObservabilityCollector,
    started: float,
    completed: bool,
) -> dict[str, object]:
    return {
        "session_id": session_id,
        "completed": completed,
        "question": question,
        "answer": final_answer or "",
        "total_latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "steps": [asdict(step) for step in steps],
        "metrics": collector.get_metrics(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def module_path_for_step(step_id: str) -> Path:
    if step_id == "ingestion":
        return MODULE_PATHS["ingestion"]
    if step_id == "retrieval":
        return MODULE_PATHS["retrieval"]
    if step_id == "router":
        return MODULE_PATHS["router"]
    if step_id == "generator":
        return MODULE_PATHS["generator"]
    if step_id == "risk_checker":
        return MODULE_PATHS["risk_checker"]
    if step_id == "observability":
        return MODULE_PATHS["observability"]
    raise KeyError(step_id)


def ensure_io_dirs(module_path: Path) -> None:
    (module_path / "input").mkdir(parents=True, exist_ok=True)
    (module_path / "output").mkdir(parents=True, exist_ok=True)


def clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_matching(source_dir: Path, target_dir: Path, patterns: Iterable[str]) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for source in source_dir.glob(pattern):
            copy_file(source, target_dir / source.name)


def copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def write_question(question: str, input_dir: Path) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "question.txt").write_text(question.strip() + "\n", encoding="utf-8")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").strip()


def read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return default


def list_relative_files(path: Path) -> list[str]:
    if not path.exists():
        return []
    return sorted(str(child.relative_to(path)).replace("\\", "/") for child in path.rglob("*") if child.is_file())


def summary_payload(result: object | None) -> dict[str, object]:
    if result is None:
        return {}
    if isinstance(result, list):
        return {"items": [summary_payload(item) for item in result[:8]], "count": len(result)}
    if hasattr(result, "__dataclass_fields__"):
        return asdict(result)
    if isinstance(result, dict):
        return result
    return {"value": str(result)}


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the six-module ContractSIGN file pipeline.")
    parser.add_argument("--pdf", required=True, help="Path to a PDF contract")
    parser.add_argument("--question", required=True, help="Question to ask")
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--use-api-router", action="store_true")
    parser.add_argument("--use-api-embeddings", action="store_true")
    parser.add_argument("--use-api-generator", action="store_true")
    args = parser.parse_args()

    result = run_pipeline(
        pdf_path=args.pdf,
        question=args.question,
        session_id=args.session_id,
        use_api_router=args.use_api_router,
        use_api_embeddings=args.use_api_embeddings,
        use_api_generator=args.use_api_generator,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
