from __future__ import annotations

from collections import Counter, deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import queue
import sqlite3
import threading
import time
from typing import Literal
from collections.abc import Iterator

logger = logging.getLogger(__name__)

SpanType = Literal["router", "retrieval", "generator", "risk_checker"]

DEFAULT_DB_FILENAME = "observability.sqlite3"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_EXPORT_FILENAME = "eval_dataset.json"
DEFAULT_METRICS_FILENAME = "metrics.json"
KNOWN_SPAN_FIELDS = {
    "session_id",
    "span_type",
    "input_snapshot",
    "output_snapshot",
    "latency_ms",
    "error",
    "schema_version",
    "created_at",
    "extra",
}
CONTENT_KEYS_TO_REMOVE = {
    "answer",
    "final_answer",
    "chunk_text",
    "text",
    "supporting_text",
    "claim",
}
HASH_KEYS = {"question", "source_file"}
METRIC_WINDOW_SECONDS = 300.0


@dataclass(frozen=True)
class Span:
    session_id: str
    span_type: str
    input_snapshot: dict
    output_snapshot: dict
    latency_ms: float
    error: str | None
    schema_version: str
    created_at: datetime
    extra: dict


@dataclass(frozen=True)
class Trace:
    session_id: str
    question: str
    final_answer: str | None
    total_latency_ms: float
    completed: bool
    created_at: datetime
    spans: list[Span]


class ObservabilityCollector:
    """Fire-and-forget observability writer backed by local SQLite."""

    def __init__(
        self,
        db_path: str | Path | None = None,
        *,
        write_delay_ms: float = 0.0,
        max_queue_size: int = 10_000,
        start_worker: bool = True,
    ) -> None:
        self.db_path = Path(db_path) if db_path is not None else Path(DEFAULT_OUTPUT_DIR) / DEFAULT_DB_FILENAME
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_delay_ms = write_delay_ms
        self._queue: queue.Queue[tuple[str, dict] | None] = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._write_events: deque[tuple[float, bool]] = deque()
        self._span_events: deque[tuple[float, str, float, bool, dict, dict]] = deque()
        self._trace_events: deque[tuple[float, bool, float]] = deque()
        self._route_distribution: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._ensure_schema()
        if start_worker:
            self._worker = threading.Thread(target=self._worker_loop, name="observability-writer", daemon=True)
            self._worker.start()

    def record_span(
        self,
        session_id: str,
        span_type: SpanType,
        input: dict,
        output: dict,
        latency_ms: float,
        error: str | None,
        schema_version: str,
        **extra: object,
    ) -> None:
        payload = {
            "session_id": session_id,
            "span_type": span_type,
            "input_snapshot": dict(input),
            "output_snapshot": dict(output),
            "latency_ms": float(latency_ms),
            "error": error,
            "schema_version": schema_version,
            "created_at": _utc_now_iso(),
            "extra": dict(extra),
        }
        self._enqueue("span", payload)
        self._update_span_metrics(payload)

    def record_trace(
        self,
        session_id: str,
        question: str,
        final_answer: str | None,
        total_latency_ms: float,
        completed: bool,
    ) -> None:
        payload = {
            "session_id": session_id,
            "question": question,
            "final_answer": final_answer,
            "total_latency_ms": float(total_latency_ms),
            "completed": bool(completed),
            "created_at": _utc_now_iso(),
        }
        self._enqueue("trace", payload)
        self._update_trace_metrics(payload)

    def get_trace(self, session_id: str) -> Trace | None:
        self.flush()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT session_id, question, final_answer, total_latency_ms, completed, created_at
                FROM traces
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return Trace(
            session_id=str(row["session_id"]),
            question=str(row["question"]),
            final_answer=row["final_answer"],
            total_latency_ms=float(row["total_latency_ms"]),
            completed=bool(row["completed"]),
            created_at=_parse_datetime(str(row["created_at"])),
            spans=self.get_spans_by_session(session_id),
        )

    def get_spans_by_session(self, session_id: str) -> list[Span]:
        self.flush()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, span_type, input_snapshot, output_snapshot, latency_ms,
                       error, schema_version, created_at, extra
                FROM spans
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [_span_from_row(row) for row in rows]

    def export_eval_dataset(
        self,
        span_type: str,
        date_range: tuple[datetime, datetime],
        anonymize: bool = True,
    ) -> list[dict]:
        self.flush()
        start, end = (_datetime_to_iso(value) for value in date_range)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    spans.session_id,
                    traces.question,
                    traces.final_answer,
                    traces.total_latency_ms,
                    traces.completed,
                    spans.span_type,
                    spans.input_snapshot,
                    spans.output_snapshot,
                    spans.latency_ms,
                    spans.error,
                    spans.schema_version,
                    spans.created_at,
                    spans.extra
                FROM spans
                LEFT JOIN traces ON traces.session_id = spans.session_id
                WHERE spans.span_type = ? AND spans.created_at >= ? AND spans.created_at <= ?
                ORDER BY spans.id ASC
                """,
                (span_type, start, end),
            ).fetchall()

        dataset = []
        for row in rows:
            item = {
                "session_id": row["session_id"],
                "trace": {
                    "question": row["question"],
                    "final_answer": row["final_answer"],
                    "total_latency_ms": row["total_latency_ms"],
                    "completed": bool(row["completed"]) if row["completed"] is not None else None,
                },
                "span": {
                    "span_type": row["span_type"],
                    "input_snapshot": _loads_json(row["input_snapshot"], {}),
                    "output_snapshot": _loads_json(row["output_snapshot"], {}),
                    "latency_ms": row["latency_ms"],
                    "error": row["error"],
                    "schema_version": row["schema_version"],
                    "created_at": row["created_at"],
                    "extra": _loads_json(row["extra"], {}),
                },
            }
            dataset.append(_anonymize(item) if anonymize else item)
        return dataset

    def get_metrics(self) -> dict[str, object]:
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            request_total = len(self._trace_events)
            errors = sum(1 for _, completed, _ in self._trace_events if not completed)
            latencies = [latency for _, _, latency in self._trace_events]
            generator_outputs = [output for _, span_type, _, _, _, output in self._span_events if span_type == "generator"]
            risk_outputs = [output for _, span_type, _, _, _, output in self._span_events if span_type == "risk_checker"]
            router_outputs = [output for _, span_type, _, _, _, output in self._span_events if span_type == "router"]
            write_errors = sum(1 for _, success in self._write_events if not success)
            write_total = len(self._write_events)

            return {
                "request_total": request_total,
                "request_error_rate": _ratio(errors, request_total),
                "p99_latency_ms": _percentile(latencies, 0.99),
                "answerable_rate": _ratio(
                    sum(1 for output in generator_outputs if output.get("answerable") is True),
                    len(generator_outputs),
                ),
                "faithfulness_proxy": _ratio(
                    sum(1 for output in generator_outputs if output.get("answerable") is True and bool(output.get("citations"))),
                    len(generator_outputs),
                ),
                "context_truncated_rate": _ratio(
                    sum(1 for output in generator_outputs if output.get("context_truncated") is True),
                    len(generator_outputs),
                ),
                "high_risk_rate": _ratio(
                    sum(1 for output in risk_outputs if output.get("risk_level") == "high"),
                    len(risk_outputs),
                ),
                "evidence_insufficient_rate": _ratio(
                    sum(1 for output in risk_outputs if output.get("evidence_sufficient") is False),
                    len(risk_outputs),
                ),
                "route_distribution": dict(self._route_distribution),
                "fallback_rate": _ratio(
                    sum(1 for output in router_outputs if output.get("fallback_triggered") is True),
                    len(router_outputs),
                ),
                "queue_depth": self._queue.qsize(),
                "write_error_rate": _ratio(write_errors, write_total),
            }

    def flush(self, timeout: float | None = 5.0) -> None:
        if self._worker is None:
            return
        start = time.monotonic()
        while self._queue.unfinished_tasks:
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError("observability writer did not flush in time")
            time.sleep(0.005)

    def close(self) -> None:
        if self._worker is None:
            return
        self.flush()
        self._stop.set()
        self._queue.put(None)
        self._worker.join(timeout=2.0)

    def __enter__(self) -> "ObservabilityCollector":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _enqueue(self, kind: str, payload: dict) -> None:
        try:
            self._queue.put_nowait((kind, payload))
        except queue.Full:
            logger.warning("Observability queue is full, dropping %s event", kind)
            self._record_write_result(False)

    def _worker_loop(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            try:
                if item is None:
                    return
                kind, payload = item
                if self._write_delay_ms:
                    time.sleep(self._write_delay_ms / 1000.0)
                self._write_event(kind, payload)
                self._record_write_result(True)
            except Exception as exc:
                logger.error("Observability write failed: %s", exc)
                self._record_write_result(False)
            finally:
                self._queue.task_done()

    def _write_event(self, kind: str, payload: dict) -> None:
        with self._connect() as conn:
            if kind == "span":
                conn.execute(
                    """
                    INSERT INTO spans (
                        session_id, span_type, input_snapshot, output_snapshot, latency_ms,
                        error, schema_version, created_at, extra
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        payload["session_id"],
                        payload["span_type"],
                        _dumps_json(payload["input_snapshot"]),
                        _dumps_json(payload["output_snapshot"]),
                        payload["latency_ms"],
                        payload["error"],
                        payload["schema_version"],
                        payload["created_at"],
                        _dumps_json(payload["extra"]),
                    ),
                )
            elif kind == "trace":
                conn.execute(
                    """
                    INSERT INTO traces (
                        session_id, question, final_answer, total_latency_ms, completed, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        question = excluded.question,
                        final_answer = excluded.final_answer,
                        total_latency_ms = excluded.total_latency_ms,
                        completed = excluded.completed,
                        created_at = excluded.created_at
                    """,
                    (
                        payload["session_id"],
                        payload["question"],
                        payload["final_answer"],
                        payload["total_latency_ms"],
                        int(payload["completed"]),
                        payload["created_at"],
                    ),
                )
            conn.commit()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS traces (
                    session_id TEXT PRIMARY KEY,
                    question TEXT NOT NULL,
                    final_answer TEXT,
                    total_latency_ms REAL NOT NULL,
                    completed INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS spans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    span_type TEXT NOT NULL,
                    input_snapshot TEXT NOT NULL,
                    output_snapshot TEXT NOT NULL,
                    latency_ms REAL NOT NULL,
                    error TEXT,
                    schema_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    extra TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_session ON spans(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_type_created ON spans(span_type, created_at)")
            conn.commit()

    def _update_span_metrics(self, payload: dict) -> None:
        now = time.monotonic()
        span_type = str(payload["span_type"])
        output = dict(payload["output_snapshot"])
        route = output.get("route") or payload["input_snapshot"].get("route")
        with self._lock:
            self._span_events.append(
                (
                    now,
                    span_type,
                    float(payload["latency_ms"]),
                    payload["error"] is not None,
                    dict(payload["input_snapshot"]),
                    output,
                )
            )
            if span_type == "router" and route:
                self._route_distribution[str(route)] += 1
            self._prune(now)

    def _update_trace_metrics(self, payload: dict) -> None:
        now = time.monotonic()
        with self._lock:
            self._trace_events.append((now, bool(payload["completed"]), float(payload["total_latency_ms"])))
            self._prune(now)

    def _record_write_result(self, success: bool) -> None:
        now = time.monotonic()
        with self._lock:
            self._write_events.append((now, success))
            self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - METRIC_WINDOW_SECONDS
        for events in (self._write_events, self._span_events, self._trace_events):
            while events and events[0][0] < cutoff:
                events.popleft()


def export_directory(
    db_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    span_type: str = "generator",
    anonymize: bool = True,
) -> dict[str, object]:
    collector = ObservabilityCollector(db_path=db_path)
    start = datetime.fromtimestamp(0, tz=timezone.utc)
    end = datetime.now(tz=timezone.utc)
    dataset = collector.export_eval_dataset(span_type=span_type, date_range=(start, end), anonymize=anonymize)
    metrics = collector.get_metrics()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / DEFAULT_EXPORT_FILENAME).write_text(_dumps_json(dataset), encoding="utf-8")
    (output_path / DEFAULT_METRICS_FILENAME).write_text(_dumps_json(metrics), encoding="utf-8")
    collector.close()
    return {
        "db_file": str(collector.db_path),
        "export_file": DEFAULT_EXPORT_FILENAME,
        "metrics_file": DEFAULT_METRICS_FILENAME,
        "span_type": span_type,
        "rows": len(dataset),
        "anonymize": anonymize,
    }


def _span_from_row(row: sqlite3.Row) -> Span:
    return Span(
        session_id=str(row["session_id"]),
        span_type=str(row["span_type"]),
        input_snapshot=_loads_json(row["input_snapshot"], {}),
        output_snapshot=_loads_json(row["output_snapshot"], {}),
        latency_ms=float(row["latency_ms"]),
        error=row["error"],
        schema_version=str(row["schema_version"]),
        created_at=_parse_datetime(str(row["created_at"])),
        extra=_loads_json(row["extra"], {}),
    )


def _anonymize(value: object) -> object:
    if isinstance(value, dict):
        cleaned: dict[str, object] = {}
        for key, child in value.items():
            if key in CONTENT_KEYS_TO_REMOVE:
                continue
            if key in HASH_KEYS and child is not None:
                cleaned[key] = _hash_text(str(child))
            else:
                cleaned[key] = _anonymize(child)
        return cleaned
    if isinstance(value, list):
        return [_anonymize(item) for item in value]
    return value


def _hash_text(value: str) -> dict[str, object]:
    return {
        "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        "length": len(value),
    }


def _dumps_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=_json_default)


def _loads_json(value: str | None, default: object) -> object:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return _datetime_to_iso(value)
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    return str(value)


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _datetime_to_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 6)


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile)))
    return float(ordered[index])
