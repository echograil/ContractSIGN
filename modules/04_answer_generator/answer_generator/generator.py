from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
from urllib import error, request
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

Route = Literal["qa", "summary", "generation", "agent"]
ModelTier = Literal["small", "large"]

DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_QUESTION_FILENAME = "question.txt"
DEFAULT_RETRIEVAL_FILENAME = "retrieval_results.json"
DEFAULT_ROUTER_FILENAME = "router_output.json"
DEFAULT_GENERATOR_OUTPUT_FILENAME = "generator_output.json"
DEFAULT_ANSWER_FILENAME = "answer.txt"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_PROMPT_TEMPLATE = "answer_generator_v0.2.md"
DEFAULT_SMALL_MODEL = "gpt-5.4-mini"
DEFAULT_LARGE_MODEL = "gpt-5.4"
DEFAULT_OPENAI_BASE_URL = "https://aihubmix.com/v1"

MODEL_CONTEXT_LIMITS: dict[ModelTier, int] = {
    "small": 16_000,
    "large": 32_000,
}
CONTEXT_USAGE_RATIO = 0.6
MIN_RELEVANCE_SCORE = 0.05
MAX_LOCAL_CLAIMS = 5

ANSWER_UNAVAILABLE = (
    "Based on the currently retrieved document content, this question cannot be answered."
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.IGNORECASE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")

STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "could",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "should",
        "the",
        "this",
        "to",
        "under",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page_or_clause: str | None
    relevance_score: float


@dataclass(frozen=True)
class GeneratorInput:
    question: str
    route: Route
    retrieved_chunks: list[Chunk]
    model_tier: ModelTier
    session_id: str


@dataclass(frozen=True)
class Citation:
    chunk_id: str
    claim: str
    supporting_text: str


@dataclass(frozen=True)
class GeneratorOutput:
    answer: str
    citations: list[Citation]
    answerable: bool
    confidence: float
    conflict_detected: bool
    context_truncated: bool
    prompt_template_id: str


@dataclass(frozen=True)
class BatchGeneratorResult:
    question_file: str
    router_file: str
    retrieval_file: str
    answer_file: str
    result_file: str
    route: Route
    answerable: bool
    citation_count: int
    status: str


class _GeneratorBackend(Protocol):
    def generate(
        self,
        generator_input: GeneratorInput,
        chunks: list[Chunk],
        prompt_template_id: str,
        context_truncated: bool,
    ) -> GeneratorOutput:
        ...


class AnswerGenerator:
    """ContractSIGN answer generator implementing the Spec V0.2 contract.

    The default backend is deterministic and extractive: every claim is copied
    or tightly derived from a retrieved chunk, making local tests credential-free.
    With use_api_generator=True, an OpenAI-compatible chat model is tried first
    and falls back to the local backend on dependency, network, or parse failure.
    """

    def __init__(
        self,
        use_api_generator: bool = False,
        small_model: str = DEFAULT_SMALL_MODEL,
        large_model: str = DEFAULT_LARGE_MODEL,
        generator_base_url: str = DEFAULT_OPENAI_BASE_URL,
        prompt_dir: str | Path | None = None,
        generator_backend: _GeneratorBackend | None = None,
    ) -> None:
        self._prompt_dir = Path(prompt_dir) if prompt_dir is not None else Path(__file__).parents[1] / "prompts"
        self._local_backend = _LocalGeneratorBackend()
        self._generator_backend = generator_backend
        if self._generator_backend is None and use_api_generator:
            try:
                self._generator_backend = _ApiGeneratorBackend(
                    small_model=small_model,
                    large_model=large_model,
                    base_url=generator_base_url,
                    prompt_dir=self._prompt_dir,
                )
            except Exception as exc:
                logger.warning("API generator unavailable, falling back to local generation: %s", exc)

    def generate(self, generator_input: GeneratorInput) -> GeneratorOutput:
        prompt_template_id = prompt_template_id_for(self._prompt_dir / DEFAULT_PROMPT_TEMPLATE)
        selected_chunks, context_truncated = select_context_chunks(
            generator_input.retrieved_chunks,
            generator_input.model_tier,
        )

        if self._generator_backend is not None:
            try:
                return self._generator_backend.generate(
                    generator_input,
                    selected_chunks,
                    prompt_template_id,
                    context_truncated,
                )
            except Exception as exc:
                logger.warning("API generator failed, falling back to local generation: %s", exc)

        return self._local_backend.generate(
            generator_input,
            selected_chunks,
            prompt_template_id,
            context_truncated,
        )


def generate_answer(
    question: str,
    route: Route,
    retrieved_chunks: list[Chunk],
    model_tier: ModelTier,
    session_id: str = "local",
    use_api_generator: bool = False,
    small_model: str = DEFAULT_SMALL_MODEL,
    large_model: str = DEFAULT_LARGE_MODEL,
    generator_base_url: str = DEFAULT_OPENAI_BASE_URL,
) -> GeneratorOutput:
    return AnswerGenerator(
        use_api_generator=use_api_generator,
        small_model=small_model,
        large_model=large_model,
        generator_base_url=generator_base_url,
    ).generate(
        GeneratorInput(
            question=question,
            route=route,
            retrieved_chunks=retrieved_chunks,
            model_tier=model_tier,
            session_id=session_id,
        )
    )


def generate_directory(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    question_filename: str = DEFAULT_QUESTION_FILENAME,
    router_filename: str = DEFAULT_ROUTER_FILENAME,
    retrieval_filename: str = DEFAULT_RETRIEVAL_FILENAME,
    answer_filename: str = DEFAULT_ANSWER_FILENAME,
    session_id: str = "local",
    use_api_generator: bool = False,
    small_model: str = DEFAULT_SMALL_MODEL,
    large_model: str = DEFAULT_LARGE_MODEL,
    generator_base_url: str = DEFAULT_OPENAI_BASE_URL,
) -> BatchGeneratorResult:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    question_path = input_path / question_filename
    router_path = input_path / router_filename
    retrieval_path = input_path / retrieval_filename
    answer_path = output_path / answer_filename
    result_path = output_path / DEFAULT_GENERATOR_OUTPUT_FILENAME
    manifest_path = output_path / DEFAULT_MANIFEST_FILENAME

    question = _read_optional_text(question_path)
    route, model_tier = _load_router_decision(router_path)
    chunks = load_retrieved_chunks(retrieval_path)

    status = "ok"
    if not question_path.exists():
        status = "missing_question"
    elif not retrieval_path.exists():
        status = "missing_retrieval"
    elif not chunks:
        status = "empty_retrieval"

    generator_input = GeneratorInput(
        question=question,
        route=route,
        retrieved_chunks=chunks,
        model_tier=model_tier,
        session_id=session_id,
    )
    output = AnswerGenerator(
        use_api_generator=use_api_generator,
        small_model=small_model,
        large_model=large_model,
        generator_base_url=generator_base_url,
    ).generate(generator_input)

    _write_json(result_path, _generator_output_payload(output))
    _write_text_answer(answer_path, format_answer(output))

    result = BatchGeneratorResult(
        question_file=question_filename,
        router_file=router_filename,
        retrieval_file=retrieval_filename,
        answer_file=answer_filename,
        result_file=DEFAULT_GENERATOR_OUTPUT_FILENAME,
        route=route,
        answerable=output.answerable,
        citation_count=len(output.citations),
        status=status,
    )
    _write_json(manifest_path, asdict(result))
    return result


def select_context_chunks(chunks: list[Chunk], model_tier: ModelTier) -> tuple[list[Chunk], bool]:
    budget = int(MODEL_CONTEXT_LIMITS[model_tier] * CONTEXT_USAGE_RATIO)
    selected: list[Chunk] = []
    used_tokens = 0
    truncated = False

    for chunk in sorted(chunks, key=lambda item: item.relevance_score, reverse=True):
        chunk_tokens = estimate_tokens(chunk.text)
        if selected and used_tokens + chunk_tokens > budget:
            truncated = True
            continue
        if not selected and chunk_tokens > budget:
            truncated = True
            continue
        selected.append(chunk)
        used_tokens += chunk_tokens

    return selected, truncated


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def prompt_template_id_for(path: Path) -> str:
    if path.exists():
        content = path.read_text(encoding="utf-8", errors="replace")
    else:
        content = DEFAULT_PROMPT_TEMPLATE
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    return f"{path.name}:{digest}"


def load_retrieved_chunks(path: str | Path) -> list[Chunk]:
    result_path = Path(path)
    if not result_path.exists():
        return []
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Cannot read retrieval results %s: %s", result_path, exc)
        return []
    if not isinstance(payload, list):
        logger.warning("Retrieval results are not a list: %s", result_path)
        return []

    chunks: list[Chunk] = []
    for index, row in enumerate(payload):
        parsed = _parse_retrieval_row(row, index)
        if parsed is not None:
            chunks.append(parsed)
    return chunks


def format_answer(output: GeneratorOutput) -> str:
    lines = [
        output.answer,
        "",
        f"answerable={str(output.answerable).lower()}",
        f"confidence={output.confidence:.2f}",
        f"conflict_detected={str(output.conflict_detected).lower()}",
        f"context_truncated={str(output.context_truncated).lower()}",
        f"prompt_template_id={output.prompt_template_id}",
    ]
    if output.citations:
        lines.append("")
        lines.append("citations:")
        for citation in output.citations:
            lines.append(f"- {citation.chunk_id}: {citation.supporting_text}")
    return "\n".join(lines) + "\n"


class _LocalGeneratorBackend:
    def generate(
        self,
        generator_input: GeneratorInput,
        chunks: list[Chunk],
        prompt_template_id: str,
        context_truncated: bool,
    ) -> GeneratorOutput:
        conflict_detected = detect_conflict(chunks)
        answerable = is_answerable(generator_input, chunks)
        if not answerable:
            return GeneratorOutput(
                answer=ANSWER_UNAVAILABLE,
                citations=[],
                answerable=False,
                confidence=0.15 if chunks else 0.0,
                conflict_detected=conflict_detected,
                context_truncated=context_truncated,
                prompt_template_id=prompt_template_id,
            )

        citations = build_extract_citations(generator_input, chunks)
        if not citations:
            return GeneratorOutput(
                answer=ANSWER_UNAVAILABLE,
                citations=[],
                answerable=False,
                confidence=0.2,
                conflict_detected=conflict_detected,
                context_truncated=context_truncated,
                prompt_template_id=prompt_template_id,
            )

        prefix = "The following answer is based on partial retrieved results.\n\n" if context_truncated else ""
        answer = prefix + render_answer(generator_input.route, citations, conflict_detected)
        confidence = confidence_score(citations, chunks, conflict_detected, context_truncated)
        return GeneratorOutput(
            answer=answer,
            citations=citations,
            answerable=True,
            confidence=confidence,
            conflict_detected=conflict_detected,
            context_truncated=context_truncated,
            prompt_template_id=prompt_template_id,
        )


class _ApiGeneratorBackend:
    def __init__(
        self,
        small_model: str,
        large_model: str,
        base_url: str,
        prompt_dir: Path,
    ) -> None:
        _load_dotenv_if_available()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for API generation")

        self._api_key = api_key
        self._small_model = small_model
        self._large_model = large_model
        self._base_url = base_url.rstrip("/")
        self._prompt_path = prompt_dir / DEFAULT_PROMPT_TEMPLATE

    def generate(
        self,
        generator_input: GeneratorInput,
        chunks: list[Chunk],
        prompt_template_id: str,
        context_truncated: bool,
    ) -> GeneratorOutput:
        model = self._large_model if generator_input.model_tier == "large" else self._small_model
        content = _chat_completion(
            base_url=self._base_url,
            api_key=self._api_key,
            model=model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": _api_user_prompt(
                        generator_input,
                        chunks,
                        context_truncated,
                        prompt_template_id,
                    ),
                },
            ],
        )
        return _parse_llm_generator_output(content, prompt_template_id, context_truncated)

    def _system_prompt(self) -> str:
        if self._prompt_path.exists():
            return self._prompt_path.read_text(encoding="utf-8", errors="replace")
        return DEFAULT_SYSTEM_PROMPT


def is_answerable(generator_input: GeneratorInput, chunks: list[Chunk]) -> bool:
    if not generator_input.question.strip() or not chunks:
        return False
    if max((chunk.relevance_score for chunk in chunks), default=0.0) < MIN_RELEVANCE_SCORE:
        return False
    if generator_input.route in {"summary", "generation"}:
        return True

    question_terms = set(keyword_terms(generator_input.question))
    if not question_terms:
        return True
    context_terms = set(keyword_terms(" ".join(chunk.text for chunk in chunks)))
    return bool(question_terms & context_terms)


def build_extract_citations(generator_input: GeneratorInput, chunks: list[Chunk]) -> list[Citation]:
    question_terms = set(keyword_terms(generator_input.question))
    scored_sentences: list[tuple[float, Chunk, str]] = []
    for chunk in chunks:
        for sentence in split_sentences(chunk.text):
            sentence_terms = set(keyword_terms(sentence))
            if not sentence_terms and len(sentence) < 12:
                continue
            overlap = len(question_terms & sentence_terms) if question_terms else 1
            score = chunk.relevance_score + overlap * 0.25
            if generator_input.route in {"summary", "generation"}:
                score += min(len(sentence), 240) / 1000
            if overlap > 0 or generator_input.route in {"summary", "generation"}:
                scored_sentences.append((score, chunk, sentence))

    scored_sentences.sort(key=lambda item: item[0], reverse=True)
    citations: list[Citation] = []
    seen_supporting_text: set[str] = set()
    for _, chunk, sentence in scored_sentences:
        supporting_text = sentence.strip()
        normalized = re.sub(r"\s+", " ", supporting_text.lower())
        if normalized in seen_supporting_text:
            continue
        seen_supporting_text.add(normalized)
        citations.append(
            Citation(
                chunk_id=chunk.chunk_id,
                claim=_claim_from_sentence(supporting_text),
                supporting_text=supporting_text,
            )
        )
        if len(citations) >= MAX_LOCAL_CLAIMS:
            break
    return citations


def render_answer(route: Route, citations: list[Citation], conflict_detected: bool) -> str:
    if route == "generation":
        heading = "Draft grounded in retrieved contract text:"
    elif route == "summary":
        heading = "Summary grounded in retrieved contract text:"
    elif route == "agent":
        heading = "Single-step answer grounded in retrieved contract text:"
    else:
        heading = "Answer grounded in retrieved contract text:"

    lines = [heading]
    if conflict_detected:
        lines.append("Potential conflict detected across retrieved chunks; verify the cited clauses before relying on this answer.")
    for citation in citations:
        lines.append(f"- {citation.claim} [{citation.chunk_id}]")
    return "\n".join(lines)


def confidence_score(
    citations: list[Citation],
    chunks: list[Chunk],
    conflict_detected: bool,
    context_truncated: bool,
) -> float:
    top_score = max((chunk.relevance_score for chunk in chunks), default=0.0)
    coverage_bonus = min(0.25, len(citations) * 0.05)
    penalty = (0.15 if conflict_detected else 0.0) + (0.1 if context_truncated else 0.0)
    return round(max(0.0, min(0.95, 0.35 + top_score * 0.35 + coverage_bonus - penalty)), 2)


def detect_conflict(chunks: list[Chunk]) -> bool:
    text = " ".join(chunk.text.lower() for chunk in chunks)
    if not text:
        return False

    law_matches = set(re.findall(r"laws? of ([a-z][a-z\s]+?)(?:[.,;)]| shall| govern|$)", text))
    if len(_clean_short_values(law_matches)) >= 2:
        return True

    term_years = set(re.findall(r"\b(\d+)\s*\(\d+\)?\s*years?\b|\b(\d+)\s+years?\b", text))
    flattened_years = {value for match in term_years for value in match if value}
    if "perpetuity" in text and flattened_years:
        return True
    if len(flattened_years) >= 2 and ("term" in text or "audit" in text):
        return True

    if re.search(r"\bmay\s+assign\b", text) and re.search(r"\bmay\s+not\s+assign\b|\bshall\s+not\s+assign\b", text):
        return True

    return False


def keyword_terms(text: str) -> list[str]:
    terms = []
    for match in TOKEN_RE.finditer(text):
        term = match.group(0).lower()
        if term in STOP_WORDS:
            continue
        if len(term) > 5 and term.endswith("ing"):
            term = term[:-3]
        elif len(term) > 4 and term.endswith("ed"):
            term = term[:-2]
        elif len(term) > 3 and term.endswith("s"):
            term = term[:-1]
        terms.append(term)
    return terms


def split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return []
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(cleaned) if piece.strip()]
    if len(pieces) == 1 and len(pieces[0]) > 500:
        return [pieces[0][index : index + 500].strip() for index in range(0, len(pieces[0]), 500)]
    return [piece[:700].strip() for piece in pieces]


def _claim_from_sentence(sentence: str) -> str:
    sentence = sentence.strip()
    if len(sentence) <= 260:
        return sentence
    return sentence[:257].rstrip() + "..."


def _parse_retrieval_row(row: object, index: int) -> Chunk | None:
    if not isinstance(row, dict):
        return None
    score = float(row.get("score", row.get("relevance_score", 0.0)) or 0.0)
    nested = row.get("chunk")
    if isinstance(nested, dict):
        source_file = str(nested.get("source_file", "unknown"))
        location = nested.get("location") or nested.get("page_or_clause") or nested.get("clause_id")
        chunk_index = nested.get("chunk_index", index)
        chunk_id = str(nested.get("chunk_id") or f"{source_file}#{chunk_index}")
        text = str(nested.get("text", ""))
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            source_file=source_file,
            page_or_clause=str(location) if location is not None else None,
            relevance_score=score,
        )

    try:
        return Chunk(
            chunk_id=str(row.get("chunk_id") or f"{row.get('source_file', 'unknown')}#{index}"),
            text=str(row["text"]),
            source_file=str(row.get("source_file", "unknown")),
            page_or_clause=str(row["page_or_clause"]) if row.get("page_or_clause") is not None else None,
            relevance_score=score,
        )
    except KeyError:
        logger.warning("Skipping invalid retrieval row at index %s", index)
        return None


def _load_router_decision(path: Path) -> tuple[Route, ModelTier]:
    if not path.exists():
        return "qa", "small"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Cannot read router output %s: %s", path, exc)
        return "qa", "small"
    route = payload.get("route", "qa")
    model_tier = payload.get("model_tier", "small")
    if route not in {"qa", "summary", "generation", "agent"}:
        route = "qa"
    if model_tier not in {"small", "large"}:
        model_tier = "small"
    return route, model_tier


def _read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff").strip()


def _generator_output_payload(output: GeneratorOutput) -> dict[str, object]:
    return {
        "answer": output.answer,
        "citations": [asdict(citation) for citation in output.citations],
        "answerable": output.answerable,
        "confidence": output.confidence,
        "conflict_detected": output.conflict_detected,
        "context_truncated": output.context_truncated,
        "prompt_template_id": output.prompt_template_id,
    }


def _parse_llm_generator_output(
    content: str,
    prompt_template_id: str,
    context_truncated: bool,
) -> GeneratorOutput:
    payload = _extract_json_object(content)
    citations_payload = payload.get("citations", [])
    if not isinstance(citations_payload, list):
        raise ValueError("citations must be a list")
    citations = [
        Citation(
            chunk_id=str(row["chunk_id"]),
            claim=str(row["claim"]),
            supporting_text=str(row["supporting_text"]),
        )
        for row in citations_payload
        if isinstance(row, dict)
    ]
    answerable = bool(payload.get("answerable", False))
    if answerable and not citations:
        raise ValueError("answerable output must include citations")
    return GeneratorOutput(
        answer=str(payload.get("answer", ANSWER_UNAVAILABLE)),
        citations=citations,
        answerable=answerable,
        confidence=round(float(payload.get("confidence", 0.0)), 2),
        conflict_detected=bool(payload.get("conflict_detected", False)),
        context_truncated=bool(payload.get("context_truncated", context_truncated)),
        prompt_template_id=str(payload.get("prompt_template_id", prompt_template_id)),
    )


def _extract_json_object(content: str) -> dict[str, object]:
    stripped = content.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if match is None:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("generator response is not a JSON object")
    return payload


def _api_user_prompt(
    generator_input: GeneratorInput,
    chunks: list[Chunk],
    context_truncated: bool,
    prompt_template_id: str,
) -> str:
    return json.dumps(
        {
            "question": generator_input.question,
            "route": generator_input.route,
            "model_tier": generator_input.model_tier,
            "session_id": generator_input.session_id,
            "context_truncated": context_truncated,
            "prompt_template_id": prompt_template_id,
            "retrieved_chunks": [asdict(chunk) for chunk in chunks],
        },
        ensure_ascii=False,
    )


def _message_content(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = request.Request(
        f"{base_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=60) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"chat completion HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"chat completion network error: {exc.reason}") from exc

    try:
        return str(response_payload["choices"][0]["message"]["content"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected chat completion response: {response_payload}") from exc


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        load_dotenv = None

    env_paths = [
        Path(__file__).with_name(".env"),
        Path(__file__).parents[1] / ".env",
    ]
    if load_dotenv is not None:
        load_dotenv()
    for env_path in env_paths:
        if env_path.exists():
            if load_dotenv is not None:
                load_dotenv(env_path)
            else:
                _load_env_file(env_path)


def _load_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _clean_short_values(values: set[str]) -> set[str]:
    cleaned = set()
    for value in values:
        normalized = re.sub(r"\s+", " ", value).strip()
        if normalized and len(normalized.split()) <= 4:
            cleaned.add(normalized)
    return cleaned


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text_answer(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


DEFAULT_SYSTEM_PROMPT = """You are ContractSIGN AnswerGenerator V0.2.

Return only JSON with:
- answer: string
- citations: list of {chunk_id, claim, supporting_text}
- answerable: boolean
- confidence: number from 0 to 1
- conflict_detected: boolean
- context_truncated: boolean
- prompt_template_id: string

Rules:
- Use only retrieved_chunks.
- Every concrete claim in answer must have at least one citation.
- If retrieved_chunks cannot answer the question, set answerable=false and use the fixed unavailable answer.
- Do not invent legal facts, parties, dates, obligations, or remedies.
"""
