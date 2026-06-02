from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
import re
from pathlib import Path
from typing import Literal, Protocol

logger = logging.getLogger(__name__)

Route = Literal["qa", "summary", "generation", "agent"]
ModelTier = Literal["small", "large"]

DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_QUESTION_FILENAME = "question.txt"
DEFAULT_ROUTER_OUTPUT_FILENAME = "router_output.json"
DEFAULT_ANSWER_FILENAME = "answer.txt"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_ROUTER_MODEL = "gpt-5.4-mini"
DEFAULT_OPENAI_BASE_URL = "https://aihubmix.com/v1"

ROUTE_MODEL_TIERS: dict[Route, ModelTier] = {
    "qa": "small",
    "summary": "small",
    "generation": "large",
    "agent": "small",
}


@dataclass(frozen=True)
class RouterInput:
    question: str
    doc_type: str | None
    session_id: str


@dataclass(frozen=True)
class RouterOutput:
    route: Route
    confidence: float
    reasoning: str
    model_tier: ModelTier
    fallback_triggered: bool


@dataclass(frozen=True)
class BatchRouterResult:
    question_file: str
    answer_file: str
    result_file: str
    route: Route
    status: str


@dataclass(frozen=True)
class _Signal:
    route: Route
    weight: float
    label: str


class _RouterBackend(Protocol):
    def route(self, router_input: RouterInput) -> RouterOutput:
        ...


class QuestionRouter:
    """Question router implementing the Spec V0.2 public contract.

    Default mode is deterministic local routing so tests and smoke runs do not
    require credentials. With use_api_router=True, the router first asks a small
    OpenAI-compatible chat model for structured routing, then falls back to the
    local classifier if the API or dependency is unavailable.
    """

    def __init__(
        self,
        use_api_router: bool = False,
        router_model: str = DEFAULT_ROUTER_MODEL,
        router_base_url: str = DEFAULT_OPENAI_BASE_URL,
        router_backend: _RouterBackend | None = None,
    ) -> None:
        self._router_backend = router_backend
        if self._router_backend is None and use_api_router:
            try:
                self._router_backend = _ApiRouterBackend(
                    model=router_model,
                    base_url=router_base_url,
                )
            except Exception as exc:
                logger.warning("API router unavailable, falling back to local routing: %s", exc)

    def route(self, router_input: RouterInput) -> RouterOutput:
        if self._router_backend is not None:
            try:
                return self._router_backend.route(router_input)
            except Exception as exc:
                logger.warning("API router failed, falling back to local routing: %s", exc)
                output = _route_locally(router_input)
                return _with_reasoning_prefix(output, f"api_router_failed={type(exc).__name__}")

        return _route_locally(router_input)


class _ApiRouterBackend:
    def __init__(
        self,
        model: str = DEFAULT_ROUTER_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
    ) -> None:
        _load_dotenv_if_available()
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("langchain_openai is required for API routing") from exc

        self._llm = ChatOpenAI(model=model, temperature=0.0, base_url=base_url)

    def route(self, router_input: RouterInput) -> RouterOutput:
        response = self._llm.invoke(
            [
                ("system", ROUTER_SYSTEM_PROMPT),
                ("human", _router_user_prompt(router_input)),
            ]
        )
        return _parse_llm_router_output(_message_content(response))


def route_question(
    question: str,
    doc_type: str | None = None,
    session_id: str = "local",
    use_api_router: bool = False,
    router_model: str = DEFAULT_ROUTER_MODEL,
    router_base_url: str = DEFAULT_OPENAI_BASE_URL,
) -> RouterOutput:
    return QuestionRouter(
        use_api_router=use_api_router,
        router_model=router_model,
        router_base_url=router_base_url,
    ).route(RouterInput(question=question, doc_type=doc_type, session_id=session_id))


def route_directory(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    question_filename: str = DEFAULT_QUESTION_FILENAME,
    answer_filename: str = DEFAULT_ANSWER_FILENAME,
    doc_type: str | None = None,
    session_id: str = "local",
    use_api_router: bool = False,
    router_model: str = DEFAULT_ROUTER_MODEL,
    router_base_url: str = DEFAULT_OPENAI_BASE_URL,
) -> BatchRouterResult:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    question_path = input_path / question_filename
    answer_path = output_path / answer_filename
    result_path = output_path / DEFAULT_ROUTER_OUTPUT_FILENAME
    manifest_path = output_path / DEFAULT_MANIFEST_FILENAME

    if not question_path.exists():
        output = _fallback_output("missing question file")
        _write_json(result_path, asdict(output))
        _write_text_answer(answer_path, _format_answer(output))
        result = BatchRouterResult(
            question_file=question_filename,
            answer_file=answer_filename,
            result_file=DEFAULT_ROUTER_OUTPUT_FILENAME,
            route=output.route,
            status="missing_question",
        )
        _write_json(manifest_path, asdict(result))
        logger.warning("Question file does not exist: %s", question_path)
        return result

    question = question_path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff").strip()
    output = route_question(
        question=question,
        doc_type=doc_type,
        session_id=session_id,
        use_api_router=use_api_router,
        router_model=router_model,
        router_base_url=router_base_url,
    )
    _write_json(result_path, asdict(output))
    _write_text_answer(answer_path, _format_answer(output))

    result = BatchRouterResult(
        question_file=question_filename,
        answer_file=answer_filename,
        result_file=DEFAULT_ROUTER_OUTPUT_FILENAME,
        route=output.route,
        status="ok" if question else "empty_question",
    )
    _write_json(manifest_path, asdict(result))
    return result


def _route_locally(router_input: RouterInput) -> RouterOutput:
    question = _normalize_question(router_input.question)
    if not question:
        return _fallback_output("empty question")
    if _is_chinese_summary_request(router_input.question):
        return RouterOutput(
            route="summary",
            confidence=0.9,
            reasoning="local_router; chinese summary request",
            model_tier=ROUTE_MODEL_TIERS["summary"],
            fallback_triggered=False,
        )

    agent_signal = _agent_signal(question)
    if agent_signal is not None:
        return RouterOutput(
            route="agent",
            confidence=agent_signal.weight,
            reasoning=(
                "local_router; "
                f"{agent_signal.label}; current V0.2 does not execute multi-step "
                "agent flows, ask the user to split the task into one step."
            ),
            model_tier=ROUTE_MODEL_TIERS["agent"],
            fallback_triggered=False,
        )

    signals = _collect_intent_signals(question)
    if not signals:
        return _fallback_output("no strong intent signal")

    selected = _select_primary_signal(signals)
    if selected.weight < 0.55:
        return _fallback_output(f"weak intent signal: {selected.label}")

    confidence = _confidence(selected, signals)
    return RouterOutput(
        route=selected.route,
        confidence=confidence,
        reasoning=f"local_router; {_reasoning(selected, signals, router_input.doc_type)}",
        model_tier=ROUTE_MODEL_TIERS[selected.route],
        fallback_triggered=False,
    )


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question.strip().lower())


def _is_chinese_summary_request(question: str) -> bool:
    return bool(
        re.search(
            "\u4e3b\u8981\u5185\u5bb9|\u603b\u7ed3|\u6982\u62ec|\u6458\u8981|\u8bb2\u4e86\u4ec0\u4e48|\u6838\u5fc3\u5185\u5bb9|\u8981\u70b9",
            question,
        )
    )


def _collect_intent_signals(question: str) -> list[_Signal]:
    signals: list[_Signal] = []
    signals.extend(_keyword_signals(question, "generation", GENERATION_PATTERNS))
    signals.extend(_keyword_signals(question, "summary", SUMMARY_PATTERNS))
    signals.extend(_keyword_signals(question, "qa", QA_PATTERNS))
    return signals


def _keyword_signals(
    question: str,
    route: Route,
    patterns: list[tuple[str, float, str]],
) -> list[_Signal]:
    signals: list[_Signal] = []
    for pattern, weight, label in patterns:
        if re.search(pattern, question, flags=re.IGNORECASE):
            signals.append(_Signal(route=route, weight=weight, label=label))
    return signals


def _agent_signal(question: str) -> _Signal | None:
    for pattern, weight, label in AGENT_PATTERNS:
        if re.search(pattern, question, flags=re.IGNORECASE):
            return _Signal(route="agent", weight=weight, label=label)
    return None


def _select_primary_signal(signals: list[_Signal]) -> _Signal:
    route_priority: dict[Route, int] = {
        "generation": 3,
        "summary": 2,
        "qa": 1,
        "agent": 0,
    }
    return max(signals, key=lambda signal: (signal.weight, route_priority[signal.route]))


def _confidence(selected: _Signal, signals: list[_Signal]) -> float:
    competing_weights = [signal.weight for signal in signals if signal.route != selected.route]
    if not competing_weights:
        return round(min(0.95, selected.weight), 2)

    margin = selected.weight - max(competing_weights)
    return round(max(0.55, min(0.92, selected.weight + margin * 0.25)), 2)


def _reasoning(selected: _Signal, signals: list[_Signal], doc_type: str | None) -> str:
    matched = ", ".join(signal.label for signal in signals[:4])
    doc_note = f"; doc_type={doc_type}" if doc_type else ""
    return f"primary intent={selected.route} from signal '{selected.label}'{doc_note}; matched: {matched}"


def _fallback_output(reason: str) -> RouterOutput:
    return RouterOutput(
        route="qa",
        confidence=0.5,
        reasoning=f"local_router; fallback to qa because {reason}",
        model_tier=ROUTE_MODEL_TIERS["qa"],
        fallback_triggered=True,
    )


def _with_reasoning_prefix(output: RouterOutput, prefix: str) -> RouterOutput:
    return RouterOutput(
        route=output.route,
        confidence=output.confidence,
        reasoning=f"{prefix}; {output.reasoning}",
        model_tier=output.model_tier,
        fallback_triggered=True,
    )


def _parse_llm_router_output(content: str) -> RouterOutput:
    payload = _extract_json_object(content)
    route = payload.get("route")
    if route not in ROUTE_MODEL_TIERS:
        raise ValueError(f"invalid route from API router: {route!r}")

    confidence = float(payload.get("confidence", 0.0))
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"invalid confidence from API router: {confidence!r}")

    reasoning = str(payload.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("empty reasoning from API router")

    fallback_triggered = bool(payload.get("fallback_triggered", False))
    if route == "agent":
        reasoning = (
            "llm_router; current V0.2 does not execute multi-step agent flows, "
            f"ask the user to split the task into one step; {reasoning}"
        )
    else:
        reasoning = f"llm_router; {reasoning}"

    return RouterOutput(
        route=route,
        confidence=round(confidence, 2),
        reasoning=reasoning,
        model_tier=ROUTE_MODEL_TIERS[route],
        fallback_triggered=fallback_triggered,
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
        raise ValueError("API router response is not a JSON object")
    return payload


def _message_content(response: object) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _router_user_prompt(router_input: RouterInput) -> str:
    return json.dumps(
        {
            "question": router_input.question,
            "doc_type": router_input.doc_type,
            "session_id": router_input.session_id,
        },
        ensure_ascii=False,
    )


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()
    for env_path in [
        Path(__file__).with_name(".env"),
        Path(__file__).parents[1] / ".env",
    ]:
        if env_path.exists():
            load_dotenv(env_path)


def _format_answer(output: RouterOutput) -> str:
    return (
        f"route={output.route}\n"
        f"confidence={output.confidence:.2f}\n"
        f"model_tier={output.model_tier}\n"
        f"fallback_triggered={str(output.fallback_triggered).lower()}\n"
        f"reasoning={output.reasoning}\n"
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text_answer(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


GENERATION_PATTERNS: list[tuple[str, float, str]] = [
    (r"\b(draft|write|compose|generate|create|prepare|redraft|rewrite)\b", 0.9, "english generation verb"),
    (r"(起草|撰写|写一份|写一个|帮我写|生成|拟一份|拟定|改写|润色|出一版)", 0.9, "chinese generation verb"),
    (r"(template|clause|email|notice|letter).*\b(for|about)\b", 0.75, "generation artifact hint"),
    (r"(模板|条款|通知|邮件|函).*(给|关于|用于)", 0.75, "chinese artifact hint"),
]

SUMMARY_PATTERNS: list[tuple[str, float, str]] = [
    (r"\b(summarize|summary|recap|brief|outline)\b", 0.86, "english summary verb"),
    (r"\b(key points|main points|takeaways|tl;dr)\b", 0.84, "english key-points phrase"),
    (r"(总结|概括|摘要|归纳|梳理|提炼|关键点|重点|要点|列出.*风险点)", 0.86, "chinese summary phrase"),
]

QA_PATTERNS: list[tuple[str, float, str]] = [
    (r"\b(what|which|who|when|where|why|how|does|do|is|are|can|could|should)\b", 0.78, "english question word"),
    (r"(什么|哪个|哪些|谁|何时|什么时候|为什么|如何|怎么|是否|有没有|能否|可以吗|吗|么|规定|约定|要求|风险是什么)", 0.78, "chinese question phrase"),
    (r"\?$", 0.72, "question mark"),
    (r"(最大.*风险|主要.*风险|是否允许|是否需要|有无)", 0.8, "legal qa phrase"),
]

AGENT_PATTERNS: list[tuple[str, float, str]] = [
    (r"\b(first|then|after that|next)\b.*\b(then|after that|finally)\b", 0.88, "english multi-step sequence"),
    (r"(先|首先).*(然后|再|接着|最后).*(生成|起草|发送|更新|创建|检索|总结)", 0.88, "chinese multi-step sequence"),
]

ROUTER_SYSTEM_PROMPT = """You are ContractSIGN QuestionRouter V0.2.

Return only a JSON object with:
- route: one of "qa", "summary", "generation", "agent"
- confidence: number from 0 to 1
- reasoning: short non-empty routing reason for observability
- fallback_triggered: boolean

Routing policy:
- qa: answer a specific question about a contract, clause, risk, party, date, obligation, permission, or restriction.
- summary: summarize, recap, outline, list key points, extract takeaways, or aggregate risks.
- generation: draft, write, compose, generate, create, rewrite, redraft, or prepare text.
- agent: explicit multi-step workflow with chained actions. V0.2 reserves this branch and does not execute it.
- Choose one primary intent. Generation verbs have priority over summary phrases.
- If intent evidence is weak or ambiguous, return route "qa" and fallback_triggered true.
"""
