from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import re
from typing import Literal

logger = logging.getLogger(__name__)

RiskLevel = Literal["high", "medium", "low", "unknown"]
InterruptionType = Literal["blocking", "non_blocking", "none"]

DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_QUESTION_FILENAME = "question.txt"
DEFAULT_ROUTER_FILENAME = "router_output.json"
DEFAULT_RETRIEVAL_FILENAME = "retrieval_results.json"
DEFAULT_GENERATOR_OUTPUT_FILENAME = "generator_output.json"
DEFAULT_RISK_OUTPUT_FILENAME = "risk_checker_output.json"
DEFAULT_ANSWER_FILENAME = "answer.txt"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_RULES_FILENAME = "risk_rules_v0.2.yaml"
EVIDENCE_RELEVANCE_THRESHOLD = 0.65
MAX_SEMANTIC_VERIFY_RULES = 3

RISK_LEVEL_ORDER = {"unknown": 0, "low": 1, "medium": 2, "high": 3}
NEGATION_RE = re.compile(
    r"\b(no|not|none|without|does not|do not|shall not|will not|不涉及|无|没有|不适用)\b",
    re.IGNORECASE,
)
AMOUNT_RE = re.compile(
    r"(?:(?:USD|US\$|\$|RMB|CNY|¥)\s*)?(\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\s*(million|m|万|百万)?",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    source_file: str
    page_or_clause: str | None
    relevance_score: float


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
    conflict_detected: bool = False
    context_truncated: bool = False
    prompt_template_id: str | None = None


@dataclass(frozen=True)
class RiskRule:
    rule_id: str
    category: str
    description: str
    cuad_category: str | None
    keywords: list[str]
    risk_level: RiskLevel
    require_citation: bool
    human_review_threshold: float | None


@dataclass(frozen=True)
class RuleMatch:
    rule_id: str
    rule_description: str
    matched_text: str | None
    confidence: float
    is_semantic: bool
    category: str
    risk_level: RiskLevel


@dataclass(frozen=True)
class RiskCheckerInput:
    question: str
    generator_output: GeneratorOutput
    retrieved_chunks: list[Chunk]
    route: str
    session_id: str


@dataclass(frozen=True)
class RiskCheckerOutput:
    risk_level: RiskLevel
    triggered_rules: list[RuleMatch]
    rule_coverage_declared: list[str]
    evidence_sufficient: bool
    missing_evidence_hint: str | None
    human_review_required: bool
    human_review_reason: str | None
    latent_human_review_required: bool
    latent_human_review_reason: str | None
    interruption_type: InterruptionType
    final_answer: str
    uncertainty_prefix: str | None
    rules_evaluated_count: int
    bypass_reason: str | None
    semantic_verify_skipped: bool


@dataclass(frozen=True)
class BatchRiskCheckerResult:
    question_file: str
    router_file: str
    retrieval_file: str
    generator_file: str
    answer_file: str
    result_file: str
    risk_level: RiskLevel
    human_review_required: bool
    interruption_type: InterruptionType
    triggered_rule_count: int
    status: str


@dataclass(frozen=True)
class RuleSet:
    rules: list[RiskRule]
    covered_categories: list[str]
    not_covered_note: str | None


class RiskAndEvidenceChecker:
    def __init__(self, rules_path: str | Path | None = None) -> None:
        self._rules_path = Path(rules_path) if rules_path is not None else Path(__file__).parents[1] / DEFAULT_RULES_FILENAME
        self._rule_set = load_rules(self._rules_path)

    def check(self, checker_input: RiskCheckerInput) -> RiskCheckerOutput:
        raw_matches, semantic_verify_skipped = match_rules(
            self._rule_set.rules,
            checker_input.question,
            checker_input.generator_output,
            checker_input.retrieved_chunks,
        )

        if not checker_input.generator_output.answerable:
            return self._check_unanswerable(checker_input, raw_matches, semantic_verify_skipped)

        triggered_rules = raw_matches
        risk_level = highest_risk_level(triggered_rules)
        evidence_sufficient = is_evidence_sufficient(
            checker_input.generator_output,
            checker_input.retrieved_chunks,
        )
        missing_evidence_hint = (
            build_missing_evidence_hint(triggered_rules)
            if not evidence_sufficient
            else None
        )
        uncertainty_prefix = (
            "当前证据覆盖不足，请把以下回答视为需要复核的初步判断。"
            if not evidence_sufficient
            else None
        )
        latent_human_review_required, latent_human_review_reason = decide_latent_human_review(
            triggered_rules,
            checker_input,
            self._rule_set.rules,
        )
        interruption_type = decide_interruption_type(
            risk_level=risk_level,
        )
        final_answer = decorate_answer(
            checker_input.generator_output.answer,
            risk_level=risk_level,
            interruption_type=interruption_type,
            triggered_rules=triggered_rules,
            coverage=self._rule_set.covered_categories,
            missing_evidence_hint=missing_evidence_hint,
            uncertainty_prefix=uncertainty_prefix,
        )

        return RiskCheckerOutput(
            risk_level=risk_level,
            triggered_rules=triggered_rules,
            rule_coverage_declared=self._rule_set.covered_categories,
            evidence_sufficient=evidence_sufficient,
            missing_evidence_hint=missing_evidence_hint,
            human_review_required=False,
            human_review_reason=None,
            latent_human_review_required=latent_human_review_required,
            latent_human_review_reason=latent_human_review_reason,
            interruption_type=interruption_type,
            final_answer=final_answer,
            uncertainty_prefix=uncertainty_prefix,
            rules_evaluated_count=len(self._rule_set.rules),
            bypass_reason=None,
            semantic_verify_skipped=semantic_verify_skipped,
        )

    def _check_unanswerable(
        self,
        checker_input: RiskCheckerInput,
        raw_matches: list[RuleMatch],
        semantic_verify_skipped: bool,
    ) -> RiskCheckerOutput:
        latent_human_review_required, latent_human_review_reason = decide_latent_human_review(
            raw_matches,
            checker_input,
            self._rule_set.rules,
        )

        return RiskCheckerOutput(
            risk_level=highest_risk_level(raw_matches),
            triggered_rules=raw_matches,
            rule_coverage_declared=self._rule_set.covered_categories,
            evidence_sufficient=False,
            missing_evidence_hint=None,
            human_review_required=False,
            human_review_reason=None,
            latent_human_review_required=latent_human_review_required,
            latent_human_review_reason=latent_human_review_reason,
            interruption_type="none",
            final_answer=checker_input.generator_output.answer,
            uncertainty_prefix=None,
            rules_evaluated_count=len(self._rule_set.rules),
            bypass_reason="generator_answerable_false_passthrough",
            semantic_verify_skipped=semantic_verify_skipped,
        )


def check_risk(
    question: str,
    generator_output: GeneratorOutput,
    retrieved_chunks: list[Chunk],
    route: str = "qa",
    session_id: str = "local",
    rules_path: str | Path | None = None,
) -> RiskCheckerOutput:
    return RiskAndEvidenceChecker(rules_path=rules_path).check(
        RiskCheckerInput(
            question=question,
            generator_output=generator_output,
            retrieved_chunks=retrieved_chunks,
            route=route,
            session_id=session_id,
        )
    )


def check_directory(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    question_filename: str = DEFAULT_QUESTION_FILENAME,
    router_filename: str = DEFAULT_ROUTER_FILENAME,
    retrieval_filename: str = DEFAULT_RETRIEVAL_FILENAME,
    generator_filename: str = DEFAULT_GENERATOR_OUTPUT_FILENAME,
    answer_filename: str = DEFAULT_ANSWER_FILENAME,
    rules_filename: str = DEFAULT_RULES_FILENAME,
    session_id: str = "local",
) -> BatchRiskCheckerResult:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    question_path = input_path / question_filename
    router_path = input_path / router_filename
    retrieval_path = input_path / retrieval_filename
    generator_path = input_path / generator_filename
    rules_path = Path(rules_filename)
    if not rules_path.is_absolute():
        rules_path = Path(__file__).parents[1] / rules_filename

    question = _read_optional_text(question_path)
    route = _load_route(router_path)
    retrieved_chunks = load_retrieved_chunks(retrieval_path)
    generator_output = load_generator_output(generator_path)

    status = "ok"
    if not question_path.exists():
        status = "missing_question"
    elif not generator_path.exists():
        status = "missing_generator_output"
    elif not retrieval_path.exists():
        status = "missing_retrieval"

    output = check_risk(
        question=question,
        generator_output=generator_output,
        retrieved_chunks=retrieved_chunks,
        route=route,
        session_id=session_id,
        rules_path=rules_path,
    )

    result_path = output_path / DEFAULT_RISK_OUTPUT_FILENAME
    answer_path = output_path / answer_filename
    manifest_path = output_path / DEFAULT_MANIFEST_FILENAME
    _write_json(result_path, risk_output_payload(output))
    answer_path.write_text(output.final_answer + "\n", encoding="utf-8")

    result = BatchRiskCheckerResult(
        question_file=question_filename,
        router_file=router_filename,
        retrieval_file=retrieval_filename,
        generator_file=generator_filename,
        answer_file=answer_filename,
        result_file=DEFAULT_RISK_OUTPUT_FILENAME,
        risk_level=output.risk_level,
        human_review_required=output.human_review_required,
        interruption_type=output.interruption_type,
        triggered_rule_count=len(output.triggered_rules),
        status=status,
    )
    _write_json(manifest_path, asdict(result))
    return result


def match_rules(
    rules: list[RiskRule],
    question: str,
    generator_output: GeneratorOutput,
    chunks: list[Chunk],
) -> tuple[list[RuleMatch], bool]:
    searchable_items = build_searchable_items(question, generator_output, chunks)
    matches: list[RuleMatch] = []
    high_candidates = 0
    semantic_verify_skipped = False

    for rule in rules:
        matched_text = first_keyword_match(rule.keywords, searchable_items)
        if matched_text is None:
            continue
        confidence = 1.0
        is_semantic = False
        if rule.risk_level == "high":
            high_candidates += 1
            if high_candidates <= MAX_SEMANTIC_VERIFY_RULES:
                is_semantic = True
                if not semantic_verify(rule, matched_text, question):
                    continue
                confidence = 0.85
            else:
                semantic_verify_skipped = True
        matches.append(
            RuleMatch(
                rule_id=rule.rule_id,
                rule_description=rule.description,
                matched_text=matched_text,
                confidence=confidence,
                is_semantic=is_semantic,
                category=rule.category,
                risk_level=rule.risk_level,
            )
        )
    return matches, semantic_verify_skipped


def build_searchable_items(
    question: str,
    generator_output: GeneratorOutput,
    chunks: list[Chunk],
) -> list[str]:
    items = [question, generator_output.answer]
    items.extend(citation.supporting_text for citation in generator_output.citations)
    items.extend(chunk.text for chunk in chunks)
    return [item for item in items if item]


def first_keyword_match(keywords: list[str], texts: list[str]) -> str | None:
    lowered_keywords = [keyword.lower() for keyword in keywords]
    for text in texts:
        lowered = text.lower()
        if any(keyword and keyword in lowered for keyword in lowered_keywords):
            return text[:700].strip()
    return None


def semantic_verify(rule: RiskRule, matched_text: str, question: str) -> bool:
    window = matched_text[:260]
    keyword_count = sum(1 for keyword in rule.keywords if keyword.lower() in matched_text.lower())
    if keyword_count >= 2:
        return True
    combined = f"{question} {window}".lower()
    if NEGATION_RE.search(combined) and keyword_count == 1:
        return False
    return True


def is_evidence_sufficient(generator_output: GeneratorOutput, chunks: list[Chunk]) -> bool:
    top_score = max((chunk.relevance_score for chunk in chunks), default=0.0)
    return top_score >= EVIDENCE_RELEVANCE_THRESHOLD and len(generator_output.citations) > 0


def highest_risk_level(matches: list[RuleMatch]) -> RiskLevel:
    if not matches:
        return "unknown"
    return max((match.risk_level for match in matches), key=lambda level: RISK_LEVEL_ORDER[level])


def decide_latent_human_review(
    matches: list[RuleMatch],
    checker_input: RiskCheckerInput,
    rules: list[RiskRule],
) -> tuple[bool, str | None]:
    threshold_reasons = threshold_review_reasons(matches, checker_input, rules)
    if threshold_reasons:
        return True, "; ".join(threshold_reasons)

    high_matches = [match for match in matches if match.risk_level == "high"]
    if len(high_matches) >= 2:
        rule_ids = ", ".join(match.rule_id for match in high_matches)
        return True, f"多条高风险规则同时命中（{rule_ids}），需人工确认"

    if highest_risk_level(matches) == "high" and not checker_input.generator_output.answerable:
        return True, "高风险问题无法从文档中找到答案，需人工确认"

    return False, None


def threshold_review_reasons(
    matches: list[RuleMatch],
    checker_input: RiskCheckerInput,
    rules: list[RiskRule],
) -> list[str]:
    reasons: list[str] = []
    text = " ".join(build_searchable_items(
        checker_input.question,
        checker_input.generator_output,
        checker_input.retrieved_chunks,
    ))
    amounts = extract_amounts(text)
    rule_by_id = {rule.rule_id: rule for rule in rules}
    for match in matches:
        rule = rule_by_id.get(match.rule_id)
        if rule is None or rule.human_review_threshold is None:
            continue
        exceeded = [amount for amount in amounts if amount > rule.human_review_threshold]
        if exceeded:
            reasons.append(
                f"{match.rule_id} 金额 {max(exceeded):.0f} 超过人工复核阈值 {rule.human_review_threshold:.0f}"
            )
    return reasons


def extract_amounts(text: str) -> list[float]:
    amounts: list[float] = []
    for match in AMOUNT_RE.finditer(text):
        raw_value = match.group(1).replace(",", "")
        try:
            value = float(raw_value)
        except ValueError:
            continue
        suffix = (match.group(2) or "").lower()
        if suffix in {"million", "m", "百万"}:
            value *= 1_000_000
        elif suffix == "万":
            value *= 10_000
        amounts.append(value)
    return amounts


def decide_interruption_type(
    risk_level: RiskLevel,
) -> InterruptionType:
    if risk_level == "high":
        return "non_blocking"
    return "none"


def build_missing_evidence_hint(matches: list[RuleMatch]) -> str | None:
    categories = [match.category for match in matches]
    if not categories:
        return "还缺少可支撑该回答的高相关条款原文或 Citation。"
    hints = {
        "payment_risk": "还缺少明确的付款金额、付款期限或付款条件的条款原文。",
        "termination_risk": "还缺少明确的终止条件、通知期限或解除后果的条款原文。",
        "liability_cap": "还缺少责任上限金额、适用范围或例外情形的条款原文。",
        "non_compete": "还缺少竞业限制对象、期限、地域或例外情形的条款原文。",
        "anti_assignment": "还缺少转让限制、同意要求或允许转让例外的条款原文。",
    }
    return " ".join(dict.fromkeys(hints.get(category, "还缺少对应风险类别的条款原文。") for category in categories))


def decorate_answer(
    answer: str,
    risk_level: RiskLevel,
    interruption_type: InterruptionType,
    triggered_rules: list[RuleMatch],
    coverage: list[str],
    missing_evidence_hint: str | None,
    uncertainty_prefix: str | None,
) -> str:
    if interruption_type == "none" and not uncertainty_prefix:
        return answer

    lines: list[str] = []
    if interruption_type == "non_blocking":
        lines.append("【高风险提示】此回答命中高风险规则，流程继续，但请结合依据复核。")
    if uncertainty_prefix:
        lines.append(f"证据提示：{uncertainty_prefix}")
    if missing_evidence_hint:
        lines.append(f"缺失信息：{missing_evidence_hint}")
    if triggered_rules:
        rule_labels = ", ".join(f"{match.rule_id}:{match.rule_description}" for match in triggered_rules)
        lines.append(f"命中规则：{rule_labels}")
    if coverage:
        lines.append(f"本次风险判断覆盖类别：{', '.join(coverage)}")
    lines.append("")
    lines.append(answer)
    return "\n".join(lines)


def load_rules(path: str | Path) -> RuleSet:
    rules_path = Path(path)
    text = rules_path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ImportError:
        payload = parse_limited_rules_yaml(text)
    else:
        payload = yaml.safe_load(text)

    rules = [
        RiskRule(
            rule_id=str(row["id"]),
            category=str(row["category"]),
            description=str(row["description"]),
            cuad_category=str(row["cuad_category"]) if row.get("cuad_category") is not None else None,
            keywords=[str(value) for value in row.get("keywords", [])],
            risk_level=normalize_risk_level(str(row.get("risk_level", "unknown"))),
            require_citation=bool(row.get("require_citation", False)),
            human_review_threshold=(
                float(row["human_review_threshold"])
                if row.get("human_review_threshold") is not None
                else None
            ),
        )
        for row in payload.get("rules", [])
    ]
    coverage = payload.get("coverage_declaration", {})
    return RuleSet(
        rules=rules,
        covered_categories=[str(value) for value in coverage.get("covered_categories", [])],
        not_covered_note=coverage.get("not_covered_note"),
    )


def parse_limited_rules_yaml(text: str) -> dict[str, object]:
    payload: dict[str, object] = {"rules": [], "coverage_declaration": {"covered_categories": []}}
    section = None
    current_rule: dict[str, object] | None = None
    in_covered_categories = False

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "rules:":
            section = "rules"
            in_covered_categories = False
            continue
        if stripped == "coverage_declaration:":
            section = "coverage"
            in_covered_categories = False
            continue
        if section == "rules" and stripped.startswith("- id:"):
            current_rule = {"id": stripped.split(":", 1)[1].strip()}
            payload["rules"].append(current_rule)  # type: ignore[union-attr]
            continue
        if section == "rules" and current_rule is not None and ":" in stripped:
            key, value = stripped.split(":", 1)
            current_rule[key.strip()] = parse_scalar(value.strip())
            continue
        coverage = payload["coverage_declaration"]  # type: ignore[assignment]
        if section == "coverage" and stripped == "covered_categories:":
            in_covered_categories = True
            continue
        if section == "coverage" and in_covered_categories and stripped.startswith("- "):
            coverage["covered_categories"].append(stripped[2:].strip())  # type: ignore[index]
            continue
        if section == "coverage" and ":" in stripped:
            key, value = stripped.split(":", 1)
            coverage[key.strip()] = parse_scalar(value.strip())  # type: ignore[index]
            in_covered_categories = False
    return payload


def parse_scalar(value: str) -> object:
    if value == "null":
        return None
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith("[") and value.endswith("]"):
        return [item.strip().strip('"').strip("'") for item in value[1:-1].split(",") if item.strip()]
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def normalize_risk_level(value: str) -> RiskLevel:
    if value in {"high", "medium", "low", "unknown"}:
        return value  # type: ignore[return-value]
    return "unknown"


def load_generator_output(path: str | Path) -> GeneratorOutput:
    result_path = Path(path)
    if not result_path.exists():
        return GeneratorOutput(answer="", citations=[], answerable=False, confidence=0.0)
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    citations = [
        Citation(
            chunk_id=str(row.get("chunk_id", "")),
            claim=str(row.get("claim", "")),
            supporting_text=str(row.get("supporting_text", "")),
        )
        for row in payload.get("citations", [])
        if isinstance(row, dict)
    ]
    return GeneratorOutput(
        answer=str(payload.get("answer", "")),
        citations=citations,
        answerable=bool(payload.get("answerable", False)),
        confidence=float(payload.get("confidence", 0.0) or 0.0),
        conflict_detected=bool(payload.get("conflict_detected", False)),
        context_truncated=bool(payload.get("context_truncated", False)),
        prompt_template_id=payload.get("prompt_template_id"),
    )


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
        return []

    chunks: list[Chunk] = []
    for index, row in enumerate(payload):
        if not isinstance(row, dict):
            continue
        score = float(row.get("score", row.get("relevance_score", 0.0)) or 0.0)
        nested = row.get("chunk")
        if isinstance(nested, dict):
            source_file = str(nested.get("source_file", "unknown"))
            location = nested.get("location") or nested.get("page_or_clause") or nested.get("clause_id")
            chunk_index = nested.get("chunk_index", index)
            chunks.append(
                Chunk(
                    chunk_id=str(nested.get("chunk_id") or f"{source_file}#{chunk_index}"),
                    text=str(nested.get("text", "")),
                    source_file=source_file,
                    page_or_clause=str(location) if location is not None else None,
                    relevance_score=score,
                )
            )
        elif "text" in row:
            chunks.append(
                Chunk(
                    chunk_id=str(row.get("chunk_id") or f"{row.get('source_file', 'unknown')}#{index}"),
                    text=str(row["text"]),
                    source_file=str(row.get("source_file", "unknown")),
                    page_or_clause=str(row["page_or_clause"]) if row.get("page_or_clause") is not None else None,
                    relevance_score=score,
                )
            )
    return chunks


def risk_output_payload(output: RiskCheckerOutput) -> dict[str, object]:
    return {
        "risk_level": output.risk_level,
        "triggered_rules": [asdict(match) for match in output.triggered_rules],
        "rule_coverage_declared": output.rule_coverage_declared,
        "evidence_sufficient": output.evidence_sufficient,
        "missing_evidence_hint": output.missing_evidence_hint,
        "human_review_required": output.human_review_required,
        "human_review_reason": output.human_review_reason,
        "latent_human_review_required": output.latent_human_review_required,
        "latent_human_review_reason": output.latent_human_review_reason,
        "interruption_type": output.interruption_type,
        "final_answer": output.final_answer,
        "uncertainty_prefix": output.uncertainty_prefix,
        "rules_evaluated_count": output.rules_evaluated_count,
        "bypass_reason": output.bypass_reason,
        "semantic_verify_skipped": output.semantic_verify_skipped,
    }


def _load_route(path: Path) -> str:
    if not path.exists():
        return "qa"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "qa"
    return str(payload.get("route", "qa"))


def _read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff").strip()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
