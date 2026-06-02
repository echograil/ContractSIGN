from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import logging
import math
from pathlib import Path
import re
from typing import Protocol

logger = logging.getLogger(__name__)

VALID_DOC_TYPES = frozenset({"contract", "quote", "faq", "product", "unknown"})
STRATEGY_TAG = "hybrid_v0"
VECTOR_DIMENSIONS = 1536
VECTOR_QUERY_TOKEN_LIMIT = 512
KEYWORD_QUERY_TERM_LIMIT = 128
RRF_K = 60
DEFAULT_CANDIDATE_FLOOR = 50
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_QUESTION_FILENAME = "question.txt"
DEFAULT_ANSWER_FILENAME = "answer.txt"
DEFAULT_RESULTS_FILENAME = "retrieval_results.json"
DEFAULT_MANIFEST_FILENAME = "manifest.json"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_OPENAI_BASE_URL = "https://aihubmix.com/v1"
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
        "does",
        "for",
        "from",
        "if",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "there",
        "this",
        "to",
        "under",
        "what",
        "which",
        "with",
    }
)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*", re.IGNORECASE)
CJK_RE = re.compile("[\u4e00-\u9fff]")
CHINESE_QUERY_EXPANSIONS: list[tuple[re.Pattern[str], tuple[str, ...]]] = [
    (
        re.compile("\u4e3b\u8981\u5185\u5bb9|\u603b\u7ed3|\u6982\u62ec|\u6458\u8981|\u8bb2\u4e86\u4ec0\u4e48|\u5408\u540c\u5185\u5bb9|\u8981\u70b9|\u6838\u5fc3\u5185\u5bb9"),
        ("agreement", "contract", "cooperation", "party", "term", "service", "obligation", "scope", "project"),
    ),
    (
        re.compile("\u4ed8\u6b3e|\u8d39\u7528|\u4ef7\u6b3e|\u652f\u4ed8|\u62a5\u916c|\u670d\u52a1\u8d39|\u91d1\u989d|\u6ede\u7eb3\u91d1|\u8fdd\u7ea6\u91d1"),
        ("payment", "fee", "compensation", "pay", "amount", "service fee", "penalty", "overdue"),
    ),
    (
        re.compile("\u7ec8\u6b62|\u89e3\u9664|\u5230\u671f|\u671f\u9650|\u6709\u6548\u671f|\u5408\u540c\u671f|\u7eed\u7ea6"),
        ("terminate", "termination", "expire", "expiration", "term", "duration", "notice", "renewal"),
    ),
    (
        re.compile("\u98ce\u9669|\u8d23\u4efb|\u8fdd\u7ea6|\u8d54\u507f|\u635f\u5bb3|\u8865\u507f|\u4e0d\u627f\u62c5|\u8d23\u4efb\u4e0a\u9650"),
        ("risk", "liability", "breach", "damages", "indemnify", "compensation", "limitation", "cap"),
    ),
    (
        re.compile("\u8f6c\u8ba9|\u8ba9\u6e21|\u8f6c\u79fb|\u540c\u610f"),
        ("assign", "assignment", "transfer", "consent", "prior written consent"),
    ),
    (
        re.compile("\u4fdd\u5bc6|\u673a\u5bc6|\u4e0d\u62ab\u9732|\u5546\u4e1a\u79d8\u5bc6"),
        ("confidential", "confidentiality", "non-disclosure", "trade secret", "proprietary"),
    ),
    (
        re.compile("\u77e5\u8bc6\u4ea7\u6743|\u8457\u4f5c\u6743|\u5546\u6807|\u4e13\u5229|\u8bb8\u53ef|\u6388\u6743"),
        ("intellectual property", "copyright", "trademark", "patent", "license", "licensed", "right"),
    ),
    (
        re.compile("\u7ade\u4e1a|\u7981\u6b62\u7ade\u4e89|\u4e0d\u62db\u63fd|\u975e\u62db\u63fd|\u72ec\u5bb6|\u6392\u4ed6"),
        ("non-compete", "non-solicit", "solicit", "exclusive", "exclusivity", "competition"),
    ),
    (
        re.compile("\u5ba1\u8ba1|\u68c0\u67e5|\u8bb0\u5f55|\u8d26\u672c|\u62a5\u544a"),
        ("audit", "inspect", "inspection", "record", "books", "report"),
    ),
    (
        re.compile("\u901a\u77e5|\u4e66\u9762\u901a\u77e5|\u63d0\u524d\u901a\u77e5"),
        ("notice", "written notice", "notify", "days", "address"),
    ),
    (
        re.compile("\u9002\u7528\u6cd5|\u7ba1\u8f96|\u4e89\u8bae|\u4ef2\u88c1|\u8bc9\u8bbc|\u6cd5\u9662"),
        ("governing law", "law", "jurisdiction", "dispute", "arbitration", "court", "venue"),
    ),
    (
        re.compile("\u4e0d\u53ef\u6297\u529b|\u610f\u5916|\u707e\u5bb3"),
        ("force majeure", "act of god", "disaster", "beyond control"),
    ),
    (
        re.compile("\u4ea4\u4ed8|\u5c65\u884c|\u4e49\u52a1|\u670d\u52a1\u8303\u56f4|\u5de5\u4f5c\u5185\u5bb9"),
        ("deliver", "delivery", "performance", "obligation", "service", "scope", "work"),
    ),
    (
        re.compile("\u4fdd\u8bc1|\u9648\u8ff0|\u627f\u8bfa|\u4fdd\u969c"),
        ("warranty", "representation", "covenant", "undertaking", "guarantee"),
    ),
]


@dataclass(frozen=True)
class TextChunk:
    text: str
    source_file: str
    location: str
    doc_type: str
    chunk_index: int
    clause_id: str | None = None


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: TextChunk
    score: float
    strategy_tag: str


@dataclass(frozen=True)
class _IndexedChunk:
    chunk: TextChunk
    vector: "Vector"
    keyword_terms: Counter[str]
    keyword_length: int


Vector = dict[int, float] | list[float]


class _VectorBackend(Protocol):
    def embed_documents(self, texts: list[str]) -> list[Vector]:
        ...

    def embed_query(self, text: str) -> Vector:
        ...


class _HashingVectorBackend:
    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [_hashing_vector(text) for text in texts]

    def embed_query(self, text: str) -> Vector:
        return _hashing_vector(text)


class _ApiEmbeddingVectorBackend:
    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
    ) -> None:
        _load_dotenv_if_available()
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise RuntimeError("langchain_openai is required for API embeddings") from exc

        self._embeddings = OpenAIEmbeddings(model=model, base_url=base_url)

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [_normalize_dense_vector(vector) for vector in self._embeddings.embed_documents(texts)]

    def embed_query(self, text: str) -> Vector:
        return _normalize_dense_vector(self._embeddings.embed_query(text))


class RetrievalModule:
    """In-memory hybrid retriever implementing Spec V0.2.

    The vector path uses a deterministic hashing vectorizer so the module can
    run in local tests without network credentials. It preserves the interface
    and ranking mechanics expected by the production embedding-backed version.
    """

    def __init__(
        self,
        use_api_embeddings: bool = False,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        embedding_base_url: str = DEFAULT_OPENAI_BASE_URL,
    ) -> None:
        self._items: list[_IndexedChunk] = []
        self._doc_freq: Counter[str] = Counter()
        self._total_keyword_length = 0
        self._vector_backend: _VectorBackend = _build_vector_backend(
            use_api_embeddings=use_api_embeddings,
            embedding_model=embedding_model,
            embedding_base_url=embedding_base_url,
        )

    def index(self, chunks: list[TextChunk]) -> None:
        if not chunks:
            return

        clean_texts = [chunk.text.strip() for chunk in chunks]
        vectors = self._vector_backend.embed_documents(clean_texts)

        for chunk, clean_text, vector in zip(chunks, clean_texts, vectors, strict=True):
            if chunk.doc_type not in VALID_DOC_TYPES:
                logger.warning("Unknown doc_type passed through: %s", chunk.doc_type)

            keyword_terms = Counter(_keyword_terms(clean_text))
            item = _IndexedChunk(
                chunk=chunk,
                vector=vector,
                keyword_terms=keyword_terms,
                keyword_length=sum(keyword_terms.values()),
            )
            self._items.append(item)
            self._doc_freq.update(keyword_terms.keys())
            self._total_keyword_length += item.keyword_length

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filter_doc_type: str | None = None,
        filter_source_file: str | None = None,
    ) -> list[RetrievedChunk]:
        if not self._items or top_k <= 0:
            return []

        query = query.strip()
        if not query:
            return []

        candidate_limit = max(DEFAULT_CANDIDATE_FLOOR, top_k * 10)
        vector_query = _truncate_vector_query(query)
        keyword_query_terms = _truncate_keyword_terms(_keyword_terms(query))

        vector_ranked = _rank_vector(
            items=self._items,
            query_vector=self._vector_backend.embed_query(vector_query),
            limit=candidate_limit,
        )
        keyword_ranked = self._rank_keyword(keyword_query_terms, candidate_limit)
        if not vector_ranked and not keyword_ranked and CJK_RE.search(query):
            logger.info("Chinese query produced no local retrieval terms; falling back to leading chunks")
            return _leading_results(self._items, top_k)
        fused_scores = _rrf_fuse(vector_ranked, keyword_ranked)

        ranked = sorted(fused_scores.items(), key=lambda pair: pair[1], reverse=True)
        filtered: list[tuple[int, float]] = []
        seen_keys: set[tuple[str, int]] = set()
        for index, raw_score in ranked:
            chunk = self._items[index].chunk
            if filter_doc_type is not None and chunk.doc_type != filter_doc_type:
                continue
            if filter_source_file is not None and chunk.source_file != filter_source_file:
                continue

            key = (chunk.source_file, chunk.chunk_index)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            filtered.append((index, raw_score))
            if len(filtered) >= top_k:
                break

        if not filtered:
            return []

        normalized_scores = _normalize_scores([score for _, score in filtered])
        return [
            RetrievedChunk(
                chunk=self._items[index].chunk,
                score=score,
                strategy_tag=STRATEGY_TAG,
            )
            for (index, _), score in zip(filtered, normalized_scores, strict=True)
        ]

    def _rank_keyword(self, query_terms: list[str], limit: int) -> list[int]:
        if not query_terms:
            return []

        average_length = self._average_keyword_length()
        scored: list[tuple[int, float]] = []
        for index, item in enumerate(self._items):
            score = _bm25_score(
                query_terms=query_terms,
                document_terms=item.keyword_terms,
                document_length=item.keyword_length,
                average_document_length=average_length,
                document_count=len(self._items),
                document_frequency=self._doc_freq,
            )
            if score > 0:
                scored.append((index, score))

        scored.sort(key=lambda pair: pair[1], reverse=True)
        return [index for index, _ in scored[:limit]]

    def _average_keyword_length(self) -> float:
        if not self._items:
            return 0.0
        return self._total_keyword_length / len(self._items)


@dataclass(frozen=True)
class BatchRetrievalResult:
    question_file: str
    answer_file: str
    result_file: str
    chunk_count: int
    result_count: int
    status: str


def retrieve_directory(
    input_dir: str | Path = DEFAULT_INPUT_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    question_filename: str = DEFAULT_QUESTION_FILENAME,
    answer_filename: str = DEFAULT_ANSWER_FILENAME,
    top_k: int = 5,
    use_api_embeddings: bool = False,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedding_base_url: str = DEFAULT_OPENAI_BASE_URL,
) -> BatchRetrievalResult:
    """Run a user-style retrieval smoke test from folders.

    Expected input:
    - input/chunks.json from DocumentIngestion output, or one or more
      input/*.chunks.json files if chunks.json is absent.
    - input/question.txt containing the user's natural-language question.

    Written output:
    - output/answer.txt as a readable retrieval answer.
    - output/retrieval_results.json as structured results for debugging.
    - output/manifest.json as batch status.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    question_path = input_path / question_filename
    answer_path = output_path / answer_filename
    result_path = output_path / DEFAULT_RESULTS_FILENAME
    manifest_path = output_path / DEFAULT_MANIFEST_FILENAME

    if not question_path.exists():
        result = BatchRetrievalResult(
            question_file=question_filename,
            answer_file=answer_filename,
            result_file=DEFAULT_RESULTS_FILENAME,
            chunk_count=0,
            result_count=0,
            status="missing_question",
        )
        _write_text_answer(answer_path, "未找到问题文件：input/question.txt\n")
        _write_json(result_path, [])
        _write_json(manifest_path, asdict(result))
        logger.warning("Question file does not exist: %s", question_path)
        return result

    chunks = load_chunks_from_directory(input_path)
    question = question_path.read_text(encoding="utf-8", errors="replace").lstrip("\ufeff").strip()
    module = RetrievalModule(
        use_api_embeddings=use_api_embeddings,
        embedding_model=embedding_model,
        embedding_base_url=embedding_base_url,
    )
    module.index(chunks)
    results = module.retrieve(question, top_k=top_k)

    result_payload = [_retrieved_chunk_payload(result) for result in results]
    _write_text_answer(answer_path, _format_answer(question, results))
    _write_json(result_path, result_payload)

    status = "ok"
    if not chunks:
        status = "empty_index"
    elif not question:
        status = "empty_question"

    batch_result = BatchRetrievalResult(
        question_file=question_filename,
        answer_file=answer_filename,
        result_file=DEFAULT_RESULTS_FILENAME,
        chunk_count=len(chunks),
        result_count=len(results),
        status=status,
    )
    _write_json(manifest_path, asdict(batch_result))
    return batch_result


def load_chunks_from_directory(input_dir: str | Path) -> list[TextChunk]:
    input_path = Path(input_dir)
    if not input_path.exists() or not input_path.is_dir():
        logger.warning("Input directory does not exist: %s", input_path)
        return []

    aggregate_path = input_path / "chunks.json"
    if aggregate_path.exists():
        return load_chunks_from_file(aggregate_path)

    chunks: list[TextChunk] = []
    for chunk_path in sorted(input_path.glob("*.chunks.json")):
        chunks.extend(load_chunks_from_file(chunk_path))
    return chunks


def load_chunks_from_file(path: str | Path) -> list[TextChunk]:
    chunk_path = Path(path)
    try:
        payload = json.loads(chunk_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Cannot read chunks file %s: %s", chunk_path, exc)
        return []

    if not isinstance(payload, list):
        logger.warning("Chunks file is not a list: %s", chunk_path)
        return []

    chunks: list[TextChunk] = []
    for row in payload:
        if not isinstance(row, dict):
            logger.warning("Skipping non-object chunk row in %s", chunk_path)
            continue
        try:
            chunks.append(
                TextChunk(
                    text=str(row["text"]),
                    source_file=str(row["source_file"]),
                    location=str(row["location"]),
                    doc_type=str(row["doc_type"]),
                    chunk_index=int(row["chunk_index"]),
                    clause_id=row.get("clause_id"),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Skipping invalid chunk row in %s: %s", chunk_path, exc)
    return chunks


def _rank_vector(items: list[_IndexedChunk], query_vector: Vector, limit: int) -> list[int]:
    if not query_vector:
        return []

    scored = [
        (index, score)
        for index, item in enumerate(items)
        if (score := _dot(query_vector, item.vector)) > 0
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [index for index, _ in scored[:limit]]


def _rrf_fuse(*rankings: list[int]) -> dict[int, float]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, index in enumerate(ranking, start=1):
            scores[index] = scores.get(index, 0.0) + 1.0 / (RRF_K + rank)
    return scores


def _bm25_score(
    query_terms: list[str],
    document_terms: Counter[str],
    document_length: int,
    average_document_length: float,
    document_count: int,
    document_frequency: Counter[str],
) -> float:
    if not document_terms or average_document_length <= 0:
        return 0.0

    k1 = 1.5
    b = 0.75
    score = 0.0
    for term in query_terms:
        term_frequency = document_terms.get(term, 0)
        if term_frequency == 0:
            continue

        docs_with_term = document_frequency.get(term, 0)
        idf = math.log(1.0 + (document_count - docs_with_term + 0.5) / (docs_with_term + 0.5))
        denominator = term_frequency + k1 * (
            1.0 - b + b * document_length / average_document_length
        )
        score += idf * (term_frequency * (k1 + 1.0)) / denominator

    return score


def _normalize_scores(scores: list[float]) -> list[float]:
    if len(scores) == 1:
        return [1.0]

    minimum = min(scores)
    maximum = max(scores)
    if math.isclose(maximum, minimum):
        return [1.0 for _ in scores]

    return [round((score - minimum) / (maximum - minimum), 6) for score in scores]


def _truncate_vector_query(query: str) -> str:
    tokens = TOKEN_RE.findall(query)
    if len(tokens) <= VECTOR_QUERY_TOKEN_LIMIT:
        return query

    logger.warning("Vector query truncated to %s tokens", VECTOR_QUERY_TOKEN_LIMIT)
    return " ".join(tokens[:VECTOR_QUERY_TOKEN_LIMIT])


def _truncate_keyword_terms(terms: list[str]) -> list[str]:
    if len(terms) <= KEYWORD_QUERY_TERM_LIMIT:
        return terms

    logger.warning("Keyword query truncated to %s terms", KEYWORD_QUERY_TERM_LIMIT)
    return terms[:KEYWORD_QUERY_TERM_LIMIT]


def _keyword_terms(text: str) -> list[str]:
    terms = list(_chinese_query_expansions(text))
    terms.extend(match.group(0).lower() for match in TOKEN_RE.finditer(text))
    normalized: list[str] = []
    for term in terms:
        normalized_term = _normalize_term(term)
        if normalized_term in STOP_WORDS:
            continue
        normalized.append(normalized_term)
        if "-" in normalized_term:
            normalized.extend(
                part for part in normalized_term.split("-") if part and part not in STOP_WORDS
            )
    return normalized


def _chinese_query_expansions(text: str) -> list[str]:
    if not CJK_RE.search(text):
        return []
    expansions: list[str] = []
    for pattern, terms in CHINESE_QUERY_EXPANSIONS:
        if pattern.search(text):
            expansions.extend(terms)
    if not expansions:
        expansions.extend(("agreement", "contract", "party", "obligation"))
    return expansions


def _leading_results(items: list[_IndexedChunk], top_k: int) -> list[RetrievedChunk]:
    selected = items[:top_k]
    count = len(selected)
    if count == 0:
        return []
    return [
        RetrievedChunk(
            chunk=item.chunk,
            score=round(max(0.1, 1.0 - index * 0.08), 6),
            strategy_tag=f"{STRATEGY_TAG}_leading_fallback",
        )
        for index, item in enumerate(selected)
    ]


def _normalize_term(term: str) -> str:
    if "-" in term:
        return "-".join(_normalize_term(part) for part in term.split("-"))
    if len(term) > 5 and term.endswith("ing"):
        return term[:-3]
    if len(term) > 4 and term.endswith("ed"):
        return term[:-2]
    if len(term) > 3 and term.endswith("s"):
        return term[:-1]
    return term


def _build_vector_backend(
    use_api_embeddings: bool,
    embedding_model: str,
    embedding_base_url: str,
) -> _VectorBackend:
    if not use_api_embeddings:
        return _HashingVectorBackend()

    try:
        return _ApiEmbeddingVectorBackend(model=embedding_model, base_url=embedding_base_url)
    except Exception as exc:
        logger.warning("API embeddings unavailable, falling back to local hashing: %s", exc)
        return _HashingVectorBackend()


def _load_dotenv_if_available() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _normalize_dense_vector(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return []
    return [value / norm for value in vector]


def _hashing_vector(text: str) -> dict[int, float]:
    counts: dict[int, float] = {}
    for term in _keyword_terms(text):
        digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % VECTOR_DIMENSIONS
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        counts[bucket] = counts.get(bucket, 0.0) + sign

    norm = math.sqrt(sum(value * value for value in counts.values()))
    if norm == 0:
        return {}

    return {bucket: value / norm for bucket, value in counts.items()}


def _dot(left: Vector, right: Vector) -> float:
    if isinstance(left, dict) and isinstance(right, dict):
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(bucket, 0.0) for bucket, value in left.items())
    if isinstance(left, list) and isinstance(right, list):
        return sum(left_value * right_value for left_value, right_value in zip(left, right, strict=False))
    return 0.0


def _format_answer(question: str, results: list[RetrievedChunk]) -> str:
    lines = [
        "问题：",
        question or "<empty>",
        "",
        f"检索结果：{len(results)} 条",
    ]
    if not results:
        lines.append("未检索到相关片段。")
        return "\n".join(lines) + "\n"

    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        lines.extend(
            [
                "",
                f"[{index}] score={result.score:.6f} strategy={result.strategy_tag}",
                f"source_file={chunk.source_file}",
                f"location={chunk.location}",
                f"doc_type={chunk.doc_type}",
                f"chunk_index={chunk.chunk_index}",
                "text:",
                chunk.text,
            ]
        )
    return "\n".join(lines) + "\n"


def _retrieved_chunk_payload(result: RetrievedChunk) -> dict[str, object]:
    return {
        "score": result.score,
        "strategy_tag": result.strategy_tag,
        "chunk": asdict(result.chunk),
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text_answer(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
