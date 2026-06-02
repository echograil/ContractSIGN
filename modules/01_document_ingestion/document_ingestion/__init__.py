"""Document ingestion public API."""

from .ingestion import BatchIngestResult, TextChunk, ingest, ingest_directory

__all__ = ["BatchIngestResult", "TextChunk", "ingest", "ingest_directory"]
