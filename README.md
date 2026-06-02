# ContractSIGN v0.2

Six-module contract QA pipeline with folder-based `input/` -> `output/` handoff and a local HTML UI.

## What Is Included

- `modules/01_document_ingestion`: PDF ingestion, writes `chunks.json`
- `modules/02_retrieval`: hybrid retrieval, writes `retrieval_results.json`
- `modules/03_question_router`: question routing, writes `router_output.json`
- `modules/04_answer_generator`: grounded answer generation, writes `generator_output.json`
- `modules/05_risk_checker`: risk and evidence pass, writes final `answer.txt`
- `modules/06_observability`: SQLite trace store and anonymized eval export
- `scripts/pipeline_runner.py`: moves files through module `input/` and `output/` folders
- `scripts/web_server.py`: local HTML upload/question UI
- `web/index.html`: visual conveyor-style pipeline interface

## Quick Start

```powershell
python -m pip install -r requirements.txt
python scripts/web_server.py
```

Open:

```text
http://127.0.0.1:8765
```

Upload a PDF contract, enter a question, and run the pipeline.

## GitHub Pages

The static publishing page is in `docs/index.html`. It can be used as a GitHub
Pages site to explain the current user experience, module progress, and setup
instructions.

GitHub Pages cannot run the real PDF pipeline because it cannot execute Python,
receive uploaded files on a server, or write module `input/` and `output/`
folders. Use the local web server for the actual upload-and-answer workflow.

## API Key

Default mode can run locally without an API key. If you enable API routing,
API embeddings, or API generation in the UI, copy `.env.example` to `.env` and
fill in your own key. Then install optional API dependencies:

```powershell
python -m pip install -r requirements-api.txt
```

`.env` example:

```text
OPENAI_API_KEY=your_api_key_here
```

Do not commit `.env`.

## CLI

```powershell
python scripts/pipeline_runner.py --pdf path\to\contract.pdf --question "What is the payment term?"
```

Optional API flags:

```powershell
python scripts/pipeline_runner.py --pdf contract.pdf --question "Summarize termination rights" --use-api-generator
```

## File Flow

Each run uses the visible module folders:

```text
modules/01_document_ingestion/input  -> modules/01_document_ingestion/output
modules/02_retrieval/input           -> modules/02_retrieval/output
modules/03_question_router/input     -> modules/03_question_router/output
modules/04_answer_generator/input    -> modules/04_answer_generator/output
modules/05_risk_checker/input        -> modules/05_risk_checker/output
modules/06_observability/input       -> modules/06_observability/output
```

The runner clears module `input/` and `output/` folders at the beginning of each
run, then copies artifacts forward so the file movement matches the UI.

## Chinese Questions

Local retrieval includes a bilingual legal-term expansion layer for common
Chinese contract questions, including summary, payment, termination, assignment,
confidentiality, IP, non-compete, audit, notice, governing law, dispute, force
majeure, delivery, obligations, and warranty. If a Chinese query has no known
legal terms, the retriever still falls back to general contract context so the
pipeline does not fail with an empty retrieval result.

For open-ended cross-language semantic matching beyond this local dictionary,
enable API Embeddings and configure your own API key.

## Publish Notes

This folder is intended to be zipped or pushed to GitHub as-is. Runtime outputs,
uploaded PDFs, SQLite databases, and `.env` are ignored by `.gitignore`.
