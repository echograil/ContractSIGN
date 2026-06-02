from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import traceback
from urllib.parse import urlparse

try:
    from pipeline_runner import RELEASE_ROOT, WORKSPACE_ROOT, run_pipeline
except ModuleNotFoundError as exc:
    missing = exc.name or "required package"
    print(f"Missing dependency: {missing}")
    print("Install base dependencies first:")
    print("  python -m pip install -r requirements.txt")
    print("If you enable API features, also run:")
    print("  python -m pip install -r requirements-api.txt")
    raise SystemExit(1) from exc


HOST = "127.0.0.1"
PORT = 8765
WEB_ROOT = RELEASE_ROOT / "web"
UPLOAD_ROOT = WORKSPACE_ROOT / "input"


class ContractSignHandler(BaseHTTPRequestHandler):
    server_version = "ContractSIGNLocal/0.2"

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self.serve_file(WEB_ROOT / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/web/"):
            target = WEB_ROOT / path.removeprefix("/web/")
            content_type = "text/plain; charset=utf-8"
            if target.suffix == ".css":
                content_type = "text/css; charset=utf-8"
            elif target.suffix == ".js":
                content_type = "application/javascript; charset=utf-8"
            self.serve_file(target, content_type)
            return
        self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/api/run":
            self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return

        try:
            fields, files = parse_multipart(self)
            question = fields.get("question", "").strip()
            upload = files.get("pdf")
            if upload is None:
                self.send_json({"error": "Please upload a PDF contract."}, HTTPStatus.BAD_REQUEST)
                return
            if not question:
                self.send_json({"error": "Please enter a question."}, HTTPStatus.BAD_REQUEST)
                return

            UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
            filename = safe_filename(upload["filename"] or "contract.pdf")
            pdf_path = UPLOAD_ROOT / filename
            pdf_path.write_bytes(upload["content"])

            result = run_pipeline(
                pdf_path=pdf_path,
                question=question,
                use_api_router=fields.get("use_api_router") == "true",
                use_api_embeddings=fields.get("use_api_embeddings") == "true",
                use_api_generator=fields.get("use_api_generator") == "true",
            )
            self.send_json(result)
        except Exception as exc:
            traceback.print_exc()
            self.send_json({"error": str(exc), "type": type(exc).__name__}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def serve_file(self, path: Path, content_type: str) -> None:
        try:
            resolved = path.resolve()
            if not str(resolved).startswith(str(WEB_ROOT.resolve())):
                self.send_json({"error": "invalid path"}, HTTPStatus.BAD_REQUEST)
                return
            if not resolved.exists() or not resolved.is_file():
                self.send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
                return
            data = resolved.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), format % args))


def parse_multipart(handler: BaseHTTPRequestHandler) -> tuple[dict[str, str], dict[str, dict[str, object]]]:
    content_type = handler.headers.get("Content-Type", "")
    if "multipart/form-data" not in content_type or "boundary=" not in content_type:
        raise ValueError("Expected multipart/form-data.")
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"')
    body = handler.rfile.read(int(handler.headers.get("Content-Length", "0")))
    boundary_bytes = ("--" + boundary).encode("utf-8")
    fields: dict[str, str] = {}
    files: dict[str, dict[str, object]] = {}

    for part in body.split(boundary_bytes):
        part = part.strip(b"\r\n")
        if not part or part == b"--" or b"\r\n\r\n" not in part:
            continue
        raw_headers, content = part.split(b"\r\n\r\n", 1)
        if content.endswith(b"\r\n"):
            content = content[:-2]
        headers = parse_part_headers(raw_headers)
        disposition = headers.get("content-disposition", "")
        params = parse_disposition_params(disposition)
        name = params.get("name")
        if not name:
            continue
        filename = params.get("filename")
        if filename is not None:
            files[name] = {"filename": filename, "content": content}
        else:
            fields[name] = content.decode("utf-8", errors="replace")
    return fields, files


def parse_part_headers(raw_headers: bytes) -> dict[str, str]:
    headers: dict[str, str] = {}
    for line in raw_headers.decode("utf-8", errors="replace").split("\r\n"):
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        headers[key.strip().lower()] = value.strip()
    return headers


def parse_disposition_params(value: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for segment in value.split(";"):
        segment = segment.strip()
        if "=" not in segment:
            continue
        key, raw = segment.split("=", 1)
        params[key.strip().lower()] = raw.strip().strip('"')
    return params


def safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {".", "-", "_", " "} else "_" for char in value).strip()
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned or "contract.pdf"


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), ContractSignHandler)
    print(f"ContractSIGN v0.2 local UI: http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
