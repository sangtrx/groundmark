from __future__ import annotations

import json
import tempfile
import uuid
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, PlainTextResponse

from groundmark.infra.jobs import JobStore
from groundmark.infra.metrics import metrics
from groundmark.llm import LLMProviderError
from groundmark.models import ParsedDocument, ParseOptions
from groundmark.normalizers.markitdown import MarkItDownNormalizer
from groundmark.parsers.base import ParserError
from groundmark.security import SecurityError
from groundmark.service import GroundMark
from groundmark.settings import get_settings

settings = get_settings()
engine = GroundMark(settings)
jobs = JobStore(settings.resolved_database_path, settings.job_lease_seconds)

app = FastAPI(
    title="GroundMark API",
    version="0.4.0",
    description=(
        "MarkItDown-powered document intelligence with layout grounding and optional OpenAI "
        "Responses API extraction, classification, schema generation, sectioning, and vision repair."
    ),
)

if settings.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "Idempotency-Key"],
    )


def require_api_key(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if not settings.api_key:
        return
    bearer = authorization.removeprefix("Bearer ").strip() if authorization else None
    if x_api_key != settings.api_key and bearer != settings.api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid API key")


def _safe_suffix(filename: str | None) -> str:
    suffix = Path(filename or "upload.bin").suffix.lower()
    return suffix if suffix and len(suffix) <= 12 else ".bin"


def _save_upload(upload: UploadFile, destination: Path | None = None) -> Path:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if destination is None:
        handle = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=_safe_suffix(upload.filename),
            dir=settings.data_dir,
        )
        destination = Path(handle.name)
        handle.close()
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    try:
        with destination.open("wb") as output:
            while chunk := upload.file.read(1024 * 1024):
                total += len(chunk)
                if total > max_bytes:
                    raise HTTPException(
                        status_code=413, detail="upload exceeds configured size limit"
                    )
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    return destination


def _options(
    mode: str,
    parser: str,
    include_headers_footers: bool,
    include_grounding_comments: bool,
    run_repair: bool = False,
    max_repair_regions: int | None = None,
) -> ParseOptions:
    try:
        return ParseOptions(
            mode=mode,
            parser=parser,
            include_headers_footers=include_headers_footers,
            include_grounding_comments=include_grounding_comments,
            run_repair=run_repair,
            max_repair_regions=max_repair_regions,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _labels(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except json.JSONDecodeError:
        pass
    return [item.strip() for item in raw.split(",") if item.strip()]


def _unprocessable(endpoint: str, exc: Exception) -> HTTPException:
    metrics.inc("api_errors_total", endpoint=endpoint, kind=type(exc).__name__)
    return HTTPException(status_code=422, detail=str(exc))


@app.get("/", response_class=HTMLResponse)
def playground() -> str:
    html_path = Path(__file__).resolve().parents[2] / "web" / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>GroundMark</h1><p>See <a href='/docs'>API docs</a>.</p>"


@app.get("/health/live")
def live() -> dict:
    return {"status": "ok", "service": "groundmark", "version": "0.4.0"}


@app.get("/health/ready")
def ready() -> dict:
    if not MarkItDownNormalizer.available():
        raise HTTPException(status_code=503, detail="MarkItDown is unavailable")
    try:
        jobs.get("readiness-probe")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"job database unavailable: {exc}") from exc
    return {"status": "ready", "openai_configured": engine.openai_configured}


@app.get("/health")
def health_alias() -> dict:
    return live()


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    return metrics.render_prometheus()


@app.get("/v1/capabilities", dependencies=[Depends(require_api_key)])
def capabilities() -> dict:
    return {
        "normalizer": "markitdown",
        "layout": {
            "native_pdf": True,
            "docling": engine.docling_layout.available(),
        },
        "openai": {
            "configured": engine.openai_configured,
            "api": "responses",
            "default_model": settings.openai_model,
            "accurate_model": settings.openai_accurate_model,
            "vision_model": settings.resolved_openai_vision_model,
            "store": settings.openai_store,
        },
        "modes": ["fast", "balanced", "accurate"],
        "providers": ["auto", "openai", "heuristic"],
        "features": [
            "parse",
            "openai_structured_extract",
            "openai_classify",
            "openai_schema_generation",
            "openai_sectioning",
            "openai_selective_vision_repair",
            "semantic_split",
            "multi_document_split",
            "durable_jobs",
            "content_addressed_cache",
        ],
    }


@app.post("/v1/parse", dependencies=[Depends(require_api_key)])
def parse_document(
    document: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()] = "balanced",
    parser: Annotated[str, Form()] = "auto",
    include_headers_footers: Annotated[bool, Form()] = False,
    include_grounding_comments: Annotated[bool, Form()] = True,
    run_repair: Annotated[bool, Form()] = False,
    max_repair_regions: Annotated[int | None, Form()] = None,
) -> dict:
    path = _save_upload(document)
    try:
        result = engine.parse(
            path,
            _options(
                mode,
                parser,
                include_headers_footers,
                include_grounding_comments,
                run_repair,
                max_repair_regions,
            ),
            media_type=document.content_type,
            display_filename=document.filename,
        )
        return result.model_dump(mode="json")
    except (ParserError, SecurityError, ValueError, FileNotFoundError, LLMProviderError) as exc:
        raise _unprocessable("parse", exc) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/v1/jobs/parse", status_code=202, dependencies=[Depends(require_api_key)])
def enqueue_parse(
    document: Annotated[UploadFile, File()],
    mode: Annotated[str, Form()] = "balanced",
    parser: Annotated[str, Form()] = "auto",
    include_headers_footers: Annotated[bool, Form()] = False,
    include_grounding_comments: Annotated[bool, Form()] = True,
    run_repair: Annotated[bool, Form()] = False,
    max_repair_regions: Annotated[int | None, Form()] = None,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> dict:
    job_id = (
        "job_" + uuid.uuid5(uuid.NAMESPACE_URL, idempotency_key).hex
        if idempotency_key
        else "job_" + uuid.uuid4().hex
    )
    existing = jobs.get(job_id)
    if existing:
        return existing.public()
    destination = settings.data_dir / "uploads" / f"{job_id}{_safe_suffix(document.filename)}"
    path = _save_upload(document, destination)
    try:
        job = jobs.enqueue(
            path,
            document.filename or path.name,
            document.content_type,
            _options(
                mode,
                parser,
                include_headers_footers,
                include_grounding_comments,
                run_repair,
                max_repair_regions,
            ),
            job_id=job_id,
        )
        metrics.inc("jobs_total", status="queued")
        return job.public()
    except Exception:
        path.unlink(missing_ok=True)
        raise


@app.get("/v1/jobs/{job_id}", dependencies=[Depends(require_api_key)])
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    payload = job.public()
    if job.status == "succeeded":
        payload["result_url"] = f"/v1/jobs/{job_id}/result"
    return payload


@app.get("/v1/jobs/{job_id}/result", dependencies=[Depends(require_api_key)])
def get_job_result(job_id: str) -> dict:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status != "succeeded" or not job.result_path:
        raise HTTPException(status_code=409, detail=f"job status is {job.status}")
    path = Path(job.result_path)
    if not path.is_file():
        raise HTTPException(status_code=410, detail="job result expired or missing")
    return ParsedDocument.model_validate_json(path.read_text(encoding="utf-8")).model_dump(
        mode="json"
    )


@app.delete("/v1/jobs/{job_id}", status_code=204, dependencies=[Depends(require_api_key)])
def delete_job(job_id: str) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    Path(job.input_path).unlink(missing_ok=True)
    if job.result_path:
        Path(job.result_path).unlink(missing_ok=True)
    jobs.delete(job_id)


@app.post("/v1/extract", dependencies=[Depends(require_api_key)])
def extract_document(
    document: Annotated[UploadFile, File()],
    extraction_schema: Annotated[str, Form(alias="schema")],
    mode: Annotated[str, Form()] = "balanced",
    provider: Annotated[str, Form()] = "auto",
    model: Annotated[str | None, Form()] = None,
) -> dict:
    path = _save_upload(document)
    try:
        parsed = engine.parse(
            path,
            ParseOptions(mode=mode),
            media_type=document.content_type,
            display_filename=document.filename,
        )
        result = engine.extract(
            parsed,
            json.loads(extraction_schema),
            provider=provider,  # type: ignore[arg-type]
            model=model,
        )
        return {
            "parse": parsed.model_dump(mode="json"),
            "extraction": result.model_dump(mode="json"),
        }
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=422, detail=f"invalid JSON schema: {exc}") from exc
    except (ParserError, SecurityError, ValueError, LLMProviderError) as exc:
        raise _unprocessable("extract", exc) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/v1/extract/build-schema", dependencies=[Depends(require_api_key)])
def build_extract_schema(
    document: Annotated[UploadFile, File()],
    provider: Annotated[str, Form()] = "auto",
    model: Annotated[str | None, Form()] = None,
) -> dict:
    path = _save_upload(document)
    try:
        parsed = engine.parse(
            path,
            display_filename=document.filename,
            media_type=document.content_type,
        )
        return engine.build_schema(parsed, provider=provider, model=model)  # type: ignore[arg-type]
    except (ParserError, SecurityError, ValueError, LLMProviderError) as exc:
        raise _unprocessable("build_schema", exc) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/v1/classify", dependencies=[Depends(require_api_key)])
def classify_document_endpoint(
    document: Annotated[UploadFile, File()],
    provider: Annotated[str, Form()] = "auto",
    labels: Annotated[str | None, Form()] = None,
    model: Annotated[str | None, Form()] = None,
) -> dict:
    path = _save_upload(document)
    try:
        parsed = engine.parse(
            path, display_filename=document.filename, media_type=document.content_type
        )
        results = engine.classify(
            parsed,
            provider=provider,  # type: ignore[arg-type]
            labels=_labels(labels),
            model=model,
        )
        return {"classifications": [item.model_dump(mode="json") for item in results]}
    except (ParserError, SecurityError, ValueError, LLMProviderError) as exc:
        raise _unprocessable("classify", exc) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/v1/section", dependencies=[Depends(require_api_key)])
def section_document(
    document: Annotated[UploadFile, File()],
    provider: Annotated[str, Form()] = "auto",
    model: Annotated[str | None, Form()] = None,
) -> dict:
    path = _save_upload(document)
    try:
        parsed = engine.parse(
            path, display_filename=document.filename, media_type=document.content_type
        )
        sections = engine.section(parsed, provider=provider, model=model)  # type: ignore[arg-type]
        return {"sections": [section.model_dump(mode="json") for section in sections]}
    except (ParserError, SecurityError, ValueError, LLMProviderError) as exc:
        raise _unprocessable("section", exc) from exc
    finally:
        path.unlink(missing_ok=True)


@app.post("/v1/split", dependencies=[Depends(require_api_key)])
def split_endpoint(
    document: Annotated[UploadFile, File()],
    strategy: Annotated[str, Form()] = "semantic",
    max_chars: Annotated[int, Form()] = 5000,
    overlap_blocks: Annotated[int, Form()] = 1,
) -> dict:
    path = _save_upload(document)
    try:
        parsed = engine.parse(
            path, display_filename=document.filename, media_type=document.content_type
        )
        if strategy == "documents":
            return {
                "segments": [
                    item.model_dump(mode="json") for item in engine.split_documents(parsed)
                ]
            }
        if strategy != "semantic":
            raise HTTPException(status_code=422, detail="strategy must be semantic or documents")
        chunks = engine.split(parsed, max_chars=max_chars, overlap_blocks=overlap_blocks)
        return {"chunks": [chunk.model_dump(mode="json") for chunk in chunks]}
    finally:
        path.unlink(missing_ok=True)
