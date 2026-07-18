from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Literal

from groundmark.classification import (
    classify_document,
    classify_document_openai,
    split_document,
)
from groundmark.extraction import (
    HeuristicExtractor,
    OpenAIExtractor,
    build_schema_from_document,
    build_schema_from_document_openai,
)
from groundmark.fusion import fuse_markdown_with_layout
from groundmark.infra.cache import CacheStore
from groundmark.infra.metrics import metrics
from groundmark.layout.docling import DoclingLayoutExtractor
from groundmark.layout.pdf import PDFLayoutExtractor
from groundmark.llm import LLMProviderError, OpenAIProvider
from groundmark.markdown import pages_to_markdown
from groundmark.models import (
    BlockType,
    Chunk,
    Classification,
    DocumentMetadata,
    DocumentSegment,
    ExtractionResult,
    ParsedDocument,
    ParseOptions,
    Section,
    StageTrace,
)
from groundmark.normalizers.markitdown import MarkItDownNormalizer
from groundmark.profiling import build_profile
from groundmark.quality import evaluate_quality
from groundmark.repair import SelectiveRepair
from groundmark.sectioning import build_sections, build_sections_openai
from groundmark.security import guess_media_type, validate_local_file
from groundmark.settings import Settings, get_settings
from groundmark.splitting import semantic_chunks

ProviderMode = Literal["auto", "openai", "heuristic"]


class GroundMark:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        openai_provider: OpenAIProvider | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.openai = openai_provider
        if self.openai is None and self.settings.resolved_openai_api_key:
            self.openai = OpenAIProvider(
                api_key=self.settings.resolved_openai_api_key,
                base_url=self.settings.resolved_openai_base_url,
                default_model=self.settings.openai_model,
                vision_model=self.settings.resolved_openai_vision_model,
                timeout_seconds=self.settings.openai_timeout_seconds,
                max_retries=self.settings.openai_max_retries,
                max_output_tokens=self.settings.openai_max_output_tokens,
                store=self.settings.openai_store,
            )

        markitdown_llm_client = None
        markitdown_llm_model = None
        if self.settings.markitdown_enable_plugins and self.openai is not None:
            markitdown_llm_client = self.openai.client
            markitdown_llm_model = (
                self.settings.markitdown_llm_model or self.settings.resolved_openai_vision_model
            )
        self.normalizer = MarkItDownNormalizer(
            enable_plugins=self.settings.markitdown_enable_plugins,
            llm_client=markitdown_llm_client,
            llm_model=markitdown_llm_model,
        )
        self.pdf_layout = PDFLayoutExtractor()
        self.docling_layout = DoclingLayoutExtractor()
        self.repair = SelectiveRepair(
            self.openai,
            max_regions=self.settings.openai_max_repair_regions,
        )
        self.heuristic_extractor = HeuristicExtractor()
        self.cache = CacheStore(self.settings.resolved_cache_dir)

    @property
    def openai_configured(self) -> bool:
        return self.openai is not None

    def _layout(self, path: Path, options: ParseOptions, trace: list[StageTrace]):
        suffix = path.suffix.lower()
        if options.parser == "markitdown":
            trace.append(StageTrace(stage="layout", provider="none", status="skipped"))
            return []

        if options.parser == "docling":
            started = time.perf_counter()
            pages, _ = self.docling_layout.extract(path, options)
            trace.append(
                StageTrace(
                    stage="layout",
                    provider=self.docling_layout.name,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
            )
            return pages

        if suffix == ".pdf":
            if (
                options.mode == "accurate"
                and options.parser == "auto"
                and self.docling_layout.available()
            ):
                started = time.perf_counter()
                try:
                    pages, _ = self.docling_layout.extract(path, options)
                    trace.append(
                        StageTrace(
                            stage="layout",
                            provider=self.docling_layout.name,
                            elapsed_ms=int((time.perf_counter() - started) * 1000),
                        )
                    )
                    return pages
                except Exception as exc:
                    trace.append(
                        StageTrace(
                            stage="layout",
                            provider=self.docling_layout.name,
                            status="fallback",
                            elapsed_ms=int((time.perf_counter() - started) * 1000),
                            details={"error": str(exc)},
                        )
                    )
            started = time.perf_counter()
            pages = self.pdf_layout.extract(path, options)
            trace.append(
                StageTrace(
                    stage="layout",
                    provider=self.pdf_layout.name,
                    elapsed_ms=int((time.perf_counter() - started) * 1000),
                )
            )
            if (
                options.parser == "auto"
                and options.mode == "balanced"
                and self.docling_layout.available()
                and any(page.metadata.get("scanned") for page in pages)
            ):
                escalate_started = time.perf_counter()
                try:
                    docling_pages, _ = self.docling_layout.extract(path, options)
                    trace.append(
                        StageTrace(
                            stage="layout_escalation",
                            provider=self.docling_layout.name,
                            elapsed_ms=int((time.perf_counter() - escalate_started) * 1000),
                            details={"reason": "scanned pages detected"},
                        )
                    )
                    return docling_pages
                except Exception as exc:
                    trace.append(
                        StageTrace(
                            stage="layout_escalation",
                            provider=self.docling_layout.name,
                            status="fallback",
                            elapsed_ms=int((time.perf_counter() - escalate_started) * 1000),
                            details={"error": str(exc)},
                        )
                    )
            return pages

        if options.parser == "native":
            trace.append(StageTrace(stage="layout", provider="none", status="skipped"))
            return []
        if options.mode == "accurate" and self.docling_layout.available():
            started = time.perf_counter()
            try:
                pages, _ = self.docling_layout.extract(path, options)
                trace.append(
                    StageTrace(
                        stage="layout",
                        provider=self.docling_layout.name,
                        elapsed_ms=int((time.perf_counter() - started) * 1000),
                    )
                )
                return pages
            except Exception as exc:
                trace.append(
                    StageTrace(
                        stage="layout",
                        provider=self.docling_layout.name,
                        status="fallback",
                        elapsed_ms=int((time.perf_counter() - started) * 1000),
                        details={"error": str(exc)},
                    )
                )
        trace.append(StageTrace(stage="layout", provider="none", status="skipped"))
        return []

    def parse(
        self,
        path: str | Path,
        options: ParseOptions | None = None,
        media_type: str | None = None,
        display_filename: str | None = None,
    ) -> ParsedDocument:
        options = options or ParseOptions()
        if options.run_repair:
            options = options.model_copy(
                update={
                    "repair_model": options.repair_model
                    or self.settings.resolved_openai_vision_model,
                    "max_repair_regions": options.max_repair_regions
                    or self.settings.openai_max_repair_regions,
                }
            )
        path = Path(path)
        validate_local_file(path, self.settings.max_upload_mb * 1024 * 1024)
        started = time.perf_counter()
        content = path.read_bytes()
        document_sha256 = hashlib.sha256(content).hexdigest()
        cache_key = self.cache.key(document_sha256, options)
        if options.use_cache:
            cached = self.cache.get(cache_key)
            if cached:
                cached.metadata.filename = display_filename or path.name
                metrics.inc("parse_total", status="cache_hit", mode=options.mode)
                return cached

        trace: list[StageTrace] = []
        normalize_started = time.perf_counter()
        normalized = self.normalizer.convert(path)
        trace.append(
            StageTrace(
                stage="normalize",
                provider=self.normalizer.name,
                elapsed_ms=int((time.perf_counter() - normalize_started) * 1000),
                details=normalized.metadata,
            )
        )

        layout_pages = self._layout(path, options, trace)
        fuse_started = time.perf_counter()
        pages = fuse_markdown_with_layout(normalized.markdown, layout_pages, document_sha256)
        trace.append(
            StageTrace(
                stage="fusion",
                provider="groundmark-aligner",
                elapsed_ms=int((time.perf_counter() - fuse_started) * 1000),
                details={"markdown_elements": sum(len(page.blocks) for page in pages)},
            )
        )

        if not options.include_headers_footers:
            for page in pages:
                page.blocks = [
                    block
                    for block in page.blocks
                    if block.type not in {BlockType.HEADER, BlockType.FOOTER}
                ]
                for order, block in enumerate(page.blocks):
                    block.reading_order = order

        repair_logs: list[str] = []
        if options.run_repair:
            repair_started = time.perf_counter()
            repair_logs = self.repair.run(path, pages, options)
            trace.append(
                StageTrace(
                    stage="repair",
                    provider=(
                        f"openai-responses:{options.repair_model}"
                        if self.openai_configured
                        else "unconfigured"
                    ),
                    status="ok"
                    if any(log.startswith("repaired") for log in repair_logs)
                    else "skipped",
                    elapsed_ms=int((time.perf_counter() - repair_started) * 1000),
                    details={"logs": repair_logs},
                )
            )

        profile = build_profile(path, pages, options.language_hints)
        quality = evaluate_quality(pages)
        if repair_logs:
            quality["repair_logs"] = repair_logs
        markdown = pages_to_markdown(
            pages,
            options.include_grounding_comments,
            include_headers_footers=options.include_headers_footers,
        )
        sections = build_sections(pages)
        chunks = semantic_chunks(
            pages,
            max_chars=options.chunk_max_chars,
            overlap_blocks=options.chunk_overlap_blocks,
        )
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        parser_chain = "+".join(
            trace_item.provider
            for trace_item in trace
            if trace_item.stage in {"normalize", "layout", "layout_escalation"}
            and trace_item.status == "ok"
        )
        metadata = DocumentMetadata(
            filename=display_filename or path.name,
            media_type=guess_media_type(path, media_type),
            bytes=len(content),
            sha256=document_sha256,
            parser=parser_chain or "markitdown",
            elapsed_ms=elapsed_ms,
            warnings=quality.get("warnings", []),
            trace=trace,
        )
        document = ParsedDocument(
            metadata=metadata,
            profile=profile,
            pages=pages,
            markdown=markdown,
            confidence=quality["confidence"],
            quality=quality,
            sections=sections,
            chunks=chunks,
        )
        if options.use_cache:
            self.cache.put(cache_key, document)
        metrics.inc("parse_total", status="success", mode=options.mode)
        metrics.inc("parse_elapsed_ms_total", value=elapsed_ms, mode=options.mode)
        return document

    def _use_openai(self, mode: ProviderMode, default_enabled: bool) -> bool:
        if mode not in {"auto", "openai", "heuristic"}:
            raise ValueError("provider must be auto, openai, or heuristic")
        if mode == "openai":
            if self.openai is None:
                raise ValueError("provider=openai requires GROUNDMARK_OPENAI_API_KEY")
            return True
        if mode == "heuristic":
            return False
        return bool(default_enabled and self.openai is not None)

    def extract(
        self,
        document: ParsedDocument,
        schema: dict[str, Any],
        *,
        provider: ProviderMode = "auto",
        model: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
    ) -> ExtractionResult:
        ephemeral_provider = None
        if self.openai is None and endpoint and api_key:
            ephemeral_provider = OpenAIProvider(
                api_key=api_key,
                base_url=endpoint,
                default_model=model or self.settings.openai_model,
                vision_model=model or self.settings.resolved_openai_vision_model,
                timeout_seconds=self.settings.openai_timeout_seconds,
                max_retries=self.settings.openai_max_retries,
                max_output_tokens=self.settings.openai_max_output_tokens,
                store=self.settings.openai_store,
            )
        if provider not in {"auto", "openai", "heuristic"}:
            raise ValueError("provider must be auto, openai, or heuristic")
        selected = ephemeral_provider or self.openai
        use_openai = provider == "openai" or (
            provider == "auto" and self.settings.openai_default_for_extract and selected is not None
        )
        if provider == "openai" and selected is None:
            raise ValueError("provider=openai requires GROUNDMARK_OPENAI_API_KEY")
        if use_openai and selected is not None:
            extractor = OpenAIExtractor(
                selected,
                max_input_chars=self.settings.openai_max_input_chars,
                model=model,
            )
            try:
                result = extractor.extract(document, schema)
            except LLMProviderError as exc:
                if provider != "auto" or not self.settings.openai_fallback_on_error:
                    raise
                result = self.heuristic_extractor.extract(document.pages, schema)
                result.metadata["openai_fallback_error"] = str(exc)
                result.metadata["requested_provider"] = "openai-responses"
        else:
            result = self.heuristic_extractor.extract(document.pages, schema)
        metrics.inc(
            "extract_total", provider=result.provider, valid=str(result.schema_valid).lower()
        )
        return result

    def build_schema(
        self,
        document: ParsedDocument,
        *,
        provider: ProviderMode = "auto",
        model: str | None = None,
    ) -> dict[str, Any]:
        if self._use_openai(provider, self.settings.openai_default_for_schema):
            assert self.openai is not None
            try:
                return build_schema_from_document_openai(
                    document,
                    self.openai,
                    max_input_chars=self.settings.openai_max_input_chars,
                    model=model,
                )
            except LLMProviderError as exc:
                if provider != "auto" or not self.settings.openai_fallback_on_error:
                    raise
                fallback = build_schema_from_document(document)
                fallback["warnings"] = [*fallback.get("warnings", []), f"OpenAI fallback: {exc}"]
                return fallback
        return build_schema_from_document(document)

    def section(
        self,
        document: ParsedDocument,
        *,
        provider: ProviderMode = "auto",
        model: str | None = None,
    ) -> list[Section]:
        if self._use_openai(provider, self.settings.openai_default_for_section):
            assert self.openai is not None
            try:
                return build_sections_openai(
                    document,
                    self.openai,
                    model=model,
                    max_input_chars=self.settings.openai_max_input_chars,
                )
            except LLMProviderError:
                if provider != "auto" or not self.settings.openai_fallback_on_error:
                    raise
        return document.sections or build_sections(document.pages)

    def split(
        self,
        document: ParsedDocument,
        max_chars: int = 5000,
        overlap_blocks: int = 1,
    ) -> list[Chunk]:
        return semantic_chunks(document.pages, max_chars=max_chars, overlap_blocks=overlap_blocks)

    def classify(
        self,
        document: ParsedDocument,
        *,
        provider: ProviderMode = "auto",
        labels: list[str] | None = None,
        model: str | None = None,
    ) -> list[Classification]:
        if self._use_openai(provider, self.settings.openai_default_for_classify):
            assert self.openai is not None
            try:
                return classify_document_openai(
                    document,
                    self.openai,
                    labels=labels,
                    model=model,
                    max_input_chars=self.settings.openai_max_input_chars,
                )
            except LLMProviderError:
                if provider != "auto" or not self.settings.openai_fallback_on_error:
                    raise
        return classify_document(document)

    def split_documents(self, document: ParsedDocument) -> list[DocumentSegment]:
        return split_document(document)
