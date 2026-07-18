from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class BlockType(StrEnum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE = "table"
    FIGURE = "figure"
    FORMULA = "formula"
    CODE = "code"
    HEADER = "header"
    FOOTER = "footer"
    MARGINALIA = "marginalia"
    FORM_FIELD = "form_field"
    CHECKBOX = "checkbox"
    SIGNATURE = "signature"
    ATTESTATION = "attestation"
    LOGO = "logo"
    CARD = "card"
    SCAN_CODE = "scan_code"
    UNKNOWN = "unknown"


class BBox(BaseModel):
    """Normalized top-left page coordinates in [0, 1]."""

    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_order(self) -> BBox:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bbox max coordinates must be >= min coordinates")
        return self

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def union(self, other: BBox) -> BBox:
        return BBox(
            x0=min(self.x0, other.x0),
            y0=min(self.y0, other.y0),
            x1=max(self.x1, other.x1),
            y1=max(self.y1, other.y1),
        )


class Grounding(BaseModel):
    page: int = Field(ge=1)
    bbox: BBox
    source: str = "unknown"
    source_ids: list[str] = Field(default_factory=list)


class Block(BaseModel):
    id: str
    type: BlockType = BlockType.PARAGRAPH
    text: str = ""
    markdown: str | None = None
    html: str | None = None
    level: int | None = Field(default=None, ge=1, le=6)
    reading_order: int = Field(default=0, ge=0)
    confidence: float = Field(default=0.75, ge=0, le=1)
    grounding: Grounding
    metadata: dict[str, Any] = Field(default_factory=dict)

    @staticmethod
    def stable_id(document_sha256: str, page: int, bbox: BBox, text: str, role: str) -> str:
        raw = (
            f"{document_sha256}:{page}:{bbox.x0:.5f}:{bbox.y0:.5f}:"
            f"{bbox.x1:.5f}:{bbox.y1:.5f}:{role}:{text[:512]}"
        )
        return "blk_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


class Page(BaseModel):
    number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    blocks: list[Block] = Field(default_factory=list)
    confidence: float = Field(default=0.75, ge=0, le=1)
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StageTrace(BaseModel):
    stage: str
    provider: str
    status: Literal["ok", "skipped", "fallback", "error"] = "ok"
    elapsed_ms: int = Field(default=0, ge=0)
    details: dict[str, Any] = Field(default_factory=dict)


class DocumentProfile(BaseModel):
    kind: str
    pages: int = Field(default=1, ge=1)
    scanned_pages: list[int] = Field(default_factory=list)
    image_heavy_pages: list[int] = Field(default_factory=list)
    table_heavy_pages: list[int] = Field(default_factory=list)
    multi_column_pages: list[int] = Field(default_factory=list)
    has_forms: bool = False
    language_hints: list[str] = Field(default_factory=list)
    recommended_mode: Literal["fast", "balanced", "accurate"] = "balanced"


class DocumentMetadata(BaseModel):
    filename: str
    media_type: str | None = None
    bytes: int = Field(ge=0)
    sha256: str
    parser: str
    version: str = "0.4.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    elapsed_ms: int = Field(default=0, ge=0)
    cache_hit: bool = False
    warnings: list[str] = Field(default_factory=list)
    trace: list[StageTrace] = Field(default_factory=list)


class Section(BaseModel):
    id: str
    title: str
    level: int = Field(ge=1, le=6)
    start_block_id: str
    end_block_id: str | None = None
    page: int
    children: list[Section] = Field(default_factory=list)


class Chunk(BaseModel):
    id: str
    text: str
    markdown: str
    block_ids: list[str]
    pages: list[int]
    token_estimate: int
    confidence: float = Field(default=0.75, ge=0, le=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    metadata: DocumentMetadata
    profile: DocumentProfile
    pages: list[Page]
    markdown: str
    confidence: float = Field(ge=0, le=1)
    quality: dict[str, Any] = Field(default_factory=dict)
    sections: list[Section] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)


class FieldEvidence(BaseModel):
    block_id: str
    page: int
    bbox: BBox
    quote: str


class ExtractedField(BaseModel):
    path: str
    value: Any = None
    confidence: float = Field(default=0, ge=0, le=1)
    evidence: list[FieldEvidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class ExtractionResult(BaseModel):
    data: dict[str, Any]
    fields: list[ExtractedField]
    schema_valid: bool
    validation_errors: list[str] = Field(default_factory=list)
    provider: str = "heuristic"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Classification(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)
    evidence_block_ids: list[str] = Field(default_factory=list)


class DocumentSegment(BaseModel):
    id: str
    start_page: int
    end_page: int
    label: str = "document"
    confidence: float = Field(default=0.5, ge=0, le=1)
    reason: str = ""


class ParseOptions(BaseModel):
    mode: Literal["fast", "balanced", "accurate"] = "balanced"
    parser: Literal["auto", "markitdown", "native", "docling"] = "auto"
    include_headers_footers: bool = False
    include_grounding_comments: bool = True
    use_cache: bool = True
    force_ocr: bool = False
    run_repair: bool = False
    repair_endpoint: str | None = None
    repair_model: str | None = None
    repair_api_key: str | None = None
    min_confidence_for_repair: float = Field(default=0.72, ge=0, le=1)
    max_pages: int | None = Field(default=None, ge=1, le=10000)
    max_repair_regions: int | None = Field(default=None, ge=1, le=500)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    language_hints: list[str] = Field(default_factory=list)
    chunk_max_chars: int = Field(default=5000, ge=500, le=100000)
    chunk_overlap_blocks: int = Field(default=1, ge=0, le=20)

    @model_validator(mode="after")
    def validate_page_range(self) -> ParseOptions:
        if self.page_start and self.page_end and self.page_end < self.page_start:
            raise ValueError("page_end must be >= page_start")
        return self
