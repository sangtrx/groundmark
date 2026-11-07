"""Offline structural audit of grounded extraction evidence.

This checks the integrity of source references, not whether the value is
semantically entailed by the cited text. No model calls, document uploads,
or external services are involved.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from groundmark.models import Block, ExtractionResult, ParsedDocument


class AuditFinding(BaseModel):
    code: str
    location: str
    message: str


class AuditReport(BaseModel):
    ok: bool
    document_sha256: str
    fields_checked: int = Field(ge=0)
    citations_checked: int = Field(ge=0)
    findings: list[AuditFinding] = Field(default_factory=list)


def _normal(text: str) -> str:
    return " ".join(text.split())


def _data_field(data: dict[str, Any], path: str) -> tuple[bool, Any]:
    value: Any = data
    for key in path.split("."):
        if not isinstance(value, dict) or key not in value:
            return False, None
        value = value[key]
    return True, value


def _leaf_paths(data: dict[str, Any], prefix: str = "") -> list[str]:
    paths: list[str] = []
    for key, value in data.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict) and value:
            paths.extend(_leaf_paths(value, path))
        else:
            paths.append(path)
    return paths


def audit_extraction(document: ParsedDocument, result: ExtractionResult) -> AuditReport:
    """Verify internal consistency of a ParsedDocument/ExtractionResult pair.

    Fail closed on fabricated or stale references. A PASS does not guarantee
    the document's original bytes or a model's interpretation are authentic.
    """
    findings: list[AuditFinding] = []

    def fail(code: str, location: str, message: str) -> None:
        findings.append(AuditFinding(code=code, location=location, message=message))

    blocks: dict[str, Block] = {}
    page_numbers: set[int] = set()

    for page in document.pages:
        if page.number in page_numbers:
            fail("duplicate_page", f"pages[{page.number}]", "Page number occurs more than once")
        page_numbers.add(page.number)
        for block in page.blocks:
            if block.id in blocks:
                fail("duplicate_block_id", f"blocks[{block.id}]", "Block ID occurs more than once")
            else:
                blocks[block.id] = block
            if block.grounding.page != page.number:
                fail(
                    "block_page_mismatch",
                    f"blocks[{block.id}]",
                    "Block grounding page differs from its containing page",
                )

    if not result.schema_valid:
        fail("schema_invalid", "extraction.schema_valid", "Extraction reports invalid schema")
    elif result.validation_errors:
        fail(
            "schema_flag_inconsistent",
            "extraction.validation_errors",
            "Extraction reports validation errors despite schema_valid=true",
        )

    fields: set[str] = set()
    citations_checked = 0
    for field in result.fields:
        location = f"fields[{field.path}]"
        if not field.path or any(not part for part in field.path.split(".")):
            fail("invalid_field_path", location, "Field path must be non-empty dot notation")
        if field.path in fields:
            fail("duplicate_field_path", location, "Field path occurs more than once")
        fields.add(field.path)

        exists, value = _data_field(result.data, field.path)
        if not exists:
            fail("missing_data_path", location, "Field path is absent from extraction data")
        elif type(value) is not type(field.value) or value != field.value:
            fail("field_value_mismatch", location, "Field value differs from extraction data")

        if field.value is not None and not field.evidence:
            fail("missing_evidence", location, "Non-null field has no source evidence")

        seen_citations: set[str] = set()
        for evidence in field.evidence:
            citations_checked += 1
            reference = f"{location}.evidence[{evidence.block_id}]"
            if evidence.block_id in seen_citations:
                fail("duplicate_evidence", reference, "Block cited repeatedly for one field")
            seen_citations.add(evidence.block_id)
            block = blocks.get(evidence.block_id)
            if block is None:
                fail("unknown_block_id", reference, "Cited block does not exist in the document")
                continue
            if evidence.page != block.grounding.page:
                fail("evidence_page_mismatch", reference, "Cited page differs from source block")
            coordinates = ("x0", "y0", "x1", "y1")
            if any(
                abs(getattr(evidence.bbox, name) - getattr(block.grounding.bbox, name)) > 1e-6
                for name in coordinates
            ):
                fail("evidence_bbox_mismatch", reference, "Cited bbox differs from source block")
            quote = _normal(evidence.quote)
            source_text = (_normal(block.text), _normal(block.markdown or ""))
            if not quote or not any(quote in source for source in source_text):
                fail("evidence_quote_mismatch", reference, "Cited quote not found in source block")

    for path in _leaf_paths(result.data):
        if path not in fields:
            fail("missing_field_record", f"data[{path}]", "Data leaf has no field evidence record")

    return AuditReport(
        ok=not findings,
        document_sha256=document.metadata.sha256,
        fields_checked=len(result.fields),
        citations_checked=citations_checked,
        findings=findings,
    )
