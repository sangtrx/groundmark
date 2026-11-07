from __future__ import annotations

from pathlib import Path

from groundmark.evidence_audit import audit_extraction
from groundmark.models import (
    BBox,
    Block,
    DocumentMetadata,
    DocumentProfile,
    ExtractedField,
    ExtractionResult,
    FieldEvidence,
    Grounding,
    Page,
    ParsedDocument,
)


def fixtures() -> tuple[ParsedDocument, ExtractionResult]:
    bbox = BBox(x0=0.1, y0=0.2, x1=0.7, y1=0.3)
    block = Block(
        id="blk_invoice",
        text="Invoice Number: A-42",
        markdown="Invoice Number: A-42",
        grounding=Grounding(page=1, bbox=bbox, source="native"),
    )
    document = ParsedDocument(
        metadata=DocumentMetadata(
            filename="invoice.txt", bytes=20, sha256="a" * 64, parser="test"
        ),
        profile=DocumentProfile(kind="text", pages=1),
        pages=[Page(number=1, width=100, height=100, blocks=[block])],
        markdown="Invoice Number: A-42",
        confidence=1,
    )
    result = ExtractionResult(
        data={"invoice_number": "A-42"},
        fields=[
            ExtractedField(
                path="invoice_number",
                value="A-42",
                confidence=0.9,
                evidence=[
                    FieldEvidence(
                        block_id="blk_invoice",
                        page=1,
                        bbox=bbox,
                        quote="Invoice    Number: A-42",
                    )
                ],
            )
        ],
        schema_valid=True,
    )
    return document, result


def codes(document: ParsedDocument, result: ExtractionResult) -> set[str]:
    return {finding.code for finding in audit_extraction(document, result).findings}


def test_valid_evidence_accepts_whitespace_normalized_quote() -> None:
    doc, extracted = fixtures()
    report = audit_extraction(doc, extracted)
    assert report.ok
    assert report.fields_checked == 1
    assert report.citations_checked == 1
    assert report.document_sha256 == "a" * 64


def test_unknown_block_id_fails_closed() -> None:
    doc, extracted = fixtures()
    extracted.fields[0].evidence[0].block_id = "blk_fake"
    assert "unknown_block_id" in codes(doc, extracted)


def test_modified_page_bbox_and_quote_rejected() -> None:
    doc, extracted = fixtures()
    cited = extracted.fields[0].evidence[0]
    cited.page = 2
    cited.bbox = BBox(x0=0.3, y0=0.2, x1=0.7, y1=0.3)
    cited.quote = "Invoice Number: B-99"
    assert {
        "evidence_page_mismatch",
        "evidence_bbox_mismatch",
        "evidence_quote_mismatch",
    } <= codes(doc, extracted)


def test_non_null_without_evidence_is_rejected() -> None:
    doc, extracted = fixtures()
    extracted.fields[0].evidence = []
    assert "missing_evidence" in codes(doc, extracted)


def test_conflicting_data_or_missing_field_record_is_rejected() -> None:
    doc, extracted = fixtures()
    extracted.fields[0].value = "B-99"
    extracted.data["total"] = 999
    assert {"field_value_mismatch", "missing_field_record"} <= codes(doc, extracted)


def test_duplicate_ids_page_inconsistency_and_invalid_schema() -> None:
    doc, extracted = fixtures()
    doc.pages[0].blocks.append(doc.pages[0].blocks[0].model_copy(deep=True))
    doc.pages[0].blocks[1].grounding.page = 2
    extracted.schema_valid = False
    assert {"duplicate_block_id", "block_page_mismatch", "schema_invalid"} <= codes(
        doc, extracted
    )


def test_nested_fields_and_nulls_are_supported() -> None:
    doc, extracted = fixtures()
    extracted.data = {"invoice": {"number": "A-42", "missing": None}}
    extracted.fields[0].path = "invoice.number"
    extracted.fields.append(ExtractedField(path="invoice.missing", value=None))
    assert audit_extraction(doc, extracted).ok


def test_json_round_trip_and_real_cli(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from groundmark.cli import app

    doc, extracted = fixtures()
    doc_path = tmp_path / "document.json"
    extraction_path = tmp_path / "extraction.json"
    doc_path.write_text(doc.model_dump_json(), encoding="utf-8")
    extraction_path.write_text(extracted.model_dump_json(), encoding="utf-8")
    runner = CliRunner()
    passed = runner.invoke(app, ["audit", str(doc_path), str(extraction_path)])
    assert passed.exit_code == 0, passed.output
    assert '"ok": true' in passed.output
    extracted.fields[0].evidence[0].quote = "fabricated"
    extraction_path.write_text(extracted.model_dump_json(), encoding="utf-8")
    failed = runner.invoke(app, ["audit", str(doc_path), str(extraction_path)])
    assert failed.exit_code == 1, failed.output
    assert "evidence_quote_mismatch" in failed.output
    extraction_path.write_text("{invalid json", encoding="utf-8")
    invalid = runner.invoke(app, ["audit", str(doc_path), str(extraction_path)])
    assert invalid.exit_code == 2
    assert "Invalid document or extraction JSON" in invalid.output
