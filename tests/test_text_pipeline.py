from pathlib import Path

from groundmark.models import ParseOptions
from groundmark.service import GroundMark
from groundmark.settings import Settings


def _engine(tmp_path: Path) -> GroundMark:
    return GroundMark(Settings(data_dir=tmp_path / "data", max_upload_mb=10))


def test_markitdown_parse_extract_section_split(tmp_path: Path) -> None:
    sample = tmp_path / "invoice.txt"
    sample.write_text(
        "# Invoice\nInvoice Number: A-42\nVendor: ACME\nTotal Amount: 123.45\nPaid: Yes\n",
        encoding="utf-8",
    )
    engine = _engine(tmp_path)
    parsed = engine.parse(sample, ParseOptions(parser="native", use_cache=False))
    assert parsed.confidence > 0.9
    assert parsed.metadata.parser == "markitdown"
    assert "gm:id=" in parsed.markdown
    assert parsed.pages[0].blocks[0].type.value == "title"
    assert parsed.pages[0].blocks[0].metadata["grounding_quality"] == "synthetic"

    schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string", "x-aliases": ["Invoice Number"]},
            "vendor": {"type": "string"},
            "total_amount": {"type": "number", "x-aliases": ["Total Amount"]},
            "paid": {"type": "boolean"},
        },
        "required": ["invoice_number", "total_amount"],
    }
    result = engine.extract(parsed, schema)
    assert result.schema_valid
    assert result.data["invoice_number"] == "A-42"
    assert result.data["total_amount"] == 123.45
    assert result.data["paid"] is True
    assert result.fields[0].evidence[0].page == 1

    assert engine.section(parsed)[0].title == "Invoice"
    chunks = engine.split(parsed, max_chars=80)
    assert chunks and all(chunk.block_ids for chunk in chunks)


def test_schema_builder_uses_grounded_fields(tmp_path: Path) -> None:
    sample = tmp_path / "form.txt"
    sample.write_text("Name: Ada Lovelace\nAge: 36\nApproved: Yes\n", encoding="utf-8")
    engine = _engine(tmp_path)
    parsed = engine.parse(sample, ParseOptions(use_cache=False))
    built = engine.build_schema(parsed)
    properties = built["schema"]["properties"]
    assert properties["name"]["type"] == "string"
    assert properties["age"]["type"] == "integer"
    assert properties["approved"]["type"] == "boolean"
    assert built["evidence"]["name"]
