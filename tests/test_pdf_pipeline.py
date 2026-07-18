from pathlib import Path

import fitz

from groundmark.models import BlockType, ParseOptions
from groundmark.service import GroundMark
from groundmark.settings import Settings


def _make_pdf(path: Path) -> None:
    pdf = fitz.open()
    page = pdf.new_page(width=420, height=600)
    page.insert_text((50, 55), "Invoice", fontsize=22)
    page.insert_text((50, 88), "Invoice Number: INV-100", fontsize=11)
    xs = [50, 220, 350]
    ys = [130, 160, 190]
    for x in xs:
        page.draw_line((x, ys[0]), (x, ys[-1]))
    for y in ys:
        page.draw_line((xs[0], y), (xs[-1], y))
    page.insert_text((60, 150), "Item", fontsize=10)
    page.insert_text((240, 150), "Amount", fontsize=10)
    page.insert_text((60, 180), "Widget", fontsize=10)
    page.insert_text((240, 180), "25.00", fontsize=10)
    pdf.save(path)
    pdf.close()


def test_pdf_uses_markitdown_and_layout_grounding(tmp_path: Path) -> None:
    path = tmp_path / "invoice.pdf"
    _make_pdf(path)
    engine = GroundMark(Settings(data_dir=tmp_path / "data", max_upload_mb=10))
    parsed = engine.parse(path, ParseOptions(mode="balanced", use_cache=False))
    assert "markitdown" in parsed.metadata.parser
    assert "pymupdf-layout" in parsed.metadata.parser
    assert parsed.profile.pages == 1
    assert parsed.pages[0].blocks
    assert all(
        0 <= block.grounding.bbox.x0 <= block.grounding.bbox.x1 <= 1
        for block in parsed.pages[0].blocks
    )
    assert any(
        block.metadata.get("grounding_quality") == "aligned" for block in parsed.pages[0].blocks
    )
    assert any(block.type == BlockType.TABLE for block in parsed.pages[0].blocks)
