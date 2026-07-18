from __future__ import annotations

from pathlib import Path

from groundmark.models import BlockType, DocumentProfile, Page


def build_profile(
    path: Path, pages: list[Page], language_hints: list[str] | None = None
) -> DocumentProfile:
    suffix = path.suffix.lower().lstrip(".") or "unknown"
    scanned = [page.number for page in pages if page.metadata.get("scanned")]
    image_heavy = [page.number for page in pages if page.metadata.get("image_heavy")]
    multi_column = [page.number for page in pages if page.metadata.get("multi_column")]
    table_heavy = [
        page.number
        for page in pages
        if sum(block.type == BlockType.TABLE for block in page.blocks) >= 2
        or sum(len(block.text) for block in page.blocks if block.type == BlockType.TABLE) > 1000
    ]
    has_forms = any(
        block.type
        in {BlockType.FORM_FIELD, BlockType.CHECKBOX, BlockType.SIGNATURE, BlockType.ATTESTATION}
        for page in pages
        for block in page.blocks
    )
    complexity = len(scanned) * 3 + len(table_heavy) * 2 + len(multi_column) + len(image_heavy)
    recommended = "accurate" if complexity >= 4 else "balanced" if complexity else "fast"
    return DocumentProfile(
        kind=suffix,
        pages=max(len(pages), 1),
        scanned_pages=scanned,
        image_heavy_pages=image_heavy,
        table_heavy_pages=table_heavy,
        multi_column_pages=multi_column,
        has_forms=has_forms,
        language_hints=language_hints or [],
        recommended_mode=recommended,
    )
