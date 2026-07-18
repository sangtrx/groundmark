from __future__ import annotations

from pathlib import Path
from typing import Any

from groundmark.models import BBox, Block, BlockType, Grounding, Page, ParseOptions
from groundmark.parsers.base import ParserError


class DoclingLayoutExtractor:
    name = "docling"

    @staticmethod
    def available() -> bool:
        try:
            import docling  # noqa: F401

            return True
        except ImportError:
            return False

    @staticmethod
    def _label_to_type(label: str) -> BlockType:
        mapping = {
            "title": BlockType.TITLE,
            "section_header": BlockType.HEADING,
            "text": BlockType.PARAGRAPH,
            "list_item": BlockType.LIST_ITEM,
            "table": BlockType.TABLE,
            "picture": BlockType.FIGURE,
            "figure": BlockType.FIGURE,
            "formula": BlockType.FORMULA,
            "code": BlockType.CODE,
            "page_header": BlockType.HEADER,
            "page_footer": BlockType.FOOTER,
            "checkbox_selected": BlockType.CHECKBOX,
            "checkbox_unselected": BlockType.CHECKBOX,
            "form": BlockType.FORM_FIELD,
        }
        return mapping.get(label.lower(), BlockType.UNKNOWN)

    def extract(self, path: Path, options: ParseOptions) -> tuple[list[Page], str | None]:
        if not self.available():
            raise ParserError(
                "Docling is not installed. Install with: pip install 'groundmark[accurate]'"
            )
        try:
            from docling.document_converter import DocumentConverter

            result = DocumentConverter().convert(path)
            document = result.document
            raw: dict[str, Any] = document.export_to_dict()
            markdown = document.export_to_markdown()
        except Exception as exc:
            raise ParserError(f"Docling conversion failed: {exc}") from exc

        dimensions: dict[int, tuple[float, float]] = {}
        for key, page in (raw.get("pages") or {}).items():
            page_no = int(key) if str(key).isdigit() else int(page.get("page_no", 1))
            size = page.get("size") or {}
            dimensions[page_no] = (float(size.get("width", 1)), float(size.get("height", 1)))

        page_blocks: dict[int, list[Block]] = {}
        items: list[dict[str, Any]] = []
        for key in ("texts", "tables", "pictures", "groups"):
            items.extend(raw.get(key) or [])

        for item in items:
            provenance = item.get("prov") or []
            if not provenance:
                continue
            prov = provenance[0]
            page_no = int(prov.get("page_no", 1))
            width, height = dimensions.get(page_no, (1.0, 1.0))
            raw_bbox = prov.get("bbox") or {}
            left = float(raw_bbox.get("l", raw_bbox.get("left", 0)))
            right = float(raw_bbox.get("r", raw_bbox.get("right", width)))
            top = float(raw_bbox.get("t", raw_bbox.get("top", 0)))
            bottom = float(raw_bbox.get("b", raw_bbox.get("bottom", height)))
            # Docling PDF coordinates are commonly bottom-left based. Sort y values so output remains valid.
            y0, y1 = sorted((top / max(height, 1), bottom / max(height, 1)))
            bbox = BBox(
                x0=max(0, min(left / max(width, 1), 1)),
                y0=max(0, min(y0, 1)),
                x1=max(0, min(right / max(width, 1), 1)),
                y1=max(0, min(y1, 1)),
            )
            label = str(item.get("label", "unknown"))
            block_type = self._label_to_type(label)
            text = str(item.get("text") or item.get("caption_text") or "").strip()
            block_markdown: str | None = None
            if block_type == BlockType.TABLE:
                data = item.get("data")
                text = text or str(data or "")
                block_markdown = text if "|" in text else None
            if not text and block_type not in {BlockType.FIGURE, BlockType.UNKNOWN}:
                continue
            order = len(page_blocks.get(page_no, []))
            page_blocks.setdefault(page_no, []).append(
                Block(
                    id=f"docling-p{page_no}-b{order + 1}",
                    type=block_type,
                    text=text,
                    markdown=block_markdown,
                    reading_order=order,
                    confidence=0.91,
                    grounding=Grounding(
                        page=page_no,
                        bbox=bbox,
                        source=self.name,
                        source_ids=[str(item.get("self_ref") or item.get("id") or order)],
                    ),
                    metadata={"docling_label": label},
                )
            )

        page_numbers = sorted(set(dimensions) | set(page_blocks)) or [1]
        pages: list[Page] = []
        for page_no in page_numbers:
            if options.page_start and page_no < options.page_start:
                continue
            if options.page_end and page_no > options.page_end:
                continue
            width, height = dimensions.get(page_no, (1.0, 1.0))
            blocks = page_blocks.get(page_no, [])
            blocks.sort(key=lambda block: (block.grounding.bbox.y0, block.grounding.bbox.x0))
            for index, block in enumerate(blocks):
                block.reading_order = index
            pages.append(
                Page(number=page_no, width=width, height=height, blocks=blocks, confidence=0.91)
            )
        return pages, markdown
