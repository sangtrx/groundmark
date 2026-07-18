from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any

import fitz

from groundmark.models import BBox, Block, BlockType, Grounding, Page, ParseOptions
from groundmark.parsers.base import ParserError


class PDFLayoutExtractor:
    name = "pymupdf-layout"

    @staticmethod
    def _norm_bbox(raw: tuple[float, float, float, float], width: float, height: float) -> BBox:
        x0, y0, x1, y1 = raw
        return BBox(
            x0=max(0.0, min(x0 / width, 1.0)),
            y0=max(0.0, min(y0 / height, 1.0)),
            x1=max(0.0, min(x1 / width, 1.0)),
            y1=max(0.0, min(y1 / height, 1.0)),
        )

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\u00ad", "")
        text = re.sub(r"(?<=\w)-\n(?=\w)", "", text)
        text = re.sub(r"[ \t]+", " ", text)
        return re.sub(r"\n{3,}", "\n\n", text).strip()

    @staticmethod
    def _font_stats(raw_pages: list[dict[str, Any]]) -> tuple[float, float]:
        sizes: list[float] = []
        for page in raw_pages:
            for block in page.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span.get("text", "").strip()
                        if text:
                            sizes.extend([float(span.get("size", 10.0))] * max(1, len(text) // 8))
        return (median(sizes), max(sizes)) if sizes else (10.0, 10.0)

    @staticmethod
    def _classify(
        text: str, avg_size: float, max_size: float, body_size: float, flags: int, bbox: BBox
    ) -> tuple[BlockType, int | None]:
        one_line = "\n" not in text
        word_count = len(text.split())
        bold = bool(flags & 16)
        if bbox.y1 < 0.07:
            return BlockType.HEADER, None
        if bbox.y0 > 0.93:
            return BlockType.FOOTER, None
        if one_line and word_count <= 18 and avg_size >= max(body_size * 1.45, max_size * 0.85):
            return BlockType.TITLE, 1
        if one_line and word_count <= 24 and (avg_size >= body_size * 1.18 or bold):
            ratio = avg_size / max(body_size, 0.1)
            return BlockType.HEADING, 2 if ratio >= 1.35 else 3
        if re.match(r"^\s*(?:[-•▪◦*]|\d+[.)]|[A-Za-z][.)])\s+", text):
            return BlockType.LIST_ITEM, None
        if re.search(r"\[[ xX✓✔]\]", text):
            return BlockType.CHECKBOX, None
        if text.count("|") >= 2 or ("\t" in text and text.count("\t") >= 2):
            return BlockType.TABLE, None
        if re.match(r"^\s*(?:Figure|Fig\.|Chart|Diagram)\s+\d+", text, re.I):
            return BlockType.FIGURE, None
        if re.search(r"\b(?:signature|signed by|authorized signatory)\b", text, re.I):
            return BlockType.ATTESTATION, None
        if re.match(r"^\s*[\w ./()-]{2,40}\s*[:=]\s*\S+", text):
            return BlockType.FORM_FIELD, None
        return BlockType.PARAGRAPH, None

    @staticmethod
    def _column_key(block: dict[str, Any], page_width: float) -> tuple[int, float, float]:
        x0, y0, x1, _ = block["bbox"]
        width = x1 - x0
        if width > page_width * 0.72:
            return -1, y0, x0
        center = (x0 + x1) / 2
        return (0 if center < page_width * 0.52 else 1), y0, x0

    def extract(self, path: Path, options: ParseOptions) -> list[Page]:
        try:
            pdf = fitz.open(path)
        except Exception as exc:
            raise ParserError(f"unable to open PDF: {exc}") from exc
        if pdf.needs_pass:
            pdf.close()
            raise ParserError("password-protected PDFs are not supported")

        total_pages = len(pdf)
        start = (options.page_start or 1) - 1
        end = min(options.page_end or total_pages, total_pages)
        if start >= total_pages:
            pdf.close()
            raise ParserError("page_start exceeds PDF page count")
        if options.max_pages and end - start > options.max_pages:
            end = start + options.max_pages

        selected = [pdf[i] for i in range(start, end)]
        raw_pages = [page.get_text("dict", sort=False) for page in selected]
        body_size, max_size = self._font_stats(raw_pages)
        pages: list[Page] = []
        repeated_top: Counter[str] = Counter()
        repeated_bottom: Counter[str] = Counter()

        for absolute_index, (page, raw) in enumerate(
            zip(selected, raw_pages, strict=True), start=start + 1
        ):
            width, height = float(page.rect.width), float(page.rect.height)
            raw_blocks = [b for b in raw.get("blocks", []) if b.get("type") == 0]
            raw_blocks.sort(key=lambda b: self._column_key(b, width))
            blocks: list[Block] = []
            left_centers = 0
            right_centers = 0

            # PyMuPDF table detection gives a stronger structural signal than text heuristics.
            try:
                table_finder = page.find_tables()
                for table_index, table in enumerate(table_finder.tables):
                    rows = table.extract() or []
                    clean_rows = [[str(cell or "").strip() for cell in row] for row in rows]
                    if not clean_rows or not any(any(cell for cell in row) for row in clean_rows):
                        continue
                    width_cells = max(len(row) for row in clean_rows)
                    clean_rows = [row + [""] * (width_cells - len(row)) for row in clean_rows]
                    header = clean_rows[0]
                    markdown_rows = [
                        "| " + " | ".join(cell.replace("|", "\\|") for cell in header) + " |",
                        "| " + " | ".join("---" for _ in header) + " |",
                    ]
                    markdown_rows.extend(
                        "| " + " | ".join(cell.replace("|", "\\|") for cell in row) + " |"
                        for row in clean_rows[1:]
                    )
                    bbox = self._norm_bbox(tuple(table.bbox), width, height)
                    blocks.append(
                        Block(
                            id=f"layout-p{absolute_index}-table{table_index + 1}",
                            type=BlockType.TABLE,
                            text="\n".join("\t".join(row) for row in clean_rows),
                            markdown="\n".join(markdown_rows),
                            reading_order=0,
                            confidence=0.9,
                            grounding=Grounding(
                                page=absolute_index,
                                bbox=bbox,
                                source=self.name,
                                source_ids=[f"fitz-table:{absolute_index}:{table_index + 1}"],
                            ),
                            metadata={"rows": len(clean_rows), "columns": width_cells},
                        )
                    )
            except Exception:
                pass

            # Preserve image regions as figure chunks even when the normalizer omits them.
            try:
                for image_index, info in enumerate(page.get_image_info(xrefs=True)):
                    raw_bbox = info.get("bbox")
                    if not raw_bbox:
                        continue
                    bbox = self._norm_bbox(tuple(raw_bbox), width, height)
                    if bbox.area < 0.003:
                        continue
                    blocks.append(
                        Block(
                            id=f"layout-p{absolute_index}-image{image_index + 1}",
                            type=BlockType.FIGURE,
                            text="",
                            reading_order=0,
                            confidence=0.82,
                            grounding=Grounding(
                                page=absolute_index,
                                bbox=bbox,
                                source=self.name,
                                source_ids=[f"fitz-image:{absolute_index}:{image_index + 1}"],
                            ),
                            metadata={"xref": info.get("xref")},
                        )
                    )
            except Exception:
                pass

            # Native PDF form widgets retain field names, values, and exact coordinates.
            try:
                widgets = list(page.widgets() or [])
                for widget_index, widget in enumerate(widgets):
                    bbox = self._norm_bbox(tuple(widget.rect), width, height)
                    name = str(widget.field_name or "field").strip()
                    value = str(widget.field_value or "").strip()
                    field_text = f"{name}: {value}" if value else name
                    is_checkbox = "check" in str(widget.field_type_string or "").lower()
                    blocks.append(
                        Block(
                            id=f"layout-p{absolute_index}-widget{widget_index + 1}",
                            type=BlockType.CHECKBOX if is_checkbox else BlockType.FORM_FIELD,
                            text=field_text,
                            reading_order=0,
                            confidence=0.98,
                            grounding=Grounding(
                                page=absolute_index,
                                bbox=bbox,
                                source=self.name,
                                source_ids=[f"fitz-widget:{absolute_index}:{widget_index + 1}"],
                            ),
                            metadata={
                                "field_name": name,
                                "field_value": value,
                                "field_type": widget.field_type_string,
                                "checked": bool(value) if is_checkbox else None,
                            },
                        )
                    )
            except Exception:
                pass

            for raw_block in raw_blocks:
                spans = [
                    span for line in raw_block.get("lines", []) for span in line.get("spans", [])
                ]
                text = self._clean_text(
                    "\n".join(
                        "".join(span.get("text", "") for span in line.get("spans", []))
                        for line in raw_block.get("lines", [])
                    )
                )
                if not text:
                    continue
                bbox = self._norm_bbox(tuple(raw_block["bbox"]), width, height)
                weighted = [
                    (float(span.get("size", body_size)), max(len(span.get("text", "")), 1))
                    for span in spans
                ]
                avg_size = sum(size * weight for size, weight in weighted) / max(
                    sum(weight for _, weight in weighted), 1
                )
                flags = max((int(span.get("flags", 0)) for span in spans), default=0)
                block_type, level = self._classify(text, avg_size, max_size, body_size, flags, bbox)

                if bbox.y1 < 0.08:
                    repeated_top[text[:160]] += 1
                if bbox.y0 > 0.92:
                    repeated_bottom[text[:160]] += 1
                if bbox.x1 - bbox.x0 < 0.48:
                    if (bbox.x0 + bbox.x1) / 2 < 0.5:
                        left_centers += 1
                    else:
                        right_centers += 1

                density = len(text) / max(bbox.area * 10000, 1)
                confidence = 0.95
                if "�" in text:
                    confidence -= 0.3
                if avg_size < 5:
                    confidence -= 0.12
                if density > 5:
                    confidence -= 0.1
                if block_type in {BlockType.TABLE, BlockType.FORM_FIELD}:
                    confidence -= 0.06
                if len(text) > 10000:
                    confidence -= 0.12

                blocks.append(
                    Block(
                        id=f"layout-p{absolute_index}-b{len(blocks) + 1}",
                        type=block_type,
                        text=text,
                        level=level,
                        reading_order=len(blocks),
                        confidence=max(0.1, min(confidence, 0.99)),
                        grounding=Grounding(
                            page=absolute_index,
                            bbox=bbox,
                            source=self.name,
                            source_ids=[f"fitz:{absolute_index}:{len(blocks) + 1}"],
                        ),
                        metadata={"font_size": round(avg_size, 2), "font_flags": flags},
                    )
                )

            blocks.sort(key=lambda block: (block.grounding.bbox.y0, block.grounding.bbox.x0))
            for order, block in enumerate(blocks):
                block.reading_order = order

            images = page.get_images(full=True)
            drawings = page.get_drawings()
            text_chars = sum(len(block.text) for block in blocks)
            warnings: list[str] = []
            page_confidence = sum(block.confidence for block in blocks) / max(len(blocks), 1)
            scanned = bool(images) and text_chars < 80
            image_heavy = len(images) >= 2 or (images and text_chars < 300)
            if scanned:
                warnings.append("page appears scanned; OCR or VLM escalation recommended")
                page_confidence = min(page_confidence, 0.48)
            if not blocks:
                warnings.append("no text blocks detected")
                page_confidence = 0.15
            multi_column = left_centers >= 2 and right_centers >= 2
            pages.append(
                Page(
                    number=absolute_index,
                    width=width,
                    height=height,
                    blocks=blocks,
                    confidence=page_confidence,
                    warnings=warnings,
                    metadata={
                        "image_count": len(images),
                        "drawing_count": len(drawings),
                        "text_chars": text_chars,
                        "scanned": scanned,
                        "image_heavy": image_heavy,
                        "multi_column": multi_column,
                    },
                )
            )

        threshold = max(2, math.ceil(len(pages) * 0.6))
        top_repeated = {text for text, count in repeated_top.items() if count >= threshold}
        bottom_repeated = {text for text, count in repeated_bottom.items() if count >= threshold}
        for page in pages:
            for block in page.blocks:
                key = block.text[:160]
                if key in top_repeated:
                    block.type = BlockType.HEADER
                if key in bottom_repeated:
                    block.type = BlockType.FOOTER

        pdf.close()
        return pages
