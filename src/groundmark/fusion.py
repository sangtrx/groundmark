from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher

from groundmark.markdown_ast import MarkdownNode, parse_markdown
from groundmark.models import BBox, Block, BlockType, Grounding, Page


def _normalize(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return re.sub(r"[^\w]+", " ", text.lower(), flags=re.UNICODE).strip()


def _similarity(left: str, right: str) -> float:
    a, b = _normalize(left), _normalize(right)
    if not a or not b:
        return 0.0
    sequence = SequenceMatcher(None, a[:8000], b[:8000]).ratio()
    a_tokens, b_tokens = set(a.split()), set(b.split())
    overlap = len(a_tokens & b_tokens) / max(len(a_tokens | b_tokens), 1)
    containment = min(len(a), len(b)) / max(len(a), len(b)) if a in b or b in a else 0.0
    return 0.55 * sequence + 0.35 * overlap + 0.10 * containment


def _union(blocks: list[Block]) -> BBox:
    bbox = blocks[0].grounding.bbox
    for block in blocks[1:]:
        bbox = bbox.union(block.grounding.bbox)
    return bbox


def _synthetic_pages(nodes: list[MarkdownNode], document_sha256: str) -> list[Page]:
    denom = max(len(nodes), 1)
    blocks: list[Block] = []
    for index, node in enumerate(nodes):
        bbox = BBox(x0=0, y0=index / denom, x1=1, y1=min((index + 1) / denom, 1))
        blocks.append(
            Block(
                id=Block.stable_id(document_sha256, 1, bbox, node.text, node.type.value),
                type=node.type,
                text=node.text,
                markdown=node.markdown,
                level=node.level,
                reading_order=index,
                confidence=0.95,
                grounding=Grounding(page=1, bbox=bbox, source="markitdown"),
                metadata={"grounding_quality": "synthetic"},
            )
        )
    return [Page(number=1, width=1, height=1, blocks=blocks, confidence=0.95)]


def fuse_markdown_with_layout(
    markdown: str, layout_pages: list[Page], document_sha256: str
) -> list[Page]:
    nodes = parse_markdown(markdown)
    if not layout_pages:
        return _synthetic_pages(nodes, document_sha256)
    if not nodes:
        recovered: list[Page] = []
        for page in layout_pages:
            blocks: list[Block] = []
            for order, source in enumerate(page.blocks):
                source.id = Block.stable_id(
                    document_sha256,
                    page.number,
                    source.grounding.bbox,
                    source.text,
                    source.type.value,
                )
                source.reading_order = order
                source.metadata["grounding_quality"] = "layout-fallback"
                blocks.append(source)
            recovered.append(
                Page(
                    number=page.number,
                    width=page.width,
                    height=page.height,
                    blocks=blocks,
                    confidence=page.confidence,
                    warnings=list(page.warnings),
                    metadata=dict(page.metadata),
                )
            )
        return recovered

    flat = [block for page in layout_pages for block in page.blocks]
    used: set[int] = set()
    fused_by_page: dict[int, list[Block]] = defaultdict(list)
    cursor = 0
    last_page = layout_pages[0].number

    for node in nodes:
        matched_candidate: list[Block] | None = None
        best: tuple[float, int, int, list[Block]] | None = None
        # Search near the previous match first, then fall back to the full document.
        starts = list(range(max(0, cursor - 2), min(len(flat), cursor + 18)))
        if not starts:
            starts = list(range(len(flat)))
        for start in starts:
            if start in used:
                continue
            for width in range(1, min(8, len(flat) - start) + 1):
                indices = range(start, start + width)
                if any(index in used for index in indices):
                    break
                candidate = flat[start : start + width]
                if len({block.grounding.page for block in candidate}) > 1:
                    break
                candidate_text = "\n".join(block.text for block in candidate)
                score = _similarity(node.text, candidate_text)
                if node.type == BlockType.TABLE and any(
                    block.type == BlockType.TABLE for block in candidate
                ):
                    score += 0.08
                if node.type in {BlockType.TITLE, BlockType.HEADING} and any(
                    block.type in {BlockType.TITLE, BlockType.HEADING} for block in candidate
                ):
                    score += 0.05
                if best is None or score > best[0]:
                    best = (score, start, width, candidate)

        if best and best[0] >= 0.34:
            score, start, width, candidate = best
            matched_candidate = candidate
            for index in range(start, start + width):
                used.add(index)
            cursor = start + width
            page_no = candidate[0].grounding.page
            last_page = page_no
            bbox = _union(candidate)
            source_ids = [
                source_id for block in candidate for source_id in block.grounding.source_ids
            ]
            layout_conf = sum(block.confidence for block in candidate) / len(candidate)
            confidence = min(0.99, 0.45 + 0.35 * layout_conf + 0.20 * min(score, 1.0))
            metadata = {
                "alignment_score": round(score, 4),
                "grounding_quality": "aligned",
                "layout_types": [block.type.value for block in candidate],
            }
            grounding = Grounding(
                page=page_no,
                bbox=bbox,
                source=f"markitdown+{candidate[0].grounding.source}",
                source_ids=source_ids,
            )
        else:
            page_no = last_page
            existing = fused_by_page[page_no]
            y0 = min(0.95, (existing[-1].grounding.bbox.y1 + 0.005) if existing else 0.0)
            bbox = BBox(x0=0, y0=y0, x1=1, y1=min(1.0, y0 + 0.04))
            confidence = 0.52
            metadata = {"alignment_score": 0.0, "grounding_quality": "inferred"}
            grounding = Grounding(page=page_no, bbox=bbox, source="markitdown-unmatched")

        resolved_type = node.type
        resolved_markdown = node.markdown
        resolved_text = node.text
        if matched_candidate and all(item.type == BlockType.HEADER for item in matched_candidate):
            resolved_type = BlockType.HEADER
        elif matched_candidate and all(item.type == BlockType.FOOTER for item in matched_candidate):
            resolved_type = BlockType.FOOTER
        elif matched_candidate and any(item.type == BlockType.TABLE for item in matched_candidate):
            table_item = next(item for item in matched_candidate if item.type == BlockType.TABLE)
            resolved_type = BlockType.TABLE
            resolved_markdown = table_item.markdown or node.markdown
            resolved_text = table_item.text or node.text
        elif matched_candidate and any(
            item.type == BlockType.FORM_FIELD for item in matched_candidate
        ):
            resolved_type = BlockType.FORM_FIELD
        block = Block(
            id=Block.stable_id(document_sha256, page_no, bbox, resolved_text, resolved_type.value),
            type=resolved_type,
            text=resolved_text,
            markdown=resolved_markdown,
            level=node.level,
            reading_order=len(fused_by_page[page_no]),
            confidence=confidence,
            grounding=grounding,
            metadata=metadata,
        )
        fused_by_page[page_no].append(block)

    # Preserve non-text visual/layout elements that MarkItDown may omit.
    for index, layout_block in enumerate(flat):
        if index in used or layout_block.type not in {
            BlockType.FIGURE,
            BlockType.ATTESTATION,
            BlockType.SCAN_CODE,
            BlockType.LOGO,
            BlockType.TABLE,
            BlockType.FORM_FIELD,
            BlockType.CHECKBOX,
        }:
            continue
        page_no = layout_block.grounding.page
        bbox = layout_block.grounding.bbox
        layout_block.id = Block.stable_id(
            document_sha256, page_no, bbox, layout_block.text, layout_block.type.value
        )
        layout_block.reading_order = len(fused_by_page[page_no])
        layout_block.metadata["grounding_quality"] = "layout-only"
        fused_by_page[page_no].append(layout_block)

    pages: list[Page] = []
    layout_map = {page.number: page for page in layout_pages}
    for page_no in sorted(set(layout_map) | set(fused_by_page)):
        source = layout_map.get(page_no)
        blocks = fused_by_page.get(page_no, [])
        blocks.sort(
            key=lambda block: (
                block.grounding.bbox.y0,
                block.grounding.bbox.x0,
                block.reading_order,
            )
        )
        for order, block in enumerate(blocks):
            block.reading_order = order
        page_conf = sum(block.confidence for block in blocks) / max(len(blocks), 1)
        pages.append(
            Page(
                number=page_no,
                width=source.width if source else 1,
                height=source.height if source else 1,
                blocks=blocks,
                confidence=page_conf,
                warnings=list(source.warnings) if source else [],
                metadata=dict(source.metadata) if source else {},
            )
        )
    return pages
