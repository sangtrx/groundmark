from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any

from groundmark.models import BlockType, Page


def _table_quality(text: str) -> tuple[float, list[str]]:
    rows = [row for row in text.splitlines() if row.strip()]
    if not rows:
        return 0.0, ["empty table"]
    widths = [len([cell for cell in re.split(r"\||\t", row) if cell.strip()]) for row in rows]
    reasons: list[str] = []
    score = 0.9
    if widths and max(widths) != min(widths):
        score -= 0.25
        reasons.append("inconsistent table column count")
    if len(rows) == 1:
        score -= 0.2
        reasons.append("single-row table")
    return max(score, 0), reasons


def evaluate_quality(pages: list[Page]) -> dict[str, Any]:
    blocks = [block for page in pages for block in page.blocks]
    warnings = [warning for page in pages for warning in page.warnings]
    type_counts = Counter(block.type.value for block in blocks)
    low_confidence = [block.id for block in blocks if block.confidence < 0.72]
    suspicious: list[dict[str, Any]] = []
    grounding_counts = Counter(
        str(block.metadata.get("grounding_quality", "unknown")) for block in blocks
    )

    weighted_score = 0.0
    total_weight = 0.0
    for block in blocks:
        reasons: list[str] = []
        score = block.confidence
        weight = max(1.0, math.sqrt(max(len(block.text), 1)))
        if block.type == BlockType.TABLE:
            table_score, table_reasons = _table_quality(block.markdown or block.text)
            score = min(score, table_score)
            reasons.extend(table_reasons)
            weight *= 1.6
        if "�" in block.text:
            score -= 0.25
            reasons.append("replacement characters detected")
        if len(block.text) > 12000:
            score -= 0.12
            reasons.append("oversized block may contain merged columns")
        if re.search(r"\b\w\s+\w\s+\w\s+\w\b", block.text):
            score -= 0.12
            reasons.append("possible OCR character spacing")
        if block.metadata.get("grounding_quality") == "inferred":
            score -= 0.15
            reasons.append("MarkItDown element could not be aligned to page layout")
        score = max(0, min(score, 1))
        weighted_score += score * weight
        total_weight += weight
        if reasons:
            suspicious.append({"block_id": block.id, "reasons": reasons})

    confidence = weighted_score / total_weight if total_weight else 0.0
    page_coverage = sum(bool(page.blocks) for page in pages) / max(len(pages), 1)
    confidence *= 0.7 + 0.3 * page_coverage
    if warnings:
        confidence *= max(0.65, 1 - min(len(warnings), 10) * 0.025)

    return {
        "confidence": round(max(0, min(confidence, 1)), 4),
        "pages": len(pages),
        "blocks": len(blocks),
        "block_types": dict(type_counts),
        "grounding": dict(grounding_counts),
        "low_confidence_blocks": low_confidence,
        "suspicious_blocks": suspicious,
        "warnings": warnings,
        "repair_recommended": bool(low_confidence or suspicious or warnings),
        "coverage": round(page_coverage, 4),
    }
