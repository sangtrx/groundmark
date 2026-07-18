from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from groundmark.models import ParsedDocument


def _normalize(text: str) -> str:
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    return re.sub(r"\s+", " ", text).strip().lower()


def evaluate_against_markdown(document: ParsedDocument, reference_markdown: str) -> dict[str, Any]:
    predicted = _normalize(document.markdown)
    reference = _normalize(reference_markdown)
    text_similarity = SequenceMatcher(None, predicted[:500000], reference[:500000]).ratio()
    predicted_tables = len(re.findall(r"^\s*\|.*\|\s*$", document.markdown, flags=re.M))
    reference_tables = len(re.findall(r"^\s*\|.*\|\s*$", reference_markdown, flags=re.M))
    table_line_recall = min(predicted_tables, reference_tables) / max(reference_tables, 1)
    grounded = sum(
        block.metadata.get("grounding_quality") == "aligned"
        for page in document.pages
        for block in page.blocks
    )
    total = sum(len(page.blocks) for page in document.pages)
    return {
        "text_similarity": round(text_similarity, 4),
        "table_line_recall": round(table_line_recall, 4),
        "grounded_block_ratio": round(grounded / max(total, 1), 4),
        "document_confidence": document.confidence,
        "blocks": total,
    }


def evaluate_manifest(engine, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = []
    for item in manifest["cases"]:
        document = engine.parse(manifest_path.parent / item["input"])
        reference = (manifest_path.parent / item["reference_markdown"]).read_text(encoding="utf-8")
        cases.append(
            {
                "name": item.get("name", item["input"]),
                **evaluate_against_markdown(document, reference),
            }
        )
    keys = ("text_similarity", "table_line_recall", "grounded_block_ratio", "document_confidence")
    aggregate = {
        key: round(sum(case[key] for case in cases) / max(len(cases), 1), 4) for key in keys
    }
    return {"aggregate": aggregate, "cases": cases}
