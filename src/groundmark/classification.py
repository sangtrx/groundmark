from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable

from groundmark.llm import OpenAIProvider
from groundmark.models import Classification, DocumentSegment, ParsedDocument

_LABELS: dict[str, tuple[str, ...]] = {
    "invoice": ("invoice", "bill to", "invoice number", "subtotal", "amount due", "tax invoice"),
    "receipt": ("receipt", "cashier", "change", "total", "payment method"),
    "contract": ("agreement", "whereas", "party", "governing law", "signature"),
    "bank_statement": (
        "account statement",
        "opening balance",
        "closing balance",
        "debit",
        "credit",
    ),
    "resume": ("experience", "education", "skills", "curriculum vitae", "employment"),
    "research_paper": ("abstract", "methodology", "references", "doi", "introduction"),
    "form": ("date of birth", "applicant", "please check", "signature", "form"),
}


def classify_document(document: ParsedDocument) -> list[Classification]:
    scores: dict[str, float] = defaultdict(float)
    evidence: dict[str, list[str]] = defaultdict(list)
    for page in document.pages:
        for block in page.blocks:
            lowered = block.text.lower()
            for label, keywords in _LABELS.items():
                hits = sum(
                    bool(re.search(rf"\b{re.escape(keyword)}\b", lowered)) for keyword in keywords
                )
                if hits:
                    scores[label] += hits * min(1.5, 0.5 + len(block.text) / 800)
                    evidence[label].append(block.id)
    if not scores:
        return [Classification(label="unknown", confidence=0.25)]
    total = sum(scores.values())
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:3]
    return [
        Classification(
            label=label,
            confidence=min(0.98, 0.35 + 0.65 * score / max(total, 1)),
            evidence_block_ids=evidence[label][:8],
        )
        for label, score in ranked
    ]


def classify_document_openai(
    document: ParsedDocument,
    provider: OpenAIProvider,
    *,
    labels: Iterable[str] | None = None,
    model: str | None = None,
    max_input_chars: int = 200_000,
) -> list[Classification]:
    allowed = sorted({str(label).strip() for label in (labels or _LABELS) if str(label).strip()})
    if not allowed:
        allowed = ["unknown"]
    response_schema = {
        "type": "object",
        "properties": {
            "classifications": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "enum": allowed},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "evidence_block_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["label", "confidence", "evidence_block_ids"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["classifications"],
        "additionalProperties": False,
    }
    records: list[str] = []
    for page in document.pages:
        for block in page.blocks:
            text = (block.markdown or block.text).strip()
            if text:
                records.append(f"[{block.id}|p{page.number}|{block.type.value}] {text}")
    context = "\n\n".join(records)[:max_input_chars]
    result = provider.structured(
        name="groundmark_classification",
        schema=response_schema,
        system=(
            "Classify the document using only the allowed labels. Return the most likely label first, "
            "at most three labels, and cite exact block IDs that support each label. Do not cite IDs "
            "that are not present in the input."
        ),
        user=f"Allowed labels: {allowed}\n\nGrounded document:\n{context}",
        model=model,
    )
    valid_ids = {block.id for page in document.pages for block in page.blocks}
    classifications: list[Classification] = []
    seen: set[str] = set()
    for item in result.data.get("classifications") or []:
        label = str(item.get("label", "")).strip()
        if label not in allowed or label in seen:
            continue
        seen.add(label)
        classifications.append(
            Classification(
                label=label,
                confidence=max(0.0, min(float(item.get("confidence", 0.0)), 1.0)),
                evidence_block_ids=[
                    str(block_id)
                    for block_id in item.get("evidence_block_ids") or []
                    if str(block_id) in valid_ids
                ][:12],
            )
        )
    return classifications or [
        Classification(label="unknown" if "unknown" in allowed else allowed[0], confidence=0.2)
    ]


def split_document(document: ParsedDocument) -> list[DocumentSegment]:
    """Conservative page-level split for concatenated document packets."""
    if not document.pages:
        return []
    boundaries = [document.pages[0].number]
    reasons: dict[int, str] = {}
    for page in document.pages[1:]:
        first_text = " ".join(block.text for block in page.blocks[:3]).lower()[:1500]
        title_like = any(
            token in first_text
            for token in ("invoice", "tax invoice", "agreement", "application form", "statement")
        )
        page_number_reset = bool(re.search(r"\bpage\s+1\s+(?:of|/)\s+\d+", first_text))
        if title_like and page_number_reset:
            boundaries.append(page.number)
            reasons[page.number] = "new title and page-number reset"
    segments: list[DocumentSegment] = []
    page_numbers = [page.number for page in document.pages]
    for index, start in enumerate(boundaries):
        end = boundaries[index + 1] - 1 if index + 1 < len(boundaries) else page_numbers[-1]
        segments.append(
            DocumentSegment(
                id=f"segment-{index + 1}",
                start_page=start,
                end_page=end,
                confidence=0.85 if start in reasons else 0.6,
                reason=reasons.get(start, "document start"),
            )
        )
    return segments
