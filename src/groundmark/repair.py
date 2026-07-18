from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fitz

from groundmark.llm import LLMProviderError, OpenAIProvider
from groundmark.models import Block, BlockType, Page, ParseOptions


@dataclass(slots=True)
class RepairCandidate:
    page: Page
    block: Block | None
    reason: str
    priority: float


class SelectiveRepair:
    """Repair only uncertain regions through OpenAI vision using the Responses API."""

    def __init__(self, provider: OpenAIProvider | None = None, max_regions: int = 20) -> None:
        self.provider = provider
        self.max_regions = max_regions

    def candidates(self, pages: list[Page], threshold: float) -> list[RepairCandidate]:
        candidates: list[RepairCandidate] = []
        for page in pages:
            if page.metadata.get("scanned") and page.confidence < threshold:
                candidates.append(
                    RepairCandidate(
                        page,
                        None,
                        "scanned low-confidence page",
                        priority=2.0 + (1.0 - page.confidence),
                    )
                )
                continue
            for block in page.blocks:
                complex_region = block.type in {
                    BlockType.TABLE,
                    BlockType.FIGURE,
                    BlockType.FORMULA,
                    BlockType.FORM_FIELD,
                    BlockType.ATTESTATION,
                    BlockType.CHECKBOX,
                }
                if block.confidence < threshold or (complex_region and block.confidence < 0.84):
                    priority = (1.0 - block.confidence) + (0.75 if complex_region else 0.0)
                    candidates.append(
                        RepairCandidate(
                            page, block, "low-confidence complex region", priority=priority
                        )
                    )
        return sorted(candidates, key=lambda item: item.priority, reverse=True)

    @staticmethod
    def _render_region(pdf_path: Path, page_number: int, block: Block | None) -> bytes:
        with fitz.open(pdf_path) as pdf:
            page = pdf[page_number - 1]
            clip = None
            if block is not None:
                bbox = block.grounding.bbox
                margin_x, margin_y = 0.015, 0.015
                clip = fitz.Rect(
                    max(0, bbox.x0 - margin_x) * page.rect.width,
                    max(0, bbox.y0 - margin_y) * page.rect.height,
                    min(1, bbox.x1 + margin_x) * page.rect.width,
                    min(1, bbox.y1 + margin_y) * page.rect.height,
                )
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=clip, alpha=False)
            return pixmap.tobytes("png")

    @staticmethod
    def _response_schema() -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "type": {"type": "string", "enum": [member.value for member in BlockType]},
                "markdown": {"type": "string"},
                "text": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "material_change": {"type": "boolean"},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["type", "markdown", "text", "confidence", "material_change", "warnings"],
            "additionalProperties": False,
        }

    def _call_model(self, image: bytes, existing: str, model: str | None) -> dict[str, Any]:
        if self.provider is None:
            raise LLMProviderError("OpenAI provider is not configured")
        result = self.provider.structured(
            name="groundmark_region_repair",
            schema=self._response_schema(),
            system=(
                "You repair one cropped region from a document. Transcribe only visible content. "
                "Preserve tables as GitHub Markdown when structurally safe; use HTML for merged cells. "
                "Preserve equations as LaTeX. Do not invent, normalize, translate, or calculate values. "
                "Set material_change=false when the existing output is already faithful."
            ),
            user=(
                "Compare the image against the existing parse and return the best faithful representation.\n\n"
                "EXISTING PARSE:\n" + existing[:12_000]
            ),
            model=model,
            image_bytes=image,
            image_mime_type="image/png",
            max_output_tokens=5000,
        )
        return {
            **result.data,
            "_metadata": {
                "model": result.model,
                "request_id": result.request_id,
                "usage": result.usage,
                "elapsed_ms": result.elapsed_ms,
            },
        }

    def run(self, pdf_path: Path, pages: list[Page], options: ParseOptions) -> list[str]:
        if pdf_path.suffix.lower() != ".pdf":
            return ["selective OpenAI vision repair supports PDF inputs only"]
        if self.provider is None:
            return ["repair requested but OPENAI_API_KEY is not configured"]
        logs: list[str] = []
        limit = options.max_repair_regions or self.max_regions
        candidates = self.candidates(pages, options.min_confidence_for_repair)[:limit]
        if not candidates:
            return ["no low-confidence regions required repair"]

        for candidate in candidates:
            target = candidate.block
            existing = (
                (target.markdown or target.text)
                if target
                else "\n\n".join(block.markdown or block.text for block in candidate.page.blocks)
            )
            try:
                result = self._call_model(
                    self._render_region(pdf_path, candidate.page.number, target),
                    existing,
                    options.repair_model,
                )
            except Exception as exc:
                logs.append(f"repair failed page={candidate.page.number}: {exc}")
                continue
            if target is None:
                logs.append(
                    f"page-level repair produced output for page={candidate.page.number}; "
                    "not auto-applied to avoid destructive replacement"
                )
                continue
            if not result.get("material_change"):
                logs.append(f"kept {target.id}: OpenAI found no material change")
                continue
            old_confidence = target.confidence
            proposed_confidence = max(0.0, min(float(result.get("confidence", 0.82)), 0.99))
            if proposed_confidence < old_confidence:
                logs.append(f"rejected repair for {target.id}: lower confidence")
                continue
            proposed_markdown = str(result.get("markdown") or "").strip()
            proposed_text = str(result.get("text") or "").strip()
            if not proposed_markdown and not proposed_text:
                logs.append(f"rejected repair for {target.id}: empty output")
                continue
            target.markdown = proposed_markdown or proposed_text
            target.text = proposed_text or proposed_markdown
            if result.get("type") in BlockType._value2member_map_:
                target.type = BlockType(result["type"])
            target.confidence = proposed_confidence
            target.metadata["repair"] = {
                "reason": candidate.reason,
                "provider": "openai-responses",
                "previous_confidence": old_confidence,
                **result.get("_metadata", {}),
                "warnings": result.get("warnings") or [],
            }
            logs.append(f"repaired {target.id}")
        if len(self.candidates(pages, options.min_confidence_for_repair)) > limit:
            logs.append(f"repair region budget reached: processed {limit} candidates")
        return logs
