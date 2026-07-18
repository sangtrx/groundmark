from __future__ import annotations

import hashlib
from typing import Any

from groundmark.llm import OpenAIProvider
from groundmark.models import BlockType, Page, ParsedDocument, Section


def _section_id(title: str, page: int, index: int) -> str:
    return "sec_" + hashlib.sha1(f"{title}:{page}:{index}".encode()).hexdigest()[:12]


def build_sections(pages: list[Page]) -> list[Section]:
    all_blocks = [block for page in pages for block in page.blocks]
    headings = [block for block in all_blocks if block.type in {BlockType.TITLE, BlockType.HEADING}]
    if not all_blocks:
        return []
    if not headings:
        return [
            Section(
                id=_section_id("Document", all_blocks[0].grounding.page, 0),
                title="Document",
                level=1,
                start_block_id=all_blocks[0].id,
                end_block_id=all_blocks[-1].id,
                page=all_blocks[0].grounding.page,
            )
        ]

    roots: list[Section] = []
    stack: list[Section] = []
    flat: list[Section] = []
    for index, heading in enumerate(headings):
        level = 1 if heading.type == BlockType.TITLE else heading.level or 2
        section = Section(
            id=_section_id(heading.text, heading.grounding.page, index),
            title=heading.text.strip() or f"Section {index + 1}",
            level=level,
            start_block_id=heading.id,
            page=heading.grounding.page,
        )
        while stack and stack[-1].level >= level:
            stack.pop()
        if stack:
            stack[-1].children.append(section)
        else:
            roots.append(section)
        stack.append(section)
        flat.append(section)

    positions = {block.id: index for index, block in enumerate(all_blocks)}
    for index, section in enumerate(flat):
        start = positions.get(section.start_block_id, 0)
        next_start = (
            positions.get(flat[index + 1].start_block_id, len(all_blocks))
            if index + 1 < len(flat)
            else len(all_blocks)
        )
        section.end_block_id = all_blocks[max(start, next_start - 1)].id
    return roots


def build_sections_openai(
    document: ParsedDocument,
    provider: OpenAIProvider,
    *,
    model: str | None = None,
    max_input_chars: int = 250_000,
) -> list[Section]:
    all_blocks = [block for page in document.pages for block in page.blocks]
    if not all_blocks:
        return []
    response_schema = {
        "type": "object",
        "properties": {
            "sections": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "level": {"type": "integer", "minimum": 1, "maximum": 6},
                        "start_block_id": {"type": "string"},
                        "end_block_id": {"type": "string"},
                    },
                    "required": ["title", "level", "start_block_id", "end_block_id"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["sections"],
        "additionalProperties": False,
    }
    context_lines = [
        f"[{block.id}|p{block.grounding.page}|{block.type.value}] {(block.markdown or block.text).strip()}"
        for block in all_blocks
        if (block.markdown or block.text).strip()
    ]
    context = "\n\n".join(context_lines)[:max_input_chars]
    result = provider.structured(
        name="groundmark_sections",
        schema=response_schema,
        system=(
            "Create a non-overlapping hierarchical section outline for the document. Use only exact "
            "block IDs from the input. Sections must be ordered by appearance. A section's start and "
            "end IDs are inclusive. Avoid tiny sections that contain only one ordinary paragraph."
        ),
        user="Build the document section outline:\n\n" + context,
        model=model,
    )
    positions = {block.id: index for index, block in enumerate(all_blocks)}
    valid: list[dict[str, Any]] = []
    last_start = -1
    for raw in result.data.get("sections") or []:
        start_id = str(raw.get("start_block_id", ""))
        end_id = str(raw.get("end_block_id", ""))
        if start_id not in positions or end_id not in positions:
            continue
        start, end = positions[start_id], positions[end_id]
        if start < last_start or end < start:
            continue
        last_start = start
        valid.append(
            {
                "title": str(raw.get("title") or "Section").strip(),
                "level": max(1, min(int(raw.get("level", 1)), 6)),
                "start_block_id": start_id,
                "end_block_id": end_id,
                "page": all_blocks[start].grounding.page,
            }
        )
    if not valid:
        return build_sections(document.pages)

    roots: list[Section] = []
    stack: list[Section] = []
    for index, raw in enumerate(valid):
        section = Section(
            id=_section_id(raw["title"], raw["page"], index),
            title=raw["title"],
            level=raw["level"],
            start_block_id=raw["start_block_id"],
            end_block_id=raw["end_block_id"],
            page=raw["page"],
        )
        while stack and stack[-1].level >= section.level:
            stack.pop()
        if stack:
            stack[-1].children.append(section)
        else:
            roots.append(section)
        stack.append(section)
    return roots
