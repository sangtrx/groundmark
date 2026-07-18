from __future__ import annotations

import hashlib

from groundmark.markdown import block_to_markdown
from groundmark.models import BlockType, Chunk, Page


def semantic_chunks(
    pages: list[Page], max_chars: int = 5000, overlap_blocks: int = 1
) -> list[Chunk]:
    blocks = [block for page in pages for block in page.blocks if block.text or block.markdown]
    chunks: list[Chunk] = []
    current = []
    current_chars = 0

    def flush() -> None:
        nonlocal current, current_chars
        if not current:
            return
        markdown = "\n\n".join(
            block_to_markdown(block, include_grounding_comments=False) for block in current
        )
        text = "\n\n".join(block.text for block in current)
        digest = hashlib.sha1("|".join(block.id for block in current).encode()).hexdigest()[:14]
        chunks.append(
            Chunk(
                id=f"chk_{digest}",
                text=text,
                markdown=markdown,
                block_ids=[block.id for block in current],
                pages=sorted({block.grounding.page for block in current}),
                token_estimate=max(1, len(text) // 4),
                confidence=sum(block.confidence for block in current) / len(current),
                metadata={
                    "starts_with": current[0].type.value,
                    "ends_with": current[-1].type.value,
                    "contains_table": any(block.type == BlockType.TABLE for block in current),
                },
            )
        )
        current = current[-overlap_blocks:] if overlap_blocks else []
        current_chars = sum(len(block.text) for block in current)

    for block in blocks:
        size = len(block.markdown or block.text)
        hard_boundary = (
            block.type in {BlockType.TITLE, BlockType.HEADING} and current_chars > max_chars * 0.45
        )
        if current and (current_chars + size > max_chars or hard_boundary):
            flush()
        current.append(block)
        current_chars += size
        if block.type == BlockType.TABLE and current_chars > max_chars * 0.75:
            flush()
    flush()
    return chunks
