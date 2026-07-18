from __future__ import annotations

import re

from groundmark.models import Block, BlockType, Page


def _table_from_text(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if all("|" in line for line in lines):
        normalized = [line if line.startswith("|") else f"| {line} |" for line in lines]
        if len(normalized) == 1:
            cells = [cell.strip() for cell in normalized[0].strip("|").split("|")]
            normalized.append("| " + " | ".join("---" for _ in cells) + " |")
        elif not re.match(r"^\|?\s*:?-+", normalized[1]):
            cells = [cell.strip() for cell in normalized[0].strip("|").split("|")]
            normalized.insert(1, "| " + " | ".join("---" for _ in cells) + " |")
        return "\n".join(normalized)
    return "```text\n" + text + "\n```"


def block_to_markdown(block: Block, include_grounding_comments: bool = True) -> str:
    if block.markdown:
        rendered = block.markdown.strip()
    else:
        text = block.text.strip()
        if block.type == BlockType.TITLE:
            rendered = f"# {text}"
        elif block.type == BlockType.HEADING:
            rendered = f"{'#' * (block.level or 2)} {text}"
        elif block.type == BlockType.LIST_ITEM:
            rendered = text if re.match(r"^(?:[-*+] |\d+[.)] )", text) else f"- {text}"
        elif block.type == BlockType.TABLE:
            rendered = _table_from_text(text)
        elif block.type == BlockType.FIGURE:
            rendered = f"> **Figure:** {text or 'Visual element'}"
        elif block.type == BlockType.FORMULA:
            rendered = f"$$\n{text}\n$$"
        elif block.type == BlockType.CODE:
            rendered = f"```\n{text}\n```"
        elif block.type == BlockType.CHECKBOX:
            checked = "x" if block.metadata.get("checked") else " "
            rendered = f"- [{checked}] {text}"
        elif block.type in {BlockType.ATTESTATION, BlockType.SIGNATURE}:
            rendered = f"> **Attestation:** {text or 'Signature or seal detected'}"
        elif block.type == BlockType.SCAN_CODE:
            rendered = f"> **Scan code:** {text or 'QR/barcode detected'}"
        else:
            rendered = text

    if not include_grounding_comments:
        return rendered
    grounding = block.grounding
    bbox = grounding.bbox
    comment = (
        f"<!-- gm:id={block.id};page={grounding.page};"
        f"bbox={bbox.x0:.4f},{bbox.y0:.4f},{bbox.x1:.4f},{bbox.y1:.4f};"
        f"confidence={block.confidence:.3f};type={block.type.value};source={grounding.source} -->"
    )
    return f"{comment}\n{rendered}"


def pages_to_markdown(
    pages: list[Page],
    include_grounding_comments: bool = True,
    include_headers_footers: bool = False,
) -> str:
    sections: list[str] = []
    for page in pages:
        rendered = [
            block_to_markdown(block, include_grounding_comments)
            for block in page.blocks
            if include_headers_footers or block.type not in {BlockType.HEADER, BlockType.FOOTER}
        ]
        if rendered:
            sections.append(f"<!-- gm:page={page.number} -->\n" + "\n\n".join(rendered))
    return ("\n\n---\n\n".join(sections).strip() + "\n") if sections else ""
