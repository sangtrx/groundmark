from __future__ import annotations

import re
from dataclasses import dataclass

from bs4 import BeautifulSoup

from groundmark.models import BlockType


@dataclass(slots=True)
class MarkdownNode:
    type: BlockType
    markdown: str
    text: str
    level: int | None = None


_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_LIST = re.compile(r"^(?:\s*[-*+]\s+|\s*\d+[.)]\s+)")
_FENCE = re.compile(r"^```|^~~~")
_TABLE_SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}")


def _plain(markdown: str) -> str:
    text = re.sub(r"<!--.*?-->", "", markdown, flags=re.S)
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[*_`~]", "", text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_markdown(markdown: str) -> list[MarkdownNode]:
    """Parse LLM-oriented Markdown into stable, coarse document elements.

    This intentionally preserves table/code blocks and does not attempt to be a full CommonMark parser.
    """
    lines = markdown.replace("\r\n", "\n").split("\n")
    nodes: list[MarkdownNode] = []
    i = 0
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        nonlocal paragraph
        raw = "\n".join(paragraph).strip()
        if raw:
            nodes.append(MarkdownNode(BlockType.PARAGRAPH, raw, _plain(raw)))
        paragraph = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped or stripped == "---":
            flush_paragraph()
            i += 1
            continue
        if stripped.startswith("<!--"):
            flush_paragraph()
            while i < len(lines) and "-->" not in lines[i]:
                i += 1
            i += 1
            continue
        heading = _HEADING.match(line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            text = _plain(heading.group(2))
            kind = BlockType.TITLE if level == 1 and not nodes else BlockType.HEADING
            nodes.append(MarkdownNode(kind, line.strip(), text, level))
            i += 1
            continue
        if _FENCE.match(stripped):
            flush_paragraph()
            fence = stripped[:3]
            body = [line]
            i += 1
            while i < len(lines):
                body.append(lines[i])
                if lines[i].strip().startswith(fence):
                    i += 1
                    break
                i += 1
            raw = "\n".join(body)
            nodes.append(MarkdownNode(BlockType.CODE, raw, _plain("\n".join(body[1:-1]))))
            continue
        if stripped.startswith("$$"):
            flush_paragraph()
            body = [line]
            i += 1
            while i < len(lines):
                body.append(lines[i])
                if lines[i].strip().endswith("$$"):
                    i += 1
                    break
                i += 1
            raw = "\n".join(body)
            nodes.append(MarkdownNode(BlockType.FORMULA, raw, _plain(raw)))
            continue
        if _LIST.match(line):
            flush_paragraph()
            body = [line]
            i += 1
            while i < len(lines) and (_LIST.match(lines[i]) or lines[i].startswith("  ")):
                body.append(lines[i])
                i += 1
            raw = "\n".join(body).strip()
            nodes.append(MarkdownNode(BlockType.LIST_ITEM, raw, _plain(raw)))
            continue
        if "|" in line and i + 1 < len(lines) and _TABLE_SEPARATOR.match(lines[i + 1]):
            flush_paragraph()
            body = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                body.append(lines[i])
                i += 1
            raw = "\n".join(body).strip()
            nodes.append(MarkdownNode(BlockType.TABLE, raw, _plain(raw)))
            continue
        if re.search(r"<table\b", line, flags=re.I):
            flush_paragraph()
            body = [line]
            i += 1
            while i < len(lines) and not re.search(r"</table>", lines[i], flags=re.I):
                body.append(lines[i])
                i += 1
            if i < len(lines):
                body.append(lines[i])
                i += 1
            raw = "\n".join(body)
            nodes.append(MarkdownNode(BlockType.TABLE, raw, _plain(raw)))
            continue
        if re.match(r"^!\[[^\]]*\]\([^)]*\)", stripped):
            flush_paragraph()
            nodes.append(MarkdownNode(BlockType.FIGURE, stripped, _plain(stripped)))
            i += 1
            continue
        paragraph.append(line)
        i += 1

    flush_paragraph()
    return [node for node in nodes if node.text or node.type in {BlockType.FIGURE, BlockType.TABLE}]
