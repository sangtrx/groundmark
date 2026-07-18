from groundmark.markdown_ast import parse_markdown
from groundmark.models import BlockType


def test_markdown_ast_preserves_structure() -> None:
    nodes = parse_markdown(
        "# Report\n\nIntro line\nSecond line\n\n| A | B |\n|---|---|\n| 1 | 2 |\n\n```python\nprint(1)\n```\n"
    )
    assert [node.type for node in nodes] == [
        BlockType.TITLE,
        BlockType.PARAGRAPH,
        BlockType.TABLE,
        BlockType.CODE,
    ]
    assert "Second line" in nodes[1].text
    assert nodes[2].markdown.startswith("| A")
