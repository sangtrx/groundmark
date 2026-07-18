from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from groundmark.parsers.base import ParserError


@dataclass(slots=True)
class NormalizedContent:
    markdown: str
    title: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class MarkItDownNormalizer:
    name = "markitdown"

    def __init__(
        self, enable_plugins: bool = False, llm_client: Any = None, llm_model: str | None = None
    ) -> None:
        self.enable_plugins = enable_plugins
        self.llm_client = llm_client
        self.llm_model = llm_model

    @staticmethod
    def available() -> bool:
        try:
            import markitdown  # noqa: F401

            return True
        except ImportError:
            return False

    def convert(self, path: Path) -> NormalizedContent:
        if not self.available():
            raise ParserError(
                "MarkItDown is required. Install groundmark with its default dependencies."
            )
        try:
            from markitdown import MarkItDown

            kwargs: dict[str, Any] = {"enable_plugins": self.enable_plugins}
            if self.llm_client is not None:
                kwargs.update(llm_client=self.llm_client, llm_model=self.llm_model)
            converter = MarkItDown(**kwargs)
            # convert_local avoids MarkItDown's permissive URL handling for untrusted uploads.
            result = converter.convert_local(path)
            markdown = (
                getattr(result, "markdown", None) or getattr(result, "text_content", None) or ""
            )
            title = getattr(result, "title", None)
            metadata = {
                "markitdown_type": type(result).__name__,
                "converter": getattr(result, "converter", None),
            }
        except Exception as exc:
            raise ParserError(f"MarkItDown conversion failed: {exc}") from exc
        markdown = str(markdown).replace("\r\n", "\n").strip()
        if not markdown:
            metadata["warning"] = "MarkItDown returned empty content; layout fallback required"
            return NormalizedContent(markdown="", title=title, metadata=metadata)
        return NormalizedContent(markdown=markdown + "\n", title=title, metadata=metadata)
