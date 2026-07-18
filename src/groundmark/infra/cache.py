from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from groundmark.models import ParsedDocument, ParseOptions


class CacheStore:
    def __init__(self, root: Path, version: str = "0.4.0") -> None:
        self.root = root
        self.version = version
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, document_sha256: str, options: ParseOptions) -> str:
        payload = json.dumps(
            options.model_dump(mode="json", exclude={"repair_api_key"}),
            sort_keys=True,
            separators=(",", ":"),
        )
        raw = f"{self.version}:{document_sha256}:{payload}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> ParsedDocument | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            document = ParsedDocument.model_validate_json(path.read_text(encoding="utf-8"))
            document.metadata.cache_hit = True
            return document
        except Exception:
            path.unlink(missing_ok=True)
            return None

    def put(self, key: str, document: ParsedDocument) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(f".{os.getpid()}.tmp")
        temp.write_text(document.model_dump_json(), encoding="utf-8")
        os.replace(temp, path)
