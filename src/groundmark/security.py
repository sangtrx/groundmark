from __future__ import annotations

import mimetypes
import zipfile
from pathlib import Path


class SecurityError(ValueError):
    pass


ALLOWED_SUFFIXES = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".xls",
    ".html",
    ".htm",
    ".txt",
    ".md",
    ".markdown",
    ".csv",
    ".json",
    ".xml",
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".wav",
    ".mp3",
    ".m4a",
    ".eml",
    ".msg",
    ".epub",
}
ZIP_CONTAINER_SUFFIXES = {".docx", ".pptx", ".xlsx", ".epub"}


def _validate_zip_container(path: Path, max_bytes: int) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            if len(infos) > 10000:
                raise SecurityError("archive contains too many entries")
            total_uncompressed = sum(info.file_size for info in infos)
            if total_uncompressed > max(max_bytes * 8, 512 * 1024 * 1024):
                raise SecurityError("archive expands beyond the configured safety limit")
            for info in infos:
                normalized = Path(info.filename)
                if normalized.is_absolute() or ".." in normalized.parts:
                    raise SecurityError("archive contains unsafe paths")
                if info.compress_size and info.file_size / info.compress_size > 1000:
                    raise SecurityError("archive contains a suspicious compression ratio")
    except zipfile.BadZipFile as exc:
        raise SecurityError("invalid Office/EPUB ZIP container") from exc


def validate_local_file(
    path: Path, max_bytes: int, allowed_suffixes: set[str] | None = None
) -> None:
    allowed_suffixes = allowed_suffixes or ALLOWED_SUFFIXES
    if not path.is_file():
        raise SecurityError(f"file does not exist: {path}")
    if path.is_symlink():
        raise SecurityError("symbolic links are not accepted")
    if path.stat().st_size > max_bytes:
        raise SecurityError(f"file exceeds upload limit of {max_bytes} bytes")
    suffix = path.suffix.lower()
    if suffix not in allowed_suffixes:
        raise SecurityError(f"unsupported or disallowed file extension: {suffix or '<none>'}")
    if suffix in ZIP_CONTAINER_SUFFIXES:
        _validate_zip_container(path, max_bytes)


def guess_media_type(path: Path, provided: str | None = None) -> str | None:
    return provided or mimetypes.guess_type(path.name)[0]
