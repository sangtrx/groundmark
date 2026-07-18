from pathlib import Path

import pytest

from groundmark.security import SecurityError, validate_local_file


def test_rejects_disallowed_archive(tmp_path: Path) -> None:
    path = tmp_path / "payload.zip"
    path.write_bytes(b"PK\\x03\\x04")
    with pytest.raises(SecurityError):
        validate_local_file(path, max_bytes=1024)
