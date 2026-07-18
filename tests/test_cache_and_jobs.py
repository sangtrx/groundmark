from pathlib import Path

from groundmark.infra.jobs import JobStore
from groundmark.models import ParseOptions
from groundmark.service import GroundMark
from groundmark.settings import Settings


def test_content_addressed_cache(tmp_path: Path) -> None:
    sample = tmp_path / "sample.md"
    sample.write_text("# Cached\n\nHello", encoding="utf-8")
    engine = GroundMark(Settings(data_dir=tmp_path / "data", max_upload_mb=10))
    first = engine.parse(sample, ParseOptions(use_cache=True))
    second = engine.parse(sample, ParseOptions(use_cache=True))
    assert not first.metadata.cache_hit
    assert second.metadata.cache_hit
    assert first.metadata.sha256 == second.metadata.sha256


def test_durable_job_claim_and_complete(tmp_path: Path) -> None:
    database = tmp_path / "jobs.sqlite3"
    store = JobStore(database, lease_seconds=30)
    input_path = tmp_path / "input.txt"
    input_path.write_text("hello", encoding="utf-8")
    queued = store.enqueue(input_path, "input.txt", "text/plain", ParseOptions())
    assert queued.status == "queued"
    claimed = store.claim()
    assert claimed and claimed.id == queued.id and claimed.status == "processing"
    result_path = tmp_path / "result.json"
    result_path.write_text("{}", encoding="utf-8")
    store.complete(queued.id, result_path)
    completed = store.get(queued.id)
    assert completed and completed.status == "succeeded"
