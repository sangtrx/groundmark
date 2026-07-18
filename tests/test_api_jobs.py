from __future__ import annotations

import importlib

from fastapi.testclient import TestClient


def test_sync_and_durable_async_api(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("GROUNDMARK_DATA_DIR", str(tmp_path / "api-data"))
    monkeypatch.delenv("GROUNDMARK_API_KEY", raising=False)

    import groundmark.settings as settings_module

    settings_module.get_settings.cache_clear()
    import groundmark.api as api_module

    api_module = importlib.reload(api_module)
    client = TestClient(api_module.app)

    response = client.post(
        "/v1/parse",
        files={"document": ("note.txt", b"# Note\nHello world", "text/plain")},
        data={"mode": "balanced"},
    )
    assert response.status_code == 200
    assert response.json()["metadata"]["parser"] == "markitdown"

    queued = client.post(
        "/v1/jobs/parse",
        headers={"Idempotency-Key": "api-job-test"},
        files={"document": ("job.txt", b"# Job\nDurable", "text/plain")},
    )
    assert queued.status_code == 202
    job_id = queued.json()["id"]

    from groundmark.worker import Worker

    worker = Worker(api_module.settings)
    assert worker.run_once()
    status = client.get(f"/v1/jobs/{job_id}")
    assert status.json()["status"] == "succeeded"
    result = client.get(f"/v1/jobs/{job_id}/result")
    assert result.status_code == 200
    assert "Durable" in result.json()["markdown"]
