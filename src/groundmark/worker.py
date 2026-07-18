from __future__ import annotations

import logging
import signal
import time
from pathlib import Path

from groundmark.infra.jobs import JobStore
from groundmark.infra.metrics import metrics
from groundmark.logging_utils import configure_logging
from groundmark.service import GroundMark
from groundmark.settings import Settings, get_settings


class Worker:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.engine = GroundMark(self.settings)
        self.jobs = JobStore(self.settings.resolved_database_path, self.settings.job_lease_seconds)
        self.running = True
        self.logger = logging.getLogger("groundmark.worker")

    def stop(self, *_args) -> None:
        self.running = False

    def run_once(self) -> bool:
        job = self.jobs.claim()
        if not job:
            return False
        try:
            document = self.engine.parse(
                Path(job.input_path),
                job.options,
                media_type=job.media_type,
                display_filename=job.filename,
            )
            result_dir = self.settings.data_dir / "results"
            result_dir.mkdir(parents=True, exist_ok=True)
            result_path = result_dir / f"{job.id}.json"
            result_path.write_text(document.model_dump_json(), encoding="utf-8")
            self.jobs.complete(job.id, result_path)
            Path(job.input_path).unlink(missing_ok=True)
            metrics.inc("jobs_total", status="succeeded")
            self.logger.info("job succeeded", extra={"job_id": job.id})
        except Exception as exc:
            retry = job.attempts < 3
            self.jobs.fail(job.id, str(exc), retry=retry)
            metrics.inc("jobs_total", status="retry" if retry else "failed")
            self.logger.exception("job failed", extra={"job_id": job.id})
        return True

    def run_forever(self) -> None:
        signal.signal(signal.SIGTERM, self.stop)
        signal.signal(signal.SIGINT, self.stop)
        while self.running:
            if not self.run_once():
                time.sleep(self.settings.worker_poll_seconds)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_json)
    Worker(settings).run_forever()


if __name__ == "__main__":
    main()
