from __future__ import annotations

import threading
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)

    def inc(self, name: str, value: float = 1, **labels: str) -> None:
        key = (name, tuple(sorted((str(k), str(v)) for k, v in labels.items())))
        with self._lock:
            self._counters[key] += value

    def render_prometheus(self) -> str:
        lines = ["# GroundMark process-local metrics"]
        with self._lock:
            items = list(self._counters.items())
        for (name, labels), value in sorted(items):
            label_text = ""
            if labels:
                escaped = [
                    f'{key}="{val.replace(chr(34), chr(92) + chr(34))}"' for key, val in labels
                ]
                label_text = "{" + ",".join(escaped) + "}"
            lines.append(f"groundmark_{name}{label_text} {value}")
        return "\n".join(lines) + "\n"


metrics = Metrics()
