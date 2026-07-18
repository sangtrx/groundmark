# Changelog

## 0.4.0 — 2026-07-18

- Replaced generic OpenAI-compatible task calls with the official OpenAI Python SDK and Responses API.
- Added strict Structured Outputs for extraction, schema proposal, classification, sectioning, and vision repair.
- Added OpenAI vision repair for cropped low-confidence PDF regions with conservative apply rules.
- Added grounded evidence validation so model-produced block IDs must exist in the parsed document.
- Added caller JSON Schema validation after structured extraction.
- Added `auto`, `openai`, and `heuristic` provider modes with explicit fallback semantics.
- Added model, timeout, retry, input/output budget, storage, and repair-region settings.
- Added token usage, latency, model, and response/request identifiers to task metadata.
- Added support for both `OPENAI_API_KEY` and `GROUNDMARK_OPENAI_API_KEY`.
- Added OpenAI integration tests using a fake Responses client; CI does not require API credits.
- Kept MarkItDown and layout parsing local, with OpenAI as a selective semantic/repair layer.

## 0.3.0 — 2026-07-18

- Made MarkItDown the canonical normalization backbone.
- Added Markdown-to-layout fusion with deterministic grounded block IDs.
- Added native PDF tables, figures, forms, scan profiling, reading order, and repeated marginalia detection.
- Added optional Docling accurate mode with native fallback.
- Added schema extraction with evidence and draft schema generation.
- Added classify, section, semantic split, and multi-document split APIs.
- Added content-addressed cache and durable SQLite WAL jobs with worker leases.
- Added API-key auth, streaming upload limits, ZIP-container checks, metrics, probes, Docker, CI, and browser console.
