# GroundMark 0.4.0 Build Report

Build date: 2026-07-18

## Release scope

This release keeps MarkItDown as the canonical local normalization backbone and adds the official OpenAI Python SDK/Responses API as a selective semantic and vision layer.

Implemented OpenAI tasks:

- strict JSON Schema extraction with grounded field evidence
- document classification constrained to caller-provided labels
- extraction-schema proposal with evidence block IDs
- optional semantic section ranges
- cropped low-confidence PDF region repair using image input
- optional MarkItDown plugin client reuse

Provider modes are `auto`, `openai`, and `heuristic`. Basic parsing remains local and does not require an API key.

## Validated

- Ruff format/check: passed
- Python compileall: passed
- pytest with external plugin autoload disabled: **12 passed**
- OpenAI fake Responses client contract tests: passed
- strict `text.format` JSON Schema payload: passed
- image input data URI payload: passed
- `store=false` default: passed
- evidence block-ID mapping: passed
- original caller JSON Schema validation: passed
- extraction/classification/schema/section OpenAI service routing: passed
- standard `OPENAI_API_KEY` and namespaced key resolution: passed
- MarkItDown text conversion: passed
- MarkItDown + PyMuPDF PDF grounding: passed
- native PDF table detection: passed
- content-addressed cache: passed
- durable async job lifecycle: passed
- synchronous API parse: passed
- wheel build: passed
- clean virtualenv dependency installation from wheel: passed
- official OpenAI SDK 2.46.0 client initialization: passed
- installed CLI parse and capabilities smoke tests: passed

Built wheel:

- `dist/groundmark-0.4.0-py3-none-any.whl`

## Test-runner note

The host image globally injects pytest instrumentation plugins. Aggregate test runs intermittently stalled inside that external plugin layer. Running the repository suite with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` completed normally with 12 passing tests. Individual repository tests also passed. CI should install only the declared development dependencies or disable unrelated global plugins.

## Not executed

- A live OpenAI request was not made because no user/provider credential was supplied. The release validates SDK initialization and exact mocked Responses API contracts without consuming API credits.
- Docker image build was not executed because the environment does not expose a Docker daemon.
- Docling model-quality validation was not rerun; it remains optional and requires model artifact access.
- Large-scale document accuracy, calibration, latency, and token-cost evaluation requires a representative private benchmark set.

## Production acceptance gates

Before high-stakes deployment, run:

1. live OpenAI canary calls in the target account/region;
2. golden-set extraction and evidence accuracy evaluation;
3. prompt-injection and malformed-document tests;
4. load tests with provider rate limits and worker concurrency;
5. token-cost and fallback-rate dashboards;
6. document retention/data-residency review.
