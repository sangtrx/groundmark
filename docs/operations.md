# Operations

## Probes

- `/health/live`: process is running
- `/health/ready`: MarkItDown import and job database are available
- `/metrics`: process-local Prometheus text metrics
- `/v1/capabilities`: reports whether OpenAI is configured and which models are selected, without returning secrets

## Data directories

`GROUNDMARK_DATA_DIR` contains:

- `cache/`: content-addressed parse results
- `uploads/`: queued job inputs
- `results/`: asynchronous job results
- `groundmark.sqlite3`: queue metadata

Back up only when job durability is required. Cache can be deleted safely.

## OpenAI configuration

GroundMark accepts either `OPENAI_API_KEY` or `GROUNDMARK_OPENAI_API_KEY`. The latter is useful when all service settings are namespaced. `OPENAI_BASE_URL` is also accepted, while `GROUNDMARK_OPENAI_BASE_URL` remains the explicit service setting.

Recommended production values:

```dotenv
OPENAI_API_KEY=...
GROUNDMARK_OPENAI_MODEL=gpt-5.6-luna
GROUNDMARK_OPENAI_ACCURATE_MODEL=gpt-5.6-terra
GROUNDMARK_OPENAI_VISION_MODEL=gpt-5.6-terra
GROUNDMARK_OPENAI_STORE=false
GROUNDMARK_OPENAI_TIMEOUT_SECONDS=180
GROUNDMARK_OPENAI_MAX_RETRIES=3
GROUNDMARK_OPENAI_MAX_INPUT_CHARS=400000
GROUNDMARK_OPENAI_MAX_OUTPUT_TOKENS=12000
GROUNDMARK_OPENAI_MAX_REPAIR_REGIONS=20
```

Do not place an API key in request bodies. Configure it once in the API and worker environment. Rotate through the deployment secret manager.

## Provider semantics

- `auto`: use OpenAI when configured for that task; fall back locally on provider failure when enabled.
- `openai`: require OpenAI and return a clear error on refusal, invalid output, timeout, or API failure.
- `heuristic`: deterministic local path, no OpenAI network request.

Extraction, classification, schema generation, and sectioning expose provider selection. Parse is local unless `run_repair=true`; repair sends cropped low-confidence regions, not the original PDF binary.

## Worker behavior

Workers claim one job with an atomic SQLite transaction and a lease. Expired leases are reclaimable. Failed jobs retry up to three attempts. Inputs are deleted after success; results remain until external retention cleanup.

Run API and workers separately in production. Put OpenAI-enabled workloads on worker pools with bounded concurrency. Monitor request count, latency, token usage metadata, fallback frequency, and repair-region budget exhaustion.

## Deployment guidance

For CPU-heavy accurate mode, size worker concurrency according to Docling/PyMuPDF memory use. Apply request deadlines greater than the configured OpenAI timeout only when synchronous semantic endpoints are enabled. Prefer async jobs for large documents.
