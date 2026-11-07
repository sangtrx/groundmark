# GroundMark 0.4.0

**MarkItDown as the universal conversion backbone, upgraded with grounded layout and the OpenAI Responses API.**

GroundMark keeps Microsoft MarkItDown's broad file coverage and LLM-friendly Markdown, then adds the controls expected from a production document-intelligence service:

- page and normalized bounding-box grounding
- reading order, tables, figures, form fields, checkboxes, formulas, and scan profiling
- stable block IDs, confidence, quality diagnostics, and field-level evidence
- OpenAI Structured Outputs for schema extraction
- OpenAI vision repair for only low-confidence PDF regions
- OpenAI classification, schema proposal, and optional semantic sectioning
- local heuristic fallback when OpenAI is unavailable in `provider=auto`
- content-addressed cache, durable jobs, worker leases, metrics, auth, Docker, and CI

GroundMark is an open production architecture. Its output and evaluation contracts are designed so that layout, OCR, extraction, and repair components can be improved independently.

## Architecture

```text
                         ┌─────────────────────────────┐
Input ── validation ───▶ │ MarkItDown normalization   │
                         └──────────────┬──────────────┘
                                        │ canonical Markdown
                         ┌──────────────▼──────────────┐
                         │ Parallel layout evidence    │
                         │ • PyMuPDF native PDF        │
                         │ • Docling accurate mode     │
                         └──────────────┬──────────────┘
                                        │
                         ┌──────────────▼──────────────┐
                         │ Markdown ↔ layout fusion    │
                         │ page, bbox, reading order   │
                         └──────────────┬──────────────┘
                                        │
             ┌──────────────────────────▼──────────────────────────┐
             │ Quality gate and task router                        │
             │                                                     │
             │ Local path               OpenAI Responses API       │
             │ • deterministic parse    • strict JSON extraction   │
             │ • heuristic fallback     • classification/schema    │
             │                          • selective vision repair   │
             └──────────────┬───────────────────────┬──────────────┘
                            │                       │
                    Grounded Markdown         Auditable JSON
```

MarkItDown remains the canonical normalization layer. OpenAI is not asked to reparse every document. The API receives compact grounded blocks for text tasks and cropped page regions only when vision repair is needed.

## OpenAI integration

GroundMark uses the **Responses API**, not the legacy Chat Completions integration.

| Task | Input | Output control | Default behavior |
|---|---|---|---|
| Extract | grounded text blocks + JSON Schema | strict Structured Outputs | OpenAI when configured |
| Classify | grounded blocks + allowed labels | strict label enum | OpenAI when configured |
| Build schema | grounded blocks | strict field-definition schema | OpenAI when configured |
| Section | ordered grounded blocks | strict section ranges | heuristic by default; OpenAI opt-in |
| Repair | cropped PNG + existing parse | strict repair schema | only with `run_repair=true` |
| MarkItDown image descriptions | file converter plugin | MarkItDown integration | opt-in |

Production controls:

- `store=false` by default
- server-side API key only; provider credentials are not accepted through public request forms
- configurable timeout, SDK retries, input/output budgets, and maximum repair regions
- exact block-ID evidence validation
- original user JSON Schema validation after model output
- `provider=auto` can fall back to local heuristics on OpenAI errors
- `provider=openai` fails explicitly instead of silently changing provider
- OpenAI request ID, token usage, latency, and model are returned in extraction/schema metadata

Current defaults are configurable. The included `.env.example` uses `gpt-5.6-luna` for high-volume text tasks and `gpt-5.6-terra` for vision/accuracy-sensitive tasks.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Set your server-side OpenAI key:

```bash
export OPENAI_API_KEY="sk-..."
```

For Docling-backed accurate layout:

```bash
pip install -e '.[accurate]'
```

For optional MarkItDown OCR plugins:

```bash
pip install -e '.[vision]'
```

## Configuration

```dotenv
OPENAI_API_KEY=
OPENAI_BASE_URL=https://api.openai.com/v1
GROUNDMARK_OPENAI_MODEL=gpt-5.6-luna
GROUNDMARK_OPENAI_ACCURATE_MODEL=gpt-5.6-terra
GROUNDMARK_OPENAI_VISION_MODEL=gpt-5.6-terra

GROUNDMARK_OPENAI_STORE=false
GROUNDMARK_OPENAI_FALLBACK_ON_ERROR=true
GROUNDMARK_OPENAI_TIMEOUT_SECONDS=180
GROUNDMARK_OPENAI_MAX_RETRIES=3
GROUNDMARK_OPENAI_MAX_INPUT_CHARS=400000
GROUNDMARK_OPENAI_MAX_OUTPUT_TOKENS=12000
GROUNDMARK_OPENAI_MAX_REPAIR_REGIONS=20

GROUNDMARK_OPENAI_DEFAULT_FOR_EXTRACT=true
GROUNDMARK_OPENAI_DEFAULT_FOR_CLASSIFY=true
GROUNDMARK_OPENAI_DEFAULT_FOR_SCHEMA=true
GROUNDMARK_OPENAI_DEFAULT_FOR_SECTION=false
```

`OPENAI_API_KEY` is the recommended standard. `GROUNDMARK_OPENAI_API_KEY` and `GROUNDMARK_OPENAI_BASE_URL` are also accepted for namespaced service deployments.

## Offline evidence audit

Verify persisted source references **without an API key**:

```bash
groundmark parse examples/invoice.txt --json /tmp/invoice.document.json --output /tmp/invoice.md
groundmark extract examples/invoice.txt --schema examples/invoice.schema.json --provider heuristic --output /tmp/invoice.extraction.json
groundmark audit /tmp/invoice.document.json /tmp/invoice.extraction.json --output /tmp/invoice.audit.json
```

The audit validates block IDs, source pages and boxes, quotes and data-path
consistency. It does **not** prove that an extracted value is semantically
entailed by its citation. See [Evidence Audit](docs/evidence-audit.md) for
the report schema, limits, and CI exit codes.

## CLI

```bash
# Local MarkItDown + grounded layout
groundmark parse report.pdf \
  --mode balanced \
  --output report.md \
  --json report.grounded.json

# Selective OpenAI vision repair for weak regions
 groundmark parse difficult-scan.pdf \
  --mode accurate \
  --repair \
  --max-repair-regions 12 \
  --output repaired.md \
  --json repaired.grounded.json

# OpenAI is selected automatically when a key is configured
groundmark extract invoice.pdf \
  --schema examples/invoice.schema.json \
  --provider auto \
  --output invoice.extraction.json

# Force behavior explicitly
groundmark extract invoice.pdf --schema examples/invoice.schema.json --provider openai
groundmark extract invoice.pdf --schema examples/invoice.schema.json --provider heuristic

# OpenAI schema proposal and classification
 groundmark build-schema invoice.pdf --provider openai --output draft.schema.json
 groundmark classify packet.pdf --provider openai --labels invoice,receipt,contract
 groundmark section report.pdf --provider openai
```

## API

```bash
groundmark serve --port 8000
groundmark-worker
```

Open `http://localhost:8000/docs` for the generated OpenAPI documentation.

### Parse with selective repair

```bash
curl -X POST http://localhost:8000/v1/parse \
  -H "Authorization: Bearer change-me" \
  -F document=@report.pdf \
  -F mode=accurate \
  -F run_repair=true \
  -F max_repair_regions=12
```

### Structured extraction

```bash
curl -X POST http://localhost:8000/v1/extract \
  -H "Authorization: Bearer change-me" \
  -F document=@invoice.pdf \
  -F 'schema={"type":"object","properties":{"invoice_number":{"type":"string"},"total":{"type":"number"}},"required":["invoice_number","total"],"additionalProperties":false}' \
  -F provider=openai
```

Response fields include exact source evidence:

```json
{
  "path": "invoice_number",
  "value": "INV-100",
  "confidence": 0.98,
  "evidence": [
    {
      "block_id": "blk_8c0d...",
      "page": 1,
      "bbox": {"x0": 0.08, "y0": 0.12, "x1": 0.46, "y1": 0.16},
      "quote": "Invoice Number: INV-100"
    }
  ]
}
```

### Classification

```bash
curl -X POST http://localhost:8000/v1/classify \
  -F document=@packet.pdf \
  -F provider=openai \
  -F 'labels=["invoice","receipt","contract"]'
```

### Schema generation and sectioning

```bash
curl -X POST http://localhost:8000/v1/extract/build-schema \
  -F document=@form.pdf \
  -F provider=openai

curl -X POST http://localhost:8000/v1/section \
  -F document=@report.pdf \
  -F provider=openai
```

### Durable async parse

```bash
curl -X POST http://localhost:8000/v1/jobs/parse \
  -H "Idempotency-Key: customer-42-report-2026-07" \
  -F document=@report.pdf \
  -F mode=accurate \
  -F run_repair=true

curl http://localhost:8000/v1/jobs/JOB_ID
curl http://localhost:8000/v1/jobs/JOB_ID/result
```

## Processing modes

| Mode | Normalization | Layout | Intended use |
|---|---|---|---|
| `fast` | MarkItDown | native PDF layout | clean digital files and indexing |
| `balanced` | MarkItDown | native PDF; scans may escalate to Docling | default traffic |
| `accurate` | MarkItDown | Docling when available, native fallback | complex tables, scans, multi-column PDFs |

`mode` controls document parsing. OpenAI task selection is controlled separately by `provider` and task-specific model overrides.

## Output grounding

Each block includes:

```json
{
  "id": "blk_8c0d...",
  "type": "table",
  "markdown": "| Item | Amount |",
  "confidence": 0.91,
  "grounding": {
    "page": 1,
    "bbox": {"x0": 0.08, "y0": 0.31, "x1": 0.92, "y1": 0.71},
    "source": "markitdown+pymupdf-layout"
  },
  "metadata": {
    "alignment_score": 0.88,
    "grounding_quality": "aligned"
  }
}
```

Grounding quality values:

- `aligned`: MarkItDown element matched page-layout evidence
- `inferred`: content is preserved but exact alignment failed
- `layout-only`: visual/structured evidence was absent from MarkItDown
- `layout-fallback`: layout/OCR recovered content when MarkItDown was empty
- `synthetic`: non-paginated format without a layout provider

## Failure behavior

| Request | Missing key | OpenAI error |
|---|---|---|
| `provider=auto` | local heuristic | local heuristic when fallback enabled |
| `provider=heuristic` | local heuristic | not applicable |
| `provider=openai` | HTTP/CLI error | HTTP/CLI error |
| parse without repair | local parse | not applicable |
| parse with repair | parse continues with repair log | failed regions remain unchanged |

This avoids a network outage breaking basic document conversion while ensuring explicit OpenAI requests never silently return a different provider.

## Security and privacy

- local-only `MarkItDown.convert_local()`
- no server-side URL ingestion endpoint
- provider API key is read from server environment, never returned by capabilities or logs
- OpenAI `store=false` by default
- upload size and extension controls
- symlink rejection
- Office/EPUB ZIP traversal, expansion-size, entry-count, and compression-ratio checks
- optional GroundMark API authentication
- non-root Docker user
- stable hashes and idempotency keys
- vision sends only selected region crops, not the whole PDF, unless a page-level scan candidate is explicitly selected

See [docs/security.md](docs/security.md).

## Docker

```bash
cp .env.example .env
# Add OPENAI_API_KEY to .env
docker compose up --build
```

Build with Docling:

```bash
docker build --build-arg INSTALL_EXTRAS=accurate -t groundmark:accurate .
```

## Tests

```bash
ruff check src tests
pytest -q
```

OpenAI tests use a fake Responses API client. They verify strict JSON Schema payloads, `store=false`, image inputs, evidence validation, extraction, classification, schema generation, and sectioning without spending API credits.

## Evaluation priorities

Production evaluation should be split by document family and include:

- normalized text similarity
- reading-order edit distance
- layout IoU/mAP
- table cell precision/recall/F1
- extraction exact match and normalized numeric match
- invalid evidence-ID rate
- unsupported-value/hallucination rate
- confidence calibration error
- repair acceptance precision
- latency and token cost per page/document

The included manifest runner covers basic text, table-line recall, grounding ratio, and document confidence. Extend it with representative invoices, financial statements, research papers, forms, Vietnamese documents, scans, handwriting, and slide decks.

## Upstream projects

- Microsoft MarkItDown: <https://github.com/microsoft/markitdown>
- OpenAI API: <https://developers.openai.com/api/>
- Docling: <https://github.com/docling-project/docling>
- PyMuPDF: <https://github.com/pymupdf/PyMuPDF>

## License

Apache-2.0. Upstream dependencies and optional models retain their own licenses.
