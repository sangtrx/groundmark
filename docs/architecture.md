# Architecture

## Core contracts

GroundMark separates the pipeline into stable contracts:

1. `NormalizedContent`: canonical Markdown produced by MarkItDown.
2. `Page` and `Block`: visual layout evidence with page number, normalized top-left coordinates, type, reading order, and confidence.
3. Fusion: alignment between canonical Markdown elements and layout evidence.
4. `ParsedDocument`: the public result with trace, profile, quality, grounded Markdown, sections, and chunks.
5. Provider adapters: Docling, OpenAI Responses API, cache, and durable jobs.

Downstream APIs do not depend directly on MarkItDown, PyMuPDF, Docling, or OpenAI response object internals.

## Parse request flow

1. Validate extension, size, symlink status, and ZIP containers.
2. Hash bytes and resolve the content-addressed cache key.
3. Normalize through MarkItDown using its local-file conversion API.
4. Route layout processing by file type, requested mode, and installed providers.
5. Parse Markdown into coarse elements without destroying tables, HTML tables, formulas, or code fences.
6. Align elements to nearby layout blocks using character similarity, token overlap, role agreement, and reading-order locality.
7. Preserve unmatched visual/layout-only elements.
8. Remove repeated headers and footers only after alignment.
9. Evaluate block/page/document quality.
10. Optionally crop only low-confidence PDF regions and repair them with OpenAI vision.
11. Generate deterministic IDs, grounded Markdown, sections, and semantic chunks.
12. Persist cache or durable job result.

Parsing and grounding remain local by default. OpenAI is used only for explicitly enabled repair or semantic tasks.

## OpenAI task flow

GroundMark uses one narrow provider contract around the OpenAI Responses API:

- strict JSON Schema output via `text.format.type=json_schema`
- optional `input_image` data URI for cropped-region repair
- bounded input context and output tokens
- SDK retries and timeout
- `store=false` by default
- response/request identifier, latency, and token usage recorded in result metadata

For extraction, the model receives compact grounded blocks rather than unbounded raw files. The response includes data plus exact evidence block IDs. GroundMark then:

1. rejects unknown evidence IDs;
2. maps evidence back to page/bbox/quote;
3. validates the model data against the caller's original JSON Schema;
4. either returns a validation failure or falls back according to provider mode.

`provider=auto` uses OpenAI when configured and can fall back to deterministic local behavior. `provider=openai` is strict and surfaces provider errors. `provider=heuristic` never makes a network call.

## Trust boundaries

LLM output never directly mutates source files. Vision repair is region-scoped and only replaces a block when the model reports a material change, emits non-empty content, and does not lower the block confidence. Page-level repair is currently advisory to avoid destructive whole-page replacement.

## Scaling boundaries

The included SQLite WAL queue targets one host or a shared POSIX volume. The small `JobStore` interface can be replaced by Postgres, SQS, Redis, or Kafka without changing parse workers.

For multi-host deployment, store uploads/results in object storage and place object keys in jobs. Keep OpenAI calls in isolated workers so API replicas remain latency-stable and provider concurrency can be controlled independently.
