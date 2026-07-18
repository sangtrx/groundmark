# Security threat model

## Untrusted inputs

Document converters process complex containers and parsers may have vulnerabilities. Treat every upload as hostile.

GroundMark mitigations:

- calls MarkItDown's local-file API instead of permissive URL conversion
- rejects symbolic links and unknown extensions
- limits upload bytes during streaming
- validates Office and EPUB ZIP containers for traversal, expansion size, entry count, and extreme compression ratio
- does not expose a URL-fetch endpoint
- runs as a non-root container user

Recommended deployment controls:

- use a read-only root filesystem and writable `/data` volume
- enforce CPU, memory, process, and request-duration limits
- isolate conversion workers from metadata services and private networks
- scan uploads when required
- encrypt volumes and define retention policies
- set `GROUNDMARK_API_KEY` or use authenticated ingress
- never log document content, prompts, Authorization headers, or API keys

## OpenAI data boundary

Local parsing, layout extraction, fusion, quality scoring, chunking, and heuristic tasks do not require OpenAI. Network transfer occurs only for OpenAI semantic tasks or selective vision repair.

- extraction/classification/schema/sectioning send a bounded text representation containing grounded block IDs and content
- repair sends a rendered crop of a low-confidence region
- `store=false` is set by default in Responses API requests
- API keys are server-side settings and are not accepted by public task endpoints
- response identifiers and token counts may be retained for audit; document text should not be logged

Deployments with strict data residency or no-egress requirements should use `provider=heuristic`, disable repair, and block outbound networking.

## Model risks

LLM/VLM output is untrusted. GroundMark validates strict JSON output, checks evidence IDs, validates extracted data against the caller's original JSON Schema, clamps confidence, preserves source grounding, and avoids destructive whole-page replacement. Applications must still apply business and domain constraints before taking actions based on extracted values.

Prompt injection inside a document is treated as document content, not an instruction. Provider system prompts explicitly require faithful transcription/extraction and the code constrains outputs to task-specific schemas.
