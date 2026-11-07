# Evidence Audit (offline)

GroundMark exports structured, page-grounded evidence. The `audit` command
verifies that a saved extraction still points to the blocks in a **specific
saved ParsedDocument**. It runs without an OpenAI key or network calls.

## Quickstart

```bash
groundmark parse examples/invoice.txt --json /tmp/invoice.document.json --output /tmp/invoice.md
groundmark extract examples/invoice.txt --schema examples/invoice.schema.json --provider heuristic --output /tmp/invoice.extraction.json
groundmark audit /tmp/invoice.document.json /tmp/invoice.extraction.json --output /tmp/invoice.audit.json
```

Use the same source file and parsing configuration for parsing and extraction.
Field evidence references are exact block IDs. An audit for a document from a
different run with different source content should fail on unknown/mismatched
references rather than silently matching approximate text.

## What a PASS checks

- Unique page numbers and block IDs; every block's page matches its container.
- One field record per leaf path in `data`, with consistent value and non-null
  fields carrying at least one citation.
- Every cited block ID exists in the saved document.
- Citation page and normalized bounding box match the block.
- Quote occurs in the source block text or Markdown (whitespace normalized).
- The extraction does not already report an invalid schema.

`audit` emits a stable JSON report with `ok`, `document_sha256`,
`fields_checked`, `citations_checked`, and `findings`
(`code`, `location`, `message`), without embedding the source text.
Exit codes: `0` pass, `1` integrity findings, `2` invalid input or I/O
error. It is safe to use as a CI step after extraction.

## Limitations (important)

This verifies **referential integrity**, not semantic entailment:
a real quote may still fail to support a claimed value. It also does not
re-read the original document bytes, independently recompute their SHA-256,
re-run the caller's JSON Schema, establish OCR correctness, or defend against
a malicious party altering both JSON files coherently. For high-stakes use,
retain authenticated source bytes and run a separate human/model
claim-vs-evidence evaluation.

Field paths use dot notation for nested JSON objects; arrays are terminal
field values. Literal dots in JSON property names are not supported.
