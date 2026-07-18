# Benchmarks

Create a manifest with relative input and reference Markdown files:

```json
{
  "cases": [
    {
      "name": "invoice-001",
      "input": "data/invoice-001.pdf",
      "reference_markdown": "references/invoice-001.md"
    }
  ]
}
```

Run:

```bash
groundmark evaluate benchmarks/manifest.json --output benchmark-results.json
```

The built-in metrics are smoke-test metrics. A production benchmark should add:

- text character/word error rate
- heading hierarchy accuracy
- reading-order edit distance
- layout detection mAP or IoU
- table cell precision, recall, and spanning-cell accuracy
- figure recall and caption association
- extraction field exact match and evidence correctness
- expected calibration error for confidence scores
- latency, throughput, peak memory, and cost by document family
