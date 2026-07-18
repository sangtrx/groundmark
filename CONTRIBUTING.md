# Contributing

1. Create a focused branch and add tests for behavioral changes.
2. Run `make lint` and `make test`.
3. Preserve the public `ParsedDocument` contract or document migrations.
4. New parser/model providers should be adapters rather than hard-coded pipeline dependencies.
5. Include benchmark cases for changes that claim accuracy improvements.
6. Never commit documents containing personal, confidential, or licensed data without permission.
