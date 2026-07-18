#!/usr/bin/env bash
set -euo pipefail
groundmark parse examples/invoice.txt --output /tmp/invoice.md --json /tmp/invoice.json
groundmark extract examples/invoice.txt --schema examples/invoice.schema.json --output /tmp/extracted.json
cat /tmp/extracted.json
