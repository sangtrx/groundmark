# Security Policy

Please report suspected vulnerabilities privately to the project maintainers. Do not include sensitive documents, API keys, or production data in a public issue.

GroundMark processes untrusted document formats. Production deployments should isolate workers, restrict filesystem and network access, enforce resource limits, and keep MarkItDown, PyMuPDF, Docling, FastAPI, and transitive dependencies patched.

See `docs/security.md` for the threat model and deployment controls.
