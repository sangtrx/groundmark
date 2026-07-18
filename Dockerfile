FROM python:3.12-slim AS runtime

ARG INSTALL_EXTRAS=""
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    GROUNDMARK_DATA_DIR=/data

RUN groupadd --gid 10001 groundmark \
    && useradd --uid 10001 --gid groundmark --create-home groundmark

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY web ./web

RUN if [ -n "$INSTALL_EXTRAS" ]; then \
      pip install ".[${INSTALL_EXTRAS}]"; \
    else \
      pip install .; \
    fi \
    && mkdir -p /data \
    && chown -R groundmark:groundmark /data /app

USER groundmark
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=3)"

CMD ["uvicorn", "groundmark.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--proxy-headers"]
