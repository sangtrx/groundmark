from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from groundmark.llm import OpenAIProvider
from groundmark.models import ParseOptions
from groundmark.service import GroundMark
from groundmark.settings import Settings


class FakeResponses:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.outputs = list(outputs)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        payload = self.outputs.pop(0)
        return SimpleNamespace(
            id=f"resp_{len(self.calls)}",
            status="completed",
            output_text=json.dumps(payload),
            usage=SimpleNamespace(input_tokens=100, output_tokens=20, total_tokens=120),
        )


class FakeClient:
    def __init__(self, outputs: list[dict[str, Any]]) -> None:
        self.responses = FakeResponses(outputs)


def _provider(outputs: list[dict[str, Any]]) -> tuple[OpenAIProvider, FakeClient]:
    client = FakeClient(outputs)
    provider = OpenAIProvider(
        api_key="test-key",
        default_model="gpt-5.6-luna",
        vision_model="gpt-5.6-terra",
        store=False,
        client=client,
    )
    return provider, client


def test_settings_accept_standard_openai_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-standard")
    monkeypatch.delenv("GROUNDMARK_OPENAI_API_KEY", raising=False)
    settings = Settings(data_dir=tmp_path / "data")
    assert settings.resolved_openai_api_key == "sk-standard"


def test_openai_provider_uses_responses_structured_output_and_image() -> None:
    provider, client = _provider([{"ok": True}])
    result = provider.structured(
        name="vision_test",
        schema={
            "type": "object",
            "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"],
            "additionalProperties": False,
        },
        system="Read the image.",
        user="Return the result.",
        image_bytes=b"fake-png",
    )
    assert result.data == {"ok": True}
    call = client.responses.calls[0]
    assert call["model"] == "gpt-5.6-terra"
    assert call["store"] is False
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["strict"] is True
    content = call["input"][1]["content"]
    assert content[0]["type"] == "input_text"
    assert content[1]["type"] == "input_image"
    assert content[1]["image_url"].startswith("data:image/png;base64,")


def test_groundmark_openai_extract_classify_schema_and_section(tmp_path: Path) -> None:
    source = tmp_path / "invoice.txt"
    source.write_text(
        "# Invoice\nInvoice Number: A-42\nVendor: ACME\nTotal Amount: 123.45\n",
        encoding="utf-8",
    )
    settings = Settings(data_dir=tmp_path / "data", max_upload_mb=10)
    local_engine = GroundMark(settings)
    document = local_engine.parse(source, ParseOptions(use_cache=False))
    evidence_id = next(
        block.id for block in document.pages[0].blocks if "Invoice Number" in block.text
    )
    first_id = document.pages[0].blocks[0].id
    last_id = document.pages[0].blocks[-1].id

    outputs = [
        {
            "data": {"invoice_number": "A-42", "total_amount": 123.45},
            "evidence": [
                {
                    "path": "invoice_number",
                    "block_ids": [evidence_id],
                    "confidence": 0.98,
                    "warnings": [],
                },
                {
                    "path": "total_amount",
                    "block_ids": [evidence_id],
                    "confidence": 0.91,
                    "warnings": [],
                },
            ],
        },
        {
            "classifications": [
                {
                    "label": "invoice",
                    "confidence": 0.99,
                    "evidence_block_ids": [evidence_id],
                }
            ]
        },
        {
            "fields": [
                {
                    "name": "invoice_number",
                    "title": "Invoice Number",
                    "description": "Unique invoice identifier.",
                    "type": "string",
                    "aliases": ["Invoice Number"],
                    "required": True,
                    "evidence_block_ids": [evidence_id],
                }
            ],
            "warnings": [],
        },
        {
            "sections": [
                {
                    "title": "Invoice",
                    "level": 1,
                    "start_block_id": first_id,
                    "end_block_id": last_id,
                }
            ]
        },
    ]
    provider, client = _provider(outputs)
    engine = GroundMark(settings, openai_provider=provider)
    schema = {
        "type": "object",
        "properties": {
            "invoice_number": {"type": "string"},
            "total_amount": {"type": "number"},
        },
        "required": ["invoice_number", "total_amount"],
        "additionalProperties": False,
    }

    extraction = engine.extract(document, schema, provider="openai")
    assert extraction.schema_valid
    assert extraction.provider == "openai-responses:gpt-5.6-luna"
    assert extraction.fields[0].evidence[0].block_id == evidence_id
    assert extraction.metadata["usage"]["total_tokens"] == 120

    classifications = engine.classify(document, provider="openai", labels=["invoice", "receipt"])
    assert classifications[0].label == "invoice"
    assert classifications[0].evidence_block_ids == [evidence_id]

    built = engine.build_schema(document, provider="openai")
    assert built["schema"]["properties"]["invoice_number"]["type"] == "string"
    assert built["provider"] == "openai-responses:gpt-5.6-luna"

    sections = engine.section(document, provider="openai")
    assert sections[0].title == "Invoice"
    assert sections[0].start_block_id == first_id
    assert len(client.responses.calls) == 4
