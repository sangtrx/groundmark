from __future__ import annotations

import base64
import copy
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


class LLMProviderError(RuntimeError):
    """Raised when a configured LLM provider cannot return a usable result."""


@dataclass(slots=True)
class LLMResult:
    data: dict[str, Any]
    model: str
    elapsed_ms: int
    request_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)


_UNSUPPORTED_SCHEMA_KEYS = {
    "$schema",
    "$id",
    "$defs",
    "definitions",
    "default",
    "examples",
    "readOnly",
    "writeOnly",
    "deprecated",
    "patternProperties",
    "unevaluatedProperties",
    "dependentSchemas",
    "dependentRequired",
    "minContains",
    "maxContains",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
}


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") == "null":
        return schema
    if "anyOf" in schema:
        variants = list(schema["anyOf"])
        if not any(
            variant.get("type") == "null" for variant in variants if isinstance(variant, dict)
        ):
            variants.append({"type": "null"})
        return {**schema, "anyOf": variants}
    schema_type = schema.get("type")
    if schema_type == "object" or "properties" in schema:
        return {"anyOf": [schema, {"type": "null"}]}
    if isinstance(schema_type, str):
        return {**schema, "type": [schema_type, "null"]}
    if isinstance(schema_type, list) and "null" not in schema_type:
        return {**schema, "type": [*schema_type, "null"]}
    return {"anyOf": [schema, {"type": "null"}]}


def sanitize_schema_for_structured_output(
    schema: dict[str, Any],
    *,
    make_properties_nullable: bool = False,
    force_all_required: bool = True,
) -> dict[str, Any]:
    """Convert general JSON Schema into the strict subset used by Structured Outputs.

    GroundMark makes optional extraction fields nullable and requires every property in the
    model-facing schema. The original user schema is still used for final validation.
    """

    def visit(node: Any, *, property_node: bool = False) -> Any:
        if isinstance(node, list):
            return [visit(item) for item in node]
        if not isinstance(node, dict):
            return node

        cleaned: dict[str, Any] = {}
        for key, value in node.items():
            if key in _UNSUPPORTED_SCHEMA_KEYS or key.startswith("x-"):
                continue
            if key == "properties":
                properties = {
                    str(name): visit(spec, property_node=True) for name, spec in value.items()
                }
                cleaned[key] = properties
                continue
            if key in {"items", "additionalProperties"}:
                cleaned[key] = visit(value)
                continue
            if key in {"allOf", "anyOf", "oneOf", "prefixItems"}:
                cleaned[key] = [visit(item) for item in value]
                continue
            cleaned[key] = copy.deepcopy(value)

        node_type = cleaned.get("type")
        if node_type == "object" or "properties" in cleaned:
            cleaned["type"] = "object"
            properties = cleaned.setdefault("properties", {})
            cleaned["additionalProperties"] = False
            if force_all_required:
                cleaned["required"] = list(properties)
            elif "required" in node:
                cleaned["required"] = list(node["required"])
        elif node_type == "array":
            cleaned.setdefault("items", {})

        if make_properties_nullable and property_node:
            cleaned = _nullable(cleaned)
        return cleaned

    result = visit(schema)
    if not isinstance(result, dict):
        raise ValueError("JSON Schema root must be an object")
    return result


def _safe_schema_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_")
    return (cleaned or "groundmark_output")[:64]


class OpenAIProvider:
    """Thin production wrapper around the OpenAI Responses API.

    The wrapper deliberately exposes one operation: a strict structured response, optionally
    with one image input. This keeps provider behavior deterministic and easy to mock in tests.
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        default_model: str = "gpt-5.6-luna",
        vision_model: str | None = None,
        timeout_seconds: float = 180,
        max_retries: int = 3,
        max_output_tokens: int = 12000,
        store: bool = False,
        client: Any | None = None,
    ) -> None:
        if not api_key and client is None:
            raise ValueError("OpenAI API key is required")
        self.default_model = default_model
        self.vision_model = vision_model or default_model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        self.store = store
        if client is not None:
            self.client = client
        else:
            try:
                from openai import OpenAI
            except (ImportError, AttributeError) as exc:
                raise LLMProviderError(
                    "The official OpenAI Python SDK is required. Install groundmark or openai>=1.66."
                ) from exc
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=timeout_seconds,
                max_retries=max_retries,
            )

    @property
    def configured(self) -> bool:
        return True

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            raw = usage.model_dump()
        elif isinstance(usage, dict):
            raw = usage
        else:
            raw = {
                name: getattr(usage, name)
                for name in ("input_tokens", "output_tokens", "total_tokens")
                if getattr(usage, name, None) is not None
            }
        return {
            str(key): int(value)
            for key, value in raw.items()
            if isinstance(value, int) and not isinstance(value, bool)
        }

    @staticmethod
    def _output_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return str(text)
        output = getattr(response, "output", None) or []
        for item in output:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", None) or []:
                if getattr(content, "type", None) == "refusal":
                    raise LLMProviderError(
                        f"OpenAI refused the request: {getattr(content, 'refusal', '')}"
                    )
                candidate = getattr(content, "text", None)
                if candidate:
                    return str(candidate)
        raise LLMProviderError("OpenAI returned no text output")

    def structured(
        self,
        *,
        name: str,
        schema: dict[str, Any],
        system: str,
        user: str,
        model: str | None = None,
        image_bytes: bytes | None = None,
        image_mime_type: str = "image/png",
        max_output_tokens: int | None = None,
    ) -> LLMResult:
        selected_model = model or (self.vision_model if image_bytes else self.default_model)
        content: list[dict[str, Any]] = [{"type": "input_text", "text": user}]
        if image_bytes is not None:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{image_mime_type};base64,{encoded}",
                    "detail": "high",
                }
            )
        payload = {
            "model": selected_model,
            "input": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": _safe_schema_name(name),
                    "strict": True,
                    "schema": sanitize_schema_for_structured_output(schema),
                }
            },
            "max_output_tokens": max_output_tokens or self.max_output_tokens,
            "store": self.store,
        }
        started = time.perf_counter()
        try:
            response = self.client.responses.create(**payload)
        except Exception as exc:
            raise LLMProviderError(f"OpenAI Responses API request failed: {exc}") from exc

        status = getattr(response, "status", "completed")
        if status != "completed":
            detail = getattr(response, "incomplete_details", None)
            raise LLMProviderError(f"OpenAI response status={status}; details={detail}")
        raw = self._output_text(response).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise LLMProviderError("OpenAI structured output was not valid JSON") from exc
        if not isinstance(data, dict):
            raise LLMProviderError("OpenAI structured output root must be a JSON object")
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        request_id = getattr(response, "_request_id", None) or getattr(response, "id", None)
        usage = self._usage(response)
        logger.info(
            "openai_response_completed",
            extra={
                "model": selected_model,
                "elapsed_ms": elapsed_ms,
                "request_id": request_id,
                "usage": usage,
            },
        )
        return LLMResult(
            data=data,
            model=selected_model,
            elapsed_ms=elapsed_ms,
            request_id=request_id,
            usage=usage,
        )
