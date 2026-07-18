from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from jsonschema import Draft202012Validator

from groundmark.llm import OpenAIProvider, sanitize_schema_for_structured_output
from groundmark.models import (
    ExtractedField,
    ExtractionResult,
    FieldEvidence,
    Page,
    ParsedDocument,
)


class HeuristicExtractor:
    name = "heuristic"

    @staticmethod
    def _flatten_properties(
        schema: dict[str, Any], prefix: str = ""
    ) -> Iterable[tuple[str, dict[str, Any]]]:
        for name, spec in (schema.get("properties") or {}).items():
            path = f"{prefix}.{name}" if prefix else name
            if spec.get("type") == "object":
                yield from HeuristicExtractor._flatten_properties(spec, path)
            else:
                yield path, spec

    @staticmethod
    def _aliases(path: str, spec: dict[str, Any]) -> list[str]:
        name = path.rsplit(".", 1)[-1]
        aliases = [name, name.replace("_", " "), name.replace("-", " ")]
        aliases.extend(spec.get("x-aliases") or [])
        if spec.get("title"):
            aliases.append(str(spec["title"]))
        return sorted({alias.strip() for alias in aliases if alias.strip()}, key=len, reverse=True)

    @staticmethod
    def _coerce(value: str, spec: dict[str, Any]) -> Any:
        kind = spec.get("type", "string")
        cleaned = value.strip().strip("|;, ")
        if kind == "number":
            number = re.sub(r"[^0-9.\-]", "", cleaned.replace(",", ""))
            return float(number) if number else None
        if kind == "integer":
            number = re.sub(r"[^0-9\-]", "", cleaned.replace(",", ""))
            return int(number) if number else None
        if kind == "boolean":
            lowered = cleaned.lower()
            if lowered in {"yes", "true", "checked", "1", "x", "✓"}:
                return True
            if lowered in {"no", "false", "unchecked", "0"}:
                return False
            return None
        if kind == "array":
            return [part.strip() for part in re.split(r"[,;]", cleaned) if part.strip()]
        return cleaned

    @staticmethod
    def _set_path(target: dict[str, Any], path: str, value: Any) -> None:
        parts = path.split(".")
        cursor = target
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
        cursor[parts[-1]] = value

    def extract(self, pages: list[Page], schema: dict[str, Any]) -> ExtractionResult:
        blocks = [block for page in pages for block in page.blocks]
        data: dict[str, Any] = {}
        fields: list[ExtractedField] = []
        for path, spec in self._flatten_properties(schema):
            best = None
            for block in blocks:
                for line in block.text.splitlines() or [block.text]:
                    for alias in self._aliases(path, spec):
                        match = re.search(
                            rf"(?i)(?:^|\b){re.escape(alias)}\s*(?::|=|-|\|)\s*(.+)$",
                            line.strip(),
                        )
                        if match:
                            confidence = 0.91 if line.lower().startswith(alias.lower()) else 0.82
                            candidate = (block, line.strip(), match.group(1), confidence)
                            if best is None or candidate[3] > best[3]:
                                best = candidate
            if best:
                block, quote, raw_value, confidence = best
                value = self._coerce(raw_value, spec)
                evidence = [
                    FieldEvidence(
                        block_id=block.id,
                        page=block.grounding.page,
                        bbox=block.grounding.bbox,
                        quote=quote[:500],
                    )
                ]
                warnings = [] if value is not None else ["matched label but could not coerce value"]
            else:
                value, confidence, evidence = None, 0.0, []
                warnings = ["field not found by conservative baseline extractor"]
            self._set_path(data, path, value)
            fields.append(
                ExtractedField(
                    path=path,
                    value=value,
                    confidence=confidence,
                    evidence=evidence,
                    warnings=warnings,
                )
            )

        errors = sorted(
            Draft202012Validator(schema).iter_errors(data), key=lambda error: list(error.path)
        )
        return ExtractionResult(
            data=data,
            fields=fields,
            schema_valid=not errors,
            validation_errors=[
                f"{'.'.join(map(str, error.path)) or '$'}: {error.message}" for error in errors
            ],
            provider=self.name,
        )


def _value_at_path(data: dict[str, Any], path: str) -> Any:
    cursor: Any = data
    for part in path.split("."):
        cursor = cursor.get(part) if isinstance(cursor, dict) else None
    return cursor


def _compact_document_context(
    document: ParsedDocument,
    schema: dict[str, Any] | None,
    max_chars: int,
) -> str:
    records: list[tuple[int, int, str]] = []
    aliases: set[str] = set()
    if schema:
        for path, spec in HeuristicExtractor._flatten_properties(schema):
            aliases.update(alias.lower() for alias in HeuristicExtractor._aliases(path, spec))

    absolute_order = 0
    for page in document.pages:
        for block in page.blocks:
            content = (block.markdown or block.text).strip()
            if not content:
                continue
            lowered = content.lower()
            score = 0
            if page.number == 1:
                score += 3
            if block.type.value in {"title", "heading", "table", "form_field", "checkbox"}:
                score += 5
            score += min(12, sum(alias in lowered for alias in aliases) * 4)
            line = (
                f"[block_id={block.id}; page={page.number}; type={block.type.value}; "
                f"confidence={block.confidence:.3f}]\n{content}"
            )
            records.append((score, absolute_order, line))
            absolute_order += 1

    full = "\n\n".join(line for _score, _order, line in records)
    if len(full) <= max_chars:
        return full

    selected: list[tuple[int, str]] = []
    used = 0
    for _score, order, line in sorted(records, key=lambda item: (-item[0], item[1])):
        size = len(line) + 2
        if used + size > max_chars:
            continue
        selected.append((order, line))
        used += size
    selected.sort(key=lambda item: item[0])
    return "\n\n".join(line for _order, line in selected)


class OpenAIExtractor:
    name = "openai-responses"

    def __init__(
        self,
        provider: OpenAIProvider,
        *,
        max_input_chars: int = 400_000,
        model: str | None = None,
    ) -> None:
        self.provider = provider
        self.max_input_chars = max_input_chars
        self.model = model

    @staticmethod
    def _response_schema(schema: dict[str, Any]) -> dict[str, Any]:
        data_schema = sanitize_schema_for_structured_output(
            schema,
            make_properties_nullable=True,
            force_all_required=True,
        )
        return {
            "type": "object",
            "properties": {
                "data": data_schema,
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "block_ids": {"type": "array", "items": {"type": "string"}},
                            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            "warnings": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["path", "block_ids", "confidence", "warnings"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["data", "evidence"],
            "additionalProperties": False,
        }

    def extract(self, document: ParsedDocument, schema: dict[str, Any]) -> ExtractionResult:
        context = _compact_document_context(document, schema, self.max_input_chars)
        system = (
            "You are GroundMark's document extraction engine. Extract only facts explicitly visible "
            "in the supplied grounded blocks. Never infer or calculate missing values. Use null when "
            "a field is absent. For every non-null leaf value, cite one or more exact block_id values. "
            "A block ID must be copied exactly from the document. Confidence reflects evidence clarity, "
            "not your general certainty."
        )
        user = (
            "Extract the document into the required schema. Preserve source spelling and numeric signs. "
            "Return evidence for every leaf path.\n\nGROUNDed DOCUMENT:\n" + context
        )
        llm_result = self.provider.structured(
            name="groundmark_extraction",
            schema=self._response_schema(schema),
            system=system,
            user=user,
            model=self.model,
        )
        data = llm_result.data.get("data") or {}
        evidence_rows = llm_result.data.get("evidence") or []
        evidence_map = {
            str(row.get("path")): row
            for row in evidence_rows
            if isinstance(row, dict) and row.get("path")
        }
        block_map = {block.id: block for page in document.pages for block in page.blocks}
        fields: list[ExtractedField] = []
        for path, _spec in HeuristicExtractor._flatten_properties(schema):
            row = evidence_map.get(path, {})
            evidence: list[FieldEvidence] = []
            for block_id in row.get("block_ids") or []:
                block = block_map.get(str(block_id))
                if block is None:
                    continue
                evidence.append(
                    FieldEvidence(
                        block_id=block.id,
                        page=block.grounding.page,
                        bbox=block.grounding.bbox,
                        quote=(block.markdown or block.text)[:500],
                    )
                )
            value = _value_at_path(data, path)
            confidence = float(row.get("confidence", 0.0))
            warnings = [str(item) for item in row.get("warnings") or []]
            if value is not None and not evidence:
                confidence = min(confidence or 0.5, 0.55)
                warnings.append("OpenAI returned a value without a valid grounded block ID")
            fields.append(
                ExtractedField(
                    path=path,
                    value=value,
                    confidence=max(0.0, min(confidence, 1.0)),
                    evidence=evidence,
                    warnings=warnings,
                )
            )

        errors = sorted(
            Draft202012Validator(schema).iter_errors(data), key=lambda error: list(error.path)
        )
        return ExtractionResult(
            data=data,
            fields=fields,
            schema_valid=not errors,
            validation_errors=[
                f"{'.'.join(map(str, error.path)) or '$'}: {error.message}" for error in errors
            ],
            provider=f"{self.name}:{llm_result.model}",
            metadata={
                "elapsed_ms": llm_result.elapsed_ms,
                "request_id": llm_result.request_id,
                "usage": llm_result.usage,
                "context_chars": len(context),
            },
        )


# Backwards-compatible name for callers importing the old class.
OpenAICompatibleExtractor = OpenAIExtractor


def build_schema_from_document(document: ParsedDocument) -> dict[str, Any]:
    """Build a conservative draft schema from visible key-value fields."""

    def key_for(label: str) -> str:
        key = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip()).strip("_").lower()
        return key[:80]

    def infer_type(value: str) -> str:
        cleaned = value.strip()
        lowered = cleaned.lower()
        if lowered in {"yes", "no", "true", "false", "checked", "unchecked"}:
            return "boolean"
        if re.fullmatch(r"[-+]?\d+", cleaned.replace(",", "")):
            return "integer"
        if re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)", cleaned.replace(",", "")):
            return "number"
        return "string"

    properties: dict[str, Any] = {}
    evidence: dict[str, list[str]] = {}
    pattern = re.compile(r"^\s*([^:=|]{2,80}?)\s*(?::|=|\|)\s*(\S.*)$")
    for page in document.pages:
        for block in page.blocks:
            for line in block.text.splitlines():
                match = pattern.match(line)
                if not match:
                    continue
                label, value = match.groups()
                key = key_for(label)
                if not key or key in properties:
                    continue
                properties[key] = {
                    "type": infer_type(value),
                    "title": label.strip(),
                    "x-aliases": [label.strip()],
                    "description": f"Extracted from a visible field labeled '{label.strip()}'.",
                }
                evidence[key] = [block.id]
    return {
        "schema": {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        },
        "evidence": evidence,
        "provider": "heuristic",
        "warnings": [] if properties else ["no key-value fields were detected"],
    }


def build_schema_from_document_openai(
    document: ParsedDocument,
    provider: OpenAIProvider,
    *,
    max_input_chars: int = 400_000,
    model: str | None = None,
) -> dict[str, Any]:
    response_schema = {
        "type": "object",
        "properties": {
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": ["string", "number", "integer", "boolean", "array"],
                        },
                        "aliases": {"type": "array", "items": {"type": "string"}},
                        "required": {"type": "boolean"},
                        "evidence_block_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "name",
                        "title",
                        "description",
                        "type",
                        "aliases",
                        "required",
                        "evidence_block_ids",
                    ],
                    "additionalProperties": False,
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["fields", "warnings"],
        "additionalProperties": False,
    }
    context = _compact_document_context(document, None, max_input_chars)
    result = provider.structured(
        name="groundmark_schema_proposal",
        schema=response_schema,
        system=(
            "Design a reusable extraction schema from the visible document fields. Include only fields "
            "supported by explicit document evidence. Use snake_case names. Do not include calculated, "
            "speculative, or decorative content."
        ),
        user="Propose extraction fields for this grounded document:\n\n" + context,
        model=model,
    )
    block_ids = {block.id for page in document.pages for block in page.blocks}
    properties: dict[str, Any] = {}
    evidence: dict[str, list[str]] = {}
    required: list[str] = []
    for field in result.data.get("fields") or []:
        name = re.sub(r"[^a-zA-Z0-9_]+", "_", str(field.get("name", ""))).strip("_").lower()[:80]
        if not name or name in properties:
            continue
        field_type = str(field.get("type", "string"))
        spec: dict[str, Any] = {
            "type": field_type,
            "title": str(field.get("title") or name.replace("_", " ").title()),
            "description": str(field.get("description") or ""),
            "x-aliases": [str(item) for item in field.get("aliases") or [] if str(item).strip()],
        }
        if field_type == "array":
            spec["items"] = {"type": "string"}
        properties[name] = spec
        valid_evidence = [
            str(item) for item in field.get("evidence_block_ids") or [] if str(item) in block_ids
        ]
        evidence[name] = valid_evidence
        if field.get("required") and valid_evidence:
            required.append(name)
    schema: dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": properties,
        "additionalProperties": False,
    }
    if required:
        schema["required"] = required
    return {
        "schema": schema,
        "evidence": evidence,
        "provider": f"openai-responses:{result.model}",
        "warnings": [str(item) for item in result.data.get("warnings") or []],
        "metadata": {
            "elapsed_ms": result.elapsed_ms,
            "request_id": result.request_id,
            "usage": result.usage,
            "context_chars": len(context),
        },
    }
