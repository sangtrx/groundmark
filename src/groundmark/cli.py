from __future__ import annotations

import json
from pathlib import Path

import typer

from groundmark.evaluation import evaluate_manifest
from groundmark.logging_utils import configure_logging
from groundmark.models import ParseOptions
from groundmark.service import GroundMark
from groundmark.settings import get_settings

app = typer.Typer(help="GroundMark: MarkItDown + grounded layout + OpenAI document intelligence")
engine = GroundMark()


@app.command()
def parse(
    input_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", "-o"),
    json_output: Path | None = typer.Option(None, "--json"),
    mode: str = typer.Option("balanced", help="fast, balanced, or accurate"),
    parser: str = typer.Option("auto", help="auto, markitdown, native, or docling"),
    include_headers_footers: bool = typer.Option(False),
    no_grounding_comments: bool = typer.Option(False),
    no_cache: bool = typer.Option(False),
    repair: bool = typer.Option(
        False, "--repair", help="Repair uncertain PDF regions with OpenAI vision"
    ),
    max_repair_regions: int | None = typer.Option(None, min=1, max=500),
) -> None:
    result = engine.parse(
        input_file,
        ParseOptions(
            mode=mode,
            parser=parser,
            include_headers_footers=include_headers_footers,
            include_grounding_comments=not no_grounding_comments,
            use_cache=not no_cache,
            run_repair=repair,
            max_repair_regions=max_repair_regions,
        ),
    )
    if output:
        output.write_text(result.markdown, encoding="utf-8")
        typer.echo(f"Markdown: {output}")
    else:
        typer.echo(result.markdown)
    if json_output:
        json_output.write_text(result.model_dump_json(indent=2), encoding="utf-8")
        typer.echo(f"Grounded JSON: {json_output}")
    typer.echo(
        f"confidence={result.confidence:.3f} parser={result.metadata.parser} "
        f"cache_hit={result.metadata.cache_hit}",
        err=True,
    )


@app.command()
def extract(
    input_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    schema_file: Path = typer.Option(..., "--schema", exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", "-o"),
    provider: str = typer.Option("auto", help="auto, openai, or heuristic"),
    model: str | None = typer.Option(None, help="Optional OpenAI model override"),
) -> None:
    parsed = engine.parse(input_file)
    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    result = engine.extract(parsed, schema, provider=provider, model=model)  # type: ignore[arg-type]
    payload = result.model_dump_json(indent=2)
    if output:
        output.write_text(payload, encoding="utf-8")
        typer.echo(f"Extraction: {output}")
    else:
        typer.echo(payload)


@app.command("build-schema")
def build_schema_command(
    input_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", "-o"),
    provider: str = typer.Option("auto", help="auto, openai, or heuristic"),
    model: str | None = typer.Option(None),
) -> None:
    parsed = engine.parse(input_file)
    payload = json.dumps(engine.build_schema(parsed, provider=provider, model=model), indent=2)  # type: ignore[arg-type]
    if output:
        output.write_text(payload, encoding="utf-8")
    typer.echo(payload)


@app.command()
def classify(
    input_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    provider: str = typer.Option("auto", help="auto, openai, or heuristic"),
    labels: str | None = typer.Option(None, help="Comma-separated allowed labels"),
    model: str | None = typer.Option(None),
) -> None:
    parsed = engine.parse(input_file)
    allowed = [item.strip() for item in labels.split(",") if item.strip()] if labels else None
    result = engine.classify(
        parsed,
        provider=provider,  # type: ignore[arg-type]
        labels=allowed,
        model=model,
    )
    typer.echo(json.dumps([item.model_dump(mode="json") for item in result], indent=2))


@app.command()
def section(
    input_file: Path = typer.Argument(..., exists=True, dir_okay=False),
    provider: str = typer.Option("auto", help="auto, openai, or heuristic"),
    model: str | None = typer.Option(None),
) -> None:
    parsed = engine.parse(input_file)
    result = engine.section(parsed, provider=provider, model=model)  # type: ignore[arg-type]
    typer.echo(json.dumps([item.model_dump(mode="json") for item in result], indent=2))


@app.command("evaluate")
def evaluate_command(
    manifest: Path = typer.Argument(..., exists=True, dir_okay=False),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    result = evaluate_manifest(engine, manifest)
    payload = json.dumps(result, indent=2)
    if output:
        output.write_text(payload, encoding="utf-8")
    typer.echo(payload)


@app.command()
def capabilities() -> None:
    settings = get_settings()
    typer.echo(
        json.dumps(
            {
                "markitdown": engine.normalizer.available(),
                "native_pdf_layout": True,
                "docling": engine.docling_layout.available(),
                "openai_configured": engine.openai_configured,
                "openai_api": "responses",
                "openai_model": settings.openai_model,
                "openai_vision_model": settings.resolved_openai_vision_model,
                "version": "0.4.0",
            },
            indent=2,
        )
    )


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000, reload: bool = False) -> None:
    import uvicorn

    configure_logging(get_settings().log_json)
    uvicorn.run("groundmark.api:app", host=host, port=port, reload=reload)


@app.command()
def worker() -> None:
    from groundmark.worker import main

    main()


if __name__ == "__main__":
    app()
