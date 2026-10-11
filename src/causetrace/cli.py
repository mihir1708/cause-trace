"""The ``causetrace`` command line (Typer).

causetrace ask <ev_id>                 # agent answer with citations, grounding and trace
causetrace eval retrieval [ARGS...]    # = scripts/eval_retrieval.py [ARGS...]
causetrace eval causal [ARGS...]       # = scripts/eval_causal.py [ARGS...] (paid)
causetrace serve                       # the HTTP API (uvicorn)
"""

import json
from typing import Annotated

import typer

app = typer.Typer(
    help="CauseTrace: NTSB causal findings from incident facts.", no_args_is_help=True
)
eval_app = typer.Typer(help="Run an evaluation (wraps the scripts in scripts/).")
app.add_typer(eval_app, name="eval")

PASS_THROUGH = {"allow_extra_args": True, "ignore_unknown_options": True}


@app.command()
def ask(
    ev_id: str,
    provider: Annotated[
        str, typer.Option(help="auto | anthropic | ollama (auto: Claude if a key is set)")
    ] = "auto",
    database_url: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Run the agent (with verify) for one stored event and print the result JSON."""
    from causetrace.agent import Agent
    from causetrace.agent.__main__ import make_provider
    from causetrace.db import connect
    from causetrace.guardrails import ground_findings
    from causetrace.llm import ReliableProvider
    from causetrace.retrieval import Retriever

    with connect(database_url) as conn:
        agent = Agent(conn, Retriever(conn), ReliableProvider(make_provider(provider)))
        try:
            result = agent.run(ev_id)
        except LookupError as err:
            typer.echo(str(err), err=True)
            raise typer.Exit(2) from err
    kept, dropped, score = ground_findings(
        result["findings"], result["passages"], before_date=result["before_date"], ev_id=ev_id
    )
    result["findings"] = kept
    result["guardrail"] = {"kept": len(kept), "dropped": dropped, "grounding_score": score}
    typer.echo(json.dumps(result, indent=2, default=str))


@eval_app.command("retrieval", context_settings=PASS_THROUGH)
def eval_retrieval(ctx: typer.Context) -> None:
    """Retrieval evaluation; extra arguments go to scripts/eval_retrieval.py."""
    from causetrace.evaluation.retrieval_eval import main

    raise typer.Exit(main(list(ctx.args)) or 0)


@eval_app.command("causal", context_settings=PASS_THROUGH)
def eval_causal(ctx: typer.Context) -> None:
    """Causal comparison (paid LLM calls); extra arguments go to scripts/eval_causal.py."""
    from causetrace.evaluation.causal_eval import main

    raise typer.Exit(main(list(ctx.args)) or 0)


@app.command()
def serve(
    host: Annotated[str, typer.Option()] = "127.0.0.1",
    port: Annotated[int, typer.Option()] = 8000,
    workers: Annotated[int, typer.Option(help="uvicorn worker processes")] = 1,
) -> None:
    """Serve the HTTP API."""
    import uvicorn

    uvicorn.run(
        "causetrace.api.app:create_app", factory=True, host=host, port=port, workers=workers
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
