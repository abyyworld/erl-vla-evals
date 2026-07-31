"""Command-line interface."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from . import __version__
from .adapters import build_policy
from .benchmark import BenchmarkResult, BenchmarkSpec, run_benchmark
from .compare import compare as run_compare
from .registry import Registry, RegistryError

app = typer.Typer(
    add_completion=False,
    help="Checkpoint registry and evaluation harness for robot policies.",
    no_args_is_help=True,
)
registry_app = typer.Typer(help="Manage the checkpoint registry.", no_args_is_help=True)
app.add_typer(registry_app, name="registry")

console = Console()

RegistryOpt = typer.Option("registry", "--registry", help="Registry root directory.")
BenchmarkOpt = typer.Option(
    "conf/benchmarks/manipulation_v1.yaml", "--benchmark", "-b", help="Benchmark spec."
)
ResultsOpt = typer.Option("results", "--results", help="Where run results are stored.")


@app.command()
def version() -> None:
    console.print(f"policy-eval-harness {__version__}")


# -- registry ---------------------------------------------------------------


@registry_app.command("add")
def registry_add(
    checkpoint: str = typer.Argument(..., help="Path to the checkpoint file."),
    registry: str = RegistryOpt,
    name: str = typer.Option(None, help="Human-readable name."),
    lineage: str = typer.Option(None, help="Path to lineage.json (auto-detected otherwise)."),
    stage: str = typer.Option("staging", help="staging | production | archived"),
    notes: str = typer.Option("", help="Free-text note."),
    require_lineage: bool = typer.Option(
        True,
        "--require-lineage/--no-require-lineage",
        help="Refuse checkpoints with no provenance. Leave on.",
    ),
) -> None:
    """Register a checkpoint, content-addressed by its bytes."""
    reg = Registry(registry)
    try:
        record, created = reg.register(
            Path(checkpoint),
            name=name,
            lineage_path=Path(lineage) if lineage else None,
            stage=stage,  # type: ignore[arg-type]
            notes=notes,
            require_lineage=require_lineage,
        )
    except RegistryError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    if created:
        console.print(f"[green]registered[/] {record.checkpoint_id}  ({record.name})")
    else:
        console.print(f"[yellow]already registered[/] {record.checkpoint_id} — identical bytes")

    gaps = record.provenance_gaps()
    if gaps:
        console.print(f"[yellow]provenance gaps:[/] {', '.join(gaps)}")
    else:
        console.print(f"dataset {record.dataset_hash}  commit {record.short_commit}")


@registry_app.command("list")
def registry_list(
    registry: str = RegistryOpt,
    stage: str = typer.Option(None, help="Filter by stage."),
) -> None:
    """List registered checkpoints, newest first."""
    records = Registry(registry).list(stage=stage)  # type: ignore[arg-type]
    if not records:
        console.print("registry is empty")
        return

    table = Table(title=f"Checkpoints ({len(records)})")
    for column in ("id", "name", "stage", "dataset", "commit", "registered"):
        table.add_column(column)
    for r in records:
        stage_colour = {"production": "green", "staging": "yellow"}.get(r.stage, "dim")
        table.add_row(
            r.checkpoint_id[:12],
            r.name,
            f"[{stage_colour}]{r.stage}[/]",
            r.dataset_hash or "—",
            r.short_commit,
            r.registered_at[:19],
        )
    console.print(table)


@registry_app.command("promote")
def registry_promote(
    checkpoint_id: str = typer.Argument(...),
    registry: str = RegistryOpt,
    stage: str = typer.Option("production", help="staging | production | archived"),
) -> None:
    """Move a checkpoint to a stage. Promoting archives the incumbent."""
    reg = Registry(registry)
    try:
        record = reg.promote(checkpoint_id, stage)  # type: ignore[arg-type]
    except RegistryError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    gaps = record.provenance_gaps()
    if stage == "production" and gaps:
        # Not blocked — sometimes you must ship a checkpoint whose provenance is
        # imperfect — but never silent.
        console.print(f"[yellow]warning:[/] promoting to production with {', '.join(gaps)}")
    console.print(f"[green]{record.checkpoint_id[:12]}[/] -> {stage}")


@registry_app.command("show")
def registry_show(
    checkpoint_id: str = typer.Argument(...),
    registry: str = RegistryOpt,
) -> None:
    """Show everything known about one checkpoint."""
    try:
        record = Registry(registry).get(checkpoint_id)
    except RegistryError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc
    console.print_json(record.model_dump_json(indent=2))


# -- evaluation -------------------------------------------------------------


@app.command()
def run(
    policy: str = typer.Argument(
        ..., help="zero | random | scripted | scripted+noise:<sigma> | <checkpoint path> | <id>"
    ),
    benchmark: str = BenchmarkOpt,
    registry: str = RegistryOpt,
    results: str = ResultsOpt,
    tag: str = typer.Option(None, help="Override the stored result name."),
) -> None:
    """Run a policy against the benchmark suite."""
    spec = BenchmarkSpec.load(Path(benchmark))
    resolved, metadata = _resolve_policy(policy, registry)

    console.print(
        f"running [bold]{spec.name}[/] — {len(spec.tasks)} task(s) × "
        f"{spec.episodes_per_task} episodes"
    )

    def progress(task_id, episodes):
        rate = sum(e.success for e in episodes) / len(episodes)
        console.print(f"  {task_id:<24} {rate:>6.1%}")

    result = run_benchmark(spec, resolved, metadata=metadata, progress=progress)
    path = result.save(
        Path(results) / f"{spec.name}__{tag or result.policy}.json".replace("/", "_")
    )

    console.print(f"\n[bold]overall {result.overall_success_rate:.1%}[/]")
    console.print(f"saved {path}")


@app.command()
def compare(
    baseline: str = typer.Argument(..., help="Baseline result JSON."),
    candidate: str = typer.Argument(..., help="Candidate result JSON."),
    tolerance: float = typer.Option(0.02, help="Allowed drop in overall success rate."),
    task_tolerance: float = typer.Option(0.10, help="Allowed drop on any single task."),
    out: str = typer.Option("results/comparison.md", help="Markdown report path."),
    fail_on_regression: bool = typer.Option(
        True, help="Exit non-zero on FAIL or INCONCLUSIVE, for CI."
    ),
) -> None:
    """Paired comparison of two runs, with a regression gate."""
    from . import report as report_mod

    base = BenchmarkResult.load(Path(baseline))
    cand = BenchmarkResult.load(Path(candidate))

    try:
        result = run_compare(base, cand, tolerance=tolerance, task_tolerance=task_tolerance)
    except ValueError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=1) from exc

    colour = {"PASS": "green", "FAIL": "red", "INCONCLUSIVE": "yellow"}[result.verdict]
    console.print(f"\n[{colour}][bold]{result.verdict}[/][/]  {result.headline()}")
    console.print(
        f"discordant pairs: {result.wins} win / {result.losses} loss, "
        f"McNemar p = {result.p_value:.4f}"
    )
    for reason in result.reasons:
        console.print(f"  · {reason}")

    if result.tasks:
        table = Table(title="Per task")
        for column in ("task", "baseline", "candidate", "difference (95% CI)", "p"):
            table.add_column(column)
        for task in result.tasks:
            label = f"[red]{task.task_id}[/]" if task.regressed else task.task_id
            table.add_row(
                label,
                f"{task.baseline_rate:.1%}",
                f"{task.candidate_rate:.1%}",
                f"{task.difference:+.1%} [{task.ci_low:+.1%}, {task.ci_high:+.1%}]",
                f"{task.p_value:.3f}",
            )
        console.print(table)

    path = report_mod.write(Path(out), report_mod.render_comparison(result))
    Path(out).with_suffix(".json").write_text(json.dumps(result.to_dict(), indent=2))
    console.print(f"\nreport: {path}")

    if fail_on_regression and result.verdict != "PASS":
        raise typer.Exit(code=1)


@app.command()
def gate(
    candidate: str = typer.Argument(..., help="Candidate checkpoint path or registry id."),
    benchmark: str = BenchmarkOpt,
    registry: str = RegistryOpt,
    results: str = ResultsOpt,
    tolerance: float = typer.Option(0.02),
    task_tolerance: float = typer.Option(0.10),
) -> None:
    """Evaluate a candidate against the production checkpoint and gate it.

    The single command CI runs: benchmark both arms on identical seeds, compare,
    and exit non-zero unless the candidate is demonstrably not a regression.
    """
    from . import report as report_mod

    reg = Registry(registry)
    incumbent = reg.production()
    if incumbent is None:
        console.print(
            "[yellow]no production checkpoint[/] — nothing to compare against. "
            "Register one and `registry promote` it first."
        )
        raise typer.Exit(code=1)

    spec = BenchmarkSpec.load(Path(benchmark))
    out_dir = Path(results)

    console.print(f"baseline:  {incumbent.checkpoint_id[:12]} ({incumbent.name})")
    base_policy, base_meta = _resolve_policy(str(reg.blob_path(incumbent)), registry)
    base_result = run_benchmark(spec, base_policy, metadata=base_meta)
    base_result.save(out_dir / f"{spec.name}__baseline.json")

    console.print(f"candidate: {candidate}")
    cand_policy, cand_meta = _resolve_policy(candidate, registry)
    cand_result = run_benchmark(spec, cand_policy, metadata=cand_meta)
    cand_result.save(out_dir / f"{spec.name}__candidate.json")

    result = run_compare(
        base_result, cand_result, tolerance=tolerance, task_tolerance=task_tolerance
    )
    colour = {"PASS": "green", "FAIL": "red", "INCONCLUSIVE": "yellow"}[result.verdict]
    console.print(f"\n[{colour}][bold]{result.verdict}[/][/]  {result.headline()}")
    for reason in result.reasons:
        console.print(f"  · {reason}")

    report_mod.write(out_dir / "gate.md", report_mod.render_comparison(result))
    (out_dir / "gate.json").write_text(json.dumps(result.to_dict(), indent=2))

    if result.verdict != "PASS":
        raise typer.Exit(code=1)


@app.command()
def report(
    result: str = typer.Argument(..., help="Benchmark result JSON."),
    out: str = typer.Option(None, help="Markdown output path."),
) -> None:
    """Render a Markdown report for a single benchmark run."""
    from . import report as report_mod

    loaded = BenchmarkResult.load(Path(result))
    markdown = report_mod.render_run(loaded)
    if out:
        console.print(f"wrote {report_mod.write(Path(out), markdown)}")
    else:
        console.print(markdown)


def _resolve_policy(spec: str, registry: str) -> tuple[object, dict]:
    """Accept a built-in name, a checkpoint path, or a registry id."""
    builtin = spec in {"zero", "random", "scripted"} or spec.startswith("scripted+noise")
    if builtin or Path(spec).exists():
        return build_policy(spec), {"policy_spec": spec}

    try:
        record = Registry(registry).resolve(spec)
    except RegistryError as exc:
        raise typer.BadParameter(str(exc)) from exc

    path = Registry(registry).blob_path(record)
    return build_policy(str(path), name=record.name), {
        "checkpoint_id": record.checkpoint_id,
        "dataset_hash": record.dataset_hash,
        "git_commit": record.git_commit,
    }


if __name__ == "__main__":
    app()
