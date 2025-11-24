#!/usr/bin/env python3
import click
import json
import logging
from subprocess import run
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

script_dir = Path(__file__).parent


def write_csv_dataset(
    name: str,
    train_data: tuple[list[str], list[list[str]]],
    val_data: tuple[list[str], list[list[str]]],
    meta: dict,
):
    train_dir = script_dir / "dataset" / name / "train"
    val_dir = script_dir / "dataset" / name / "val"
    dataset_dir = script_dir / "dataset" / name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_csv = train_dir / "train.csv"
    val_csv = val_dir / "val.csv"
    meta_json = dataset_dir / "meta.json"

    train_header, train_rows = train_data
    val_header, val_rows = val_data

    train_content = (
        ",".join(train_header)
        + "\n"
        + "\n".join([",".join(row) for row in train_rows])
        + "\n"
    )
    val_content = (
        ",".join(val_header)
        + "\n"
        + "\n".join([",".join(row) for row in val_rows])
        + "\n"
    )

    train_csv.write_text(train_content)
    val_csv.write_text(val_content)
    meta_json.write_text(json.dumps(meta, indent=2))

    click.echo(f"created {train_csv}")
    click.echo(f"created {val_csv}")
    click.echo(f"created {meta_json}")


@click.group()
def cli():
    pass


@cli.command()
def sandbox():
    """build sandbox docker image"""
    run(
        ["docker", "build", "-t", "envy", "-f", "sandbox.Dockerfile", "."],
        cwd=script_dir,
        check=True,
    )


@cli.group()
def dataset():
    """dataset commands"""
    pass


@dataset.command()
def xor():
    """generate xor dataset"""
    train_data = (
        ["x1", "x2", "y"],
        [["0", "0", "0"], ["0", "1", "1"], ["1", "0", "1"]],
    )
    val_data = (["x1", "x2", "y"], [["1", "1", "0"]])
    meta = {
        "name": "xor",
        "task": "binary classification",
        "expected_score": 1.0,
        "score_tolerance": 0.1,
        "description": "trivial xor problem — should achieve 100% accuracy",
    }
    write_csv_dataset("xor", train_data, val_data, meta)


@dataset.command()
def mnist1d():
    """generate mnist1d dataset"""
    from mnist1d.data import make_dataset, get_dataset_args

    click.echo("generating mnist1d dataset...")
    defaults = get_dataset_args()
    data = make_dataset(defaults)

    x_train, y_train = data["x"], data["y"]
    x_test, y_test = data["x_test"], data["y_test"]

    feature_cols = [f"x{i}" for i in range(x_train.shape[1])]

    train_rows = [
        [str(val) for val in x_train[i]] + [str(int(y_train[i]))]
        for i in range(len(x_train))
    ]
    val_rows = [
        [str(val) for val in x_test[i]] + [str(int(y_test[i]))]
        for i in range(len(x_test))
    ]

    train_data = (feature_cols + ["y"], train_rows)
    val_data = (feature_cols + ["y"], val_rows)
    meta = {
        "name": "mnist1d",
        "task": "10-class classification",
        "expected_score": 0.8,
        "score_tolerance": 0.1,
        "description": "mnist1d benchmark — logistic: 32%, mlp: 68%, cnn: 94%",
    }

    write_csv_dataset("mnist1d", train_data, val_data, meta)

    click.echo(f"({len(x_train)} samples, {x_train.shape[1]} features in train)")
    click.echo(f"({len(x_test)} samples, {x_test.shape[1]} features in val)")


@dataset.command()
@click.option(
    "--dataset",
    default=None,
    type=str,
    help="existing dataset to recorrupt (skips generation)",
)
@click.option(
    "--used", default=0.7, type=float, help="fraction of features used in dynamics"
)
@click.option("--corrupt", default=0.2, type=float, help="fraction of cells to corrupt")
@click.option("--num-cols", default=10, type=int, help="number of feature columns")
@click.option("--num-rows", default=1000, type=int, help="number of samples")
@click.option(
    "--complexity", default=250, type=int, help="target AST node count (±10 tolerance)"
)
@click.option(
    "--regression",
    is_flag=True,
    default=False,
    help="regression task (default: classification)",
)
@click.option(
    "--num-classes",
    default=5,
    type=int,
    help="number of classes (classification only)",
)
@click.option("--seed", default=None, type=int, help="random seed for determinism")
@click.option("--debug", is_flag=True, default=False, help="enable debug logging")
def generate(
    dataset: str | None,
    used: float,
    corrupt: float,
    num_cols: int,
    num_rows: int,
    complexity: int,
    regression: bool,
    num_classes: int,
    seed: int,
    debug: bool,
):
    """generate synthetic dataset with corruption"""
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    sys.path.insert(0, str(script_dir))

    classification = not regression

    if dataset is not None:
        from generate import recorrupt_dataset

        asyncio.run(
            recorrupt_dataset(
                dataset_name=dataset,
                corrupt=corrupt,
                seed=seed,
            )
        )
    else:
        from generate import generate_dataset

        asyncio.run(
            generate_dataset(
                used=used,
                corrupt=corrupt,
                num_cols=num_cols,
                num_rows=num_rows,
                complexity=complexity,
                classification=classification,
                num_classes=num_classes,
                seed=seed,
            )
        )


@cli.command()
@click.option(
    "--model", default="claude-haiku-4-5", type=str, help="anthropic model to use"
)
@click.option(
    "--dataset",
    default=None,
    type=str,
    help="dataset name (e.g. synth-363181). if not provided, uses most recent dataset.",
)
@click.option("--num-runs", default=3, type=int, help="number of test runs")
@click.option("--max-tokens", default=3000, type=int, help="max tokens per request")
@click.option("--max-steps", default=30, type=int, help="max steps per agent loop")
@click.option(
    "--max-eval-uses", default=10, type=int, help="max eval_model calls per agent loop"
)
@click.option(
    "--sequential",
    is_flag=True,
    default=False,
    help="run tests sequentially instead of concurrently",
)
@click.option(
    "--slow",
    is_flag=True,
    default=False,
    help="enable optuna hyperparameter tuning (default: rapid iteration only)",
)
@click.option(
    "--expert",
    is_flag=True,
    default=False,
    help="expert mode: train on golden.csv (clean data) instead of train.csv",
)
@click.option("--debug", is_flag=True, default=False, help="enable debug logging")
def eval(
    model: str,
    num_runs: int,
    max_tokens: int,
    max_steps: int,
    max_eval_uses: int,
    sequential: bool,
    slow: bool,
    expert: bool,
    dataset: str | None,
    debug: bool,
):
    """run evaluation"""
    import asyncio
    import sys

    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)

    sys.path.insert(0, str(script_dir))

    from main import run_evaluation

    if dataset is None:
        dataset_dir = script_dir / "dataset"
        if not dataset_dir.exists():
            click.echo("error: dataset directory not found", err=True)
            sys.exit(1)

        datasets = [
            d
            for d in dataset_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]
        if not datasets:
            click.echo("error: no datasets found in dataset/", err=True)
            sys.exit(1)

        most_recent = max(datasets, key=lambda d: d.stat().st_mtime)
        dataset = most_recent.name
        click.echo(f"auto-selected most recent dataset: {dataset}")

    result = asyncio.run(
        run_evaluation(
            model=model,
            dataset=dataset,
            num_runs=num_runs,
            max_steps=max_steps,
            max_tokens=max_tokens,
            max_eval_uses=max_eval_uses,
            sequential=sequential,
            slow=slow,
            expert=expert,
        )
    )

    if result.is_err():
        click.echo(f"error: {result.unwrap_err()}", err=True)
        sys.exit(1)


def main():
    cli()


if __name__ == "__main__":
    main()
