"""Freeze a matrix, train every run, then evaluate the fixed checkpoints."""

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from mosei.config import load_config
from mosei.data import load_data
from mosei.evaluate import evaluate_run
from mosei.io import write_json
from mosei.report import summarize_runs
from mosei.train import train_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/prepared"))
    parser.add_argument(
        "--configs",
        type=Path,
        nargs="+",
        default=[Path(f"configs/{n}.toml") for n in ("text", "concat", "transformer")],
    )
    parser.add_argument("--include-ablations", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1111, 1112, 1113])
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New directory for this complete experiment matrix",
    )
    parser.add_argument(
        "--epochs", type=int, help="Override all epoch limits equally, e.g. 1 for a smoke check"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Leave held-out evaluation for a later, frozen decision",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args()
    try:
        paths = args.configs + (
            [Path(f"configs/{n}.toml") for n in ("no_modality", "no_positions", "mean_pooling")]
            if args.include_ablations
            else []
        )
        configs = []
        for path in paths:
            config = load_config(path)
            changes = {"device": args.device}
            if args.epochs is not None:
                changes["epochs"] = args.epochs
            configs.append(replace(config, training=replace(config.training, **changes)))
        if len(set(args.seeds)) != len(args.seeds) or len({c.name for c in configs}) != len(
            configs
        ):
            raise ValueError("Experiment names and seeds must be unique")
        if any(
            not c.name
            or any(
                ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for ch in c.name
            )
            for c in configs
        ):
            raise ValueError("Experiment names must be safe filename components")
        data = load_data(args.data)
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = {
            "status": "training",
            "data_fingerprint": data.fingerprint,
            "configs": [asdict(c) for c in configs],
            "seeds": args.seeds,
            "evaluation": "test split; argmax; clean + 12 matched corruption conditions; noise seed 2026",
            "selection": "highest validation macro F1, then lowest validation cross-entropy",
        }
        write_json(args.output / "manifest.json", manifest)
        runs = []
        for config in configs:
            for seed in args.seeds:
                run = args.output / f"{config.name}-seed{seed}"
                train_run(data, replace(config, training=replace(config.training, seed=seed)), run)
                runs.append(run)
        manifest["status"] = "trained"
        write_json(args.output / "manifest.json", manifest)
        if not args.train_only:
            for run in runs:
                evaluate_run(data, run, robustness=True, device=args.device)
            summarize_runs(runs, args.output / "report", args.bootstrap_samples)
            manifest["status"] = "evaluated"
            write_json(args.output / "manifest.json", manifest)
        print(f"Experiment matrix {manifest['status']}: {args.output}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
