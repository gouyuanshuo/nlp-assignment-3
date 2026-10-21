"""Freeze a matrix, train every run, then evaluate the fixed checkpoints."""

import argparse
import math
import time
from dataclasses import asdict, replace
from pathlib import Path

from mosei.config import config_from_dict, load_config
from mosei.data import load_data
from mosei.evaluate import evaluate_run
from mosei.io import read_json, write_json
from mosei.report import summarize_runs
from mosei.train import train_run

HANLIN_ABLATIONS = {
    "almt": {
        "without-modality-embedding": "modality_embeddings",
        "without-ahl": "almt_use_ahl",
        "without-cross-modal-fusion": "almt_use_cross_modal_fusion",
        "without-graph-fusion": "almt_use_graph_fusion",
    },
    "graph_mfn": {
        "without-dynamic-graph": "graph_use_dynamic_graph",
        "without-view-memory": "graph_use_view_memory",
        "without-fusion-memory": "graph_use_fusion_memory",
        "without-temporal-attention": "graph_use_temporal_attention",
    },
}


def matrix_configs(
    paths, *, device="auto", epochs=None, train_fractions=None, include_hanlin_ablations=False
):
    """Expand models without creating a separate training or sampling protocol."""
    if train_fractions is not None and (
        not train_fractions
        or any(not math.isfinite(f) or not 0 < f <= 1 for f in train_fractions)
        or len(set(train_fractions)) != len(train_fractions)
    ):
        raise ValueError("Training fractions must be unique finite values in (0, 1]")
    configs = []
    for path in paths:
        config = load_config(path)
        changes = {"device": device}
        if epochs is not None:
            changes["epochs"] = epochs
        config = replace(config, training=replace(config.training, **changes))
        variants = [config]
        if include_hanlin_ablations:
            for suffix, field in HANLIN_ABLATIONS.get(config.model.name, {}).items():
                if not getattr(config.model, field):
                    raise ValueError("Component ablations require a full model configuration")
                variants.append(replace(
                    config, name=f"{config.name}-{suffix}",
                    model=replace(config.model, **{field: False}),
                ))
        for variant in variants:
            if train_fractions is None:
                configs.append(variant)
            else:
                for fraction in train_fractions:
                    percent = format(fraction * 100, ".12g").replace(".", "p")
                    configs.append(replace(
                        variant, name=f"{variant.name}-frac{percent}",
                        training=replace(variant.training, train_fraction=fraction),
                    ))
    return configs


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
    parser.add_argument(
        "--include-hanlin-ablations", action="store_true",
        help="Add four component ablations for each selected ALMT/Graph-MFN full model",
    )
    parser.add_argument(
        "--train-fractions", type=float, nargs="+",
        help="Whole-source-video training fractions, e.g. 0.1 0.5 1; validation/test stay complete",
    )
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
    parser.add_argument(
        "--resume", action="store_true", help="Resume the frozen matrix and its epoch checkpoints"
    )
    args = parser.parse_args()
    try:
        paths = args.configs + (
            [Path(f"configs/{n}.toml") for n in ("no_modality", "no_positions", "mean_pooling")]
            if args.include_ablations
            else []
        )
        configs = matrix_configs(
            paths, device=args.device, epochs=args.epochs,
            train_fractions=args.train_fractions,
            include_hanlin_ablations=args.include_hanlin_ablations,
        )
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
        args.output.mkdir(parents=True, exist_ok=args.resume)
        manifest = {
            "status": "training",
            "data_fingerprint": data.fingerprint,
            "configs": [asdict(c) for c in configs],
            "seeds": args.seeds,
            "evaluation": "test split; argmax; clean + 12 matched corruption conditions; noise seed 2026",
            "selection": "highest validation macro F1, then lowest validation cross-entropy",
            "bootstrap_samples": args.bootstrap_samples,
        }
        previous = args.output / "manifest.json"
        if previous.exists():
            recorded = read_json(previous)
            for key in (
                "data_fingerprint",
                "configs",
                "seeds",
                "evaluation",
                "selection",
                "bootstrap_samples",
            ):
                before, after = recorded[key], manifest[key]
                if key == "configs":
                    # Added model options must not invalidate old matrices when
                    # their effective defaults and architecture are unchanged.
                    before = [asdict(config_from_dict(c)) for c in before]
                    after = [asdict(config_from_dict(c)) for c in after]
                if before != after:
                    raise ValueError(f"Resume matrix changed its frozen {key}")
        write_json(args.output / "manifest.json", manifest)
        runs = []
        for config in configs:
            for seed in args.seeds:
                run = args.output / f"{config.name}-seed{seed}"
                train_run(
                    data,
                    replace(config, training=replace(config.training, seed=seed)),
                    run,
                    resume=args.resume,
                )
                runs.append(run)
        manifest["status"] = "trained"
        write_json(args.output / "manifest.json", manifest)
        if not args.train_only:
            for run in runs:
                evaluation = run / "evaluation/test"
                if args.resume and (evaluation / "metrics.json").is_file():
                    continue
                if evaluation.exists():
                    # Preserve interrupted output while recomputing its fixed evaluation.
                    evaluation.rename(evaluation.with_name(f"test-interrupted-{time.time_ns()}"))
                evaluate_run(data, run, robustness=True, device=args.device)
            report = args.output / "report"
            if not (report / "complete.json").is_file():
                if report.exists():
                    report.rename(report.with_name(f"report-interrupted-{time.time_ns()}"))
                summarize_runs(runs, report, args.bootstrap_samples)
                write_json(report / "complete.json", {"bootstrap_samples": args.bootstrap_samples})
            manifest["status"] = "evaluated"
            write_json(args.output / "manifest.json", manifest)
        print(f"Experiment matrix {manifest['status']}: {args.output}")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
