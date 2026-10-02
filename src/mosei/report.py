"""Seed summaries, paired source-video bootstrap, and exportable figures."""

import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from .evaluate import read_predictions
from .io import fingerprint, read_json, write_json


def error_examples(left: list[dict], right: list[dict], limit: int = 10) -> dict:
    right_by_id = {r["id"]: r for r in right}
    if {r["id"] for r in left} != set(right_by_id):
        raise ValueError("Error examples require matching clip IDs")
    corrected, regressed = [], []
    for a in left:
        b = right_by_id[a["id"]]
        if a["label"] != b["label"]:
            raise ValueError("Error examples require matching labels")
        row = {
            "id": a["id"],
            "video_id": a["video_id"],
            "label": a["label"],
            "left_prediction": a["prediction"],
            "right_prediction": b["prediction"],
        }
        if a["prediction"] == a["label"] and b["prediction"] != b["label"]:
            corrected.append(row)
        if b["prediction"] == b["label"] and a["prediction"] != a["label"]:
            regressed.append(row)
    return {
        "left_correct_right_wrong_count": len(corrected),
        "right_correct_left_wrong_count": len(regressed),
        "left_correct_right_wrong": corrected[:limit],
        "right_correct_left_wrong": regressed[:limit],
    }


def _f1_from_counts(counts):
    # Counts are [true0/pred0, true0/pred1, true1/pred0, true1/pred1].
    tp = counts[..., [0, 3]]
    denominator = 2 * tp + counts[..., 1, None] + counts[..., 2, None]
    return np.divide(
        2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0
    ).mean(axis=-1)


def paired_bootstrap(
    left_runs: list[list[dict]],
    right_runs: list[list[dict]],
    n_resamples: int = 2000,
    seed: int = 2026,
) -> dict:
    if not left_runs or len(left_runs) != len(right_runs) or n_resamples < 1:
        raise ValueError("Paired runs and a positive bootstrap count are required")
    reference = {r["id"]: (r["label"], r["video_id"]) for r in left_runs[0]}
    if not reference:
        raise ValueError("Cannot pair empty predictions")
    clusters = sorted({video for _, video in reference.values()})
    cluster_index = {video: i for i, video in enumerate(clusters)}
    counts = []
    for group in (left_runs, right_runs):
        group_counts = []
        for records in group:
            mapping = {r["id"]: (r["label"], r["video_id"]) for r in records}
            if len(mapping) != len(records) or mapping != reference:
                raise ValueError("Cannot pair predictions: IDs, labels or video IDs differ")
            table = np.zeros((len(clusters), 4), dtype=np.int64)
            for r in records:
                if r["prediction"] not in (0, 1) or r["label"] not in (0, 1):
                    raise ValueError("Paired predictions must be binary")
                table[cluster_index[r["video_id"]], 2 * r["label"] + r["prediction"]] += 1
            group_counts.append(table)
        counts.append(np.stack(group_counts))
    left, right = counts
    per_seed = _f1_from_counts(left.sum(axis=1)) - _f1_from_counts(right.sum(axis=1))
    rng = np.random.default_rng(seed)
    differences = np.empty(n_resamples)
    for i in range(n_resamples):
        sampled = rng.integers(0, len(clusters), len(clusters))
        differences[i] = (
            _f1_from_counts(left[:, sampled].sum(axis=1))
            - _f1_from_counts(right[:, sampled].sum(axis=1))
        ).mean()
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "mean_difference": float(per_seed.mean()),
        "per_seed_differences": per_seed.tolist(),
        "ci_low": float(low),
        "ci_high": float(high),
        "cluster_count": len(clusters),
        "resamples": n_resamples,
        "bootstrap_seed": seed,
        "interpretation": "95% percentile CI over source-video resampling; trained seeds held fixed",
    }


def _export_figures(groups, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name, runs in groups.items():
        matrix = np.mean(
            [r["evaluation"]["metrics"]["clean"]["confusion_matrix"] for r in runs.values()], axis=0
        )
        fig, ax = plt.subplots(figsize=(4.5, 4))
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="black")
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            xlabel="Predicted",
            ylabel="Annotated",
            title=f"{name}: mean counts across {len(runs)} seed(s)",
        )
        fig.tight_layout()
        fig.savefig(output / f"{name}_confusion.png", dpi=180)
        plt.close(fig)


def summarize_runs(
    run_dirs: list[Path], output: Path, bootstrap_samples: int = 2000, split: str = "test"
) -> dict:
    if not run_dirs:
        raise ValueError("At least one run is required")
    groups, protocol = {}, None
    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        run = read_json(run_dir / "run.json")
        evaluation = read_json(run_dir / "evaluation" / split / "metrics.json")
        signature = fingerprint(
            {
                k: evaluation[k]
                for k in (
                    "split",
                    "data_fingerprint",
                    "label_rule",
                    "conditions",
                    "noise_generator",
                )
            }
        )
        if protocol is not None and signature != protocol:
            raise ValueError("Cannot compare runs with different data or evaluation protocols")
        protocol = signature
        name, seed = evaluation["experiment"], evaluation["seed"]
        # Names become filenames; refuse path components from manually edited metadata.
        if not name or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
            for c in name
        ):
            raise ValueError(
                "Experiment names may contain only letters, digits, hyphens and underscores"
            )
        group = groups.setdefault(name, {})
        if seed in group:
            raise ValueError(f"Duplicate experiment/seed: {name}/{seed}")
        config = read_json(run_dir / "config.json")
        config["training"].pop("seed")
        if group and fingerprint(config) != next(iter(group.values()))["config_fingerprint"]:
            raise ValueError("Model/training settings must match within each seed group")
        group[seed] = {
            "path": run_dir,
            "run": run,
            "evaluation": evaluation,
            "config_fingerprint": fingerprint(config),
        }
    rows, pairs, examples = [], [], []
    for name, runs in sorted(groups.items()):
        first = next(iter(runs.values()))
        for condition in first["evaluation"]["metrics"]:
            row = {
                "experiment": name,
                "condition": condition,
                "n_seeds": len(runs),
                "seeds": ",".join(map(str, sorted(runs))),
            }
            for metric in ("macro_f1", "accuracy", "weighted_f1", "macro_f1_drop"):
                values = [r["evaluation"]["metrics"][condition][metric] for r in runs.values()]
                row[metric + "_mean"] = float(np.mean(values))
                row[metric + "_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            row["trainable_parameters"] = first["run"]["trainable_parameters"]
            row["training_seconds_mean"] = float(
                np.mean([r["run"]["training_seconds"] for r in runs.values()])
            )
            rows.append(row)
    for left_name, right_name in combinations(sorted(groups), 2):
        left, right = groups[left_name], groups[right_name]
        if set(left) != set(right):
            raise ValueError("Paired comparisons require the same seeds in every experiment")
        seeds = sorted(left)
        conditions = next(iter(left.values()))["evaluation"]["metrics"]
        for condition in conditions:

            def predictions(group):
                return [
                    read_predictions(group[s]["path"] / "evaluation" / split / f"{condition}.csv")
                    for s in seeds
                ]

            result = paired_bootstrap(predictions(left), predictions(right), bootstrap_samples)
            if condition == "clean":
                for seed in seeds:
                    a = read_predictions(left[seed]["path"] / "evaluation" / split / "clean.csv")
                    b = read_predictions(right[seed]["path"] / "evaluation" / split / "clean.csv")
                    examples.append(
                        {
                            "left": left_name,
                            "right": right_name,
                            "seed": seed,
                            **error_examples(a, b),
                        }
                    )
            pairs.append(
                {
                    "left": left_name,
                    "right": right_name,
                    "condition": condition,
                    "seeds": seeds,
                    **result,
                }
            )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {
        "split": split,
        "protocol_fingerprint": protocol,
        "summary": rows,
        "paired_differences": pairs,
        "label_rule": next(iter(next(iter(groups.values())).values()))["evaluation"]["label_rule"],
        "notes": [
            "Exactly neutral clips are excluded.",
            "Standard deviation uses ddof=1; absent for one seed.",
            "Bootstrap intervals measure source-video uncertainty conditional on the fitted runs.",
        ],
    }
    write_json(output / "results.json", result)
    write_json(output / "error_examples.json", examples)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if pairs:
        with (output / "paired_differences.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
            writer.writeheader()
            writer.writerows(pairs)
    lines = [
        "# Experiment results",
        "",
        f"Split: {split}. Negative y < 0; positive y > 0; neutral y = 0 excluded.",
        "",
        "| Experiment | Seeds | Macro F1 | SD | Accuracy | Weighted F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["condition"] == "clean":
            sd = "n/a" if row["macro_f1_std"] is None else f"{row['macro_f1_std']:.4f}"
            lines.append(
                f"| {row['experiment']} | {row['n_seeds']} | {row['macro_f1_mean']:.4f} | {sd} | {row['accuracy_mean']:.4f} | {row['weighted_f1_mean']:.4f} |"
            )
    lines.extend(
        [
            "",
            *result["notes"],
            "",
            "See summary.csv for corruption conditions and paired_differences.csv for differences and confidence intervals.",
        ]
    )
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    _export_figures(groups, output)
    return result
