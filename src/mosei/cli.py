"""Command-line entry points; use `python -m mosei --help`."""

import argparse
import json
from dataclasses import replace
from pathlib import Path

from torch.utils.data import Subset

from .config import load_config
from .data import MOSEI_SHA256, load_data, prepare_data
from .evaluate import evaluate_run
from .report import summarize_runs
from .train import choose_device, predict, restore_run, train_run


def _parser():
    parser = argparse.ArgumentParser(
        description="CMU-MOSEI non-neutral binary sentiment experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare", help="Verify and prepare the trusted aligned feature archive"
    )
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument(
        "--sha256",
        default=MOSEI_SHA256,
        help="Expected trusted archive checksum; defaults to official MMSA",
    )
    train = commands.add_parser("train", help="Train using only train/valid splits")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--seed", type=int)
    train.add_argument("--epochs", type=int)
    train.add_argument("--device", choices=["auto", "cpu", "cuda"])
    train.add_argument(
        "--train-fraction",
        type=float,
        dest="train_fraction",
        help="Fraction of training videos to keep, stratified by majority label",
    )
    train.add_argument(
        "--resume",
        action="store_true",
        help="Resume the saved epoch state with unchanged config/data",
    )
    evaluate = commands.add_parser("evaluate", help="Evaluate a frozen checkpoint")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--run", type=Path, required=True)
    evaluate.add_argument("--split", choices=["valid", "test"], default="test")
    evaluate.add_argument(
        "--robustness", action="store_true", help="Also run all 12 matched corruption conditions"
    )
    evaluate.add_argument("--batch-size", type=int)
    evaluate.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    report = commands.add_parser(
        "report", help="Export matched seed summaries and paired video-cluster intervals"
    )
    report.add_argument("--runs", nargs="+", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)
    report.add_argument("--split", choices=["valid", "test"], default="test")
    report.add_argument("--bootstrap-samples", type=int, default=2000)
    demo = commands.add_parser(
        "demo", help="Show one held-out clip with its transcript, annotation and predictions"
    )
    demo.add_argument("--data", type=Path, required=True)
    demo.add_argument("--run", type=Path, required=True)
    demo.add_argument("--baseline", type=Path, help="Text-only run to display alongside fusion")
    demo.add_argument("--split", choices=["valid", "test"], default="test")
    clip = demo.add_mutually_exclusive_group()
    clip.add_argument("--index", type=int, default=0)
    clip.add_argument("--clip-id")
    demo.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    csd = commands.add_parser(
        "prepare-csd", help="Prepare word-aligned CSD features for video inference"
    )
    csd.add_argument("--source", type=Path, required=True)
    csd.add_argument("--output", type=Path, required=True)
    csd.add_argument("--tools", type=Path, required=True)
    csd.add_argument("--splits", type=Path, required=True)
    csd.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    csd.add_argument("--max-clips", type=int, help="Integration smoke only: cap each split")
    doctor = commands.add_parser(
        "doctor", help="Verify models, preprocessing and native extraction tools"
    )
    doctor.add_argument("--tools", type=Path, required=True)
    doctor.add_argument("--run", type=Path, help="Also verify checkpoint compatibility")
    for name, help_text in [
        ("video", "Classify an uploaded/new local video"),
        ("serve", "Launch the localhost video-upload interface"),
    ]:
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--run", type=Path, required=True)
        command.add_argument("--tools", type=Path, required=True)
        command.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
        if name == "video":
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--output", type=Path)
        else:
            command.add_argument("--host", choices=["127.0.0.1", "localhost"], default="127.0.0.1")
            command.add_argument("--port", type=int, default=7860)
    return parser


def _demo(args):
    data = load_data(args.data)
    dataset = data.splits[args.split]
    index = dataset.ids.index(args.clip_id) if args.clip_id is not None else args.index
    if not 0 <= index < len(dataset):
        raise ValueError(f"Clip index must be between 0 and {len(dataset) - 1}")
    device = choose_device(args.device)
    model, metadata = restore_run(args.run, device)
    if metadata["data_fingerprint"] != data.fingerprint:
        raise ValueError("Demo data differs from checkpoint preprocessing")
    record = predict(model, Subset(dataset, [index]), 1, device)[0]
    names = ["negative", "positive"]
    result = {
        "id": record["id"],
        "split": args.split,
        "transcript": dataset.raw_text[index],
        "annotation": names[record["label"]],
        "annotation_score": record["score"],
        "prediction": names[record["prediction"]],
        "probabilities": {"negative": record["prob_negative"], "positive": record["prob_positive"]},
        "task": "Positive versus negative sentiment; neutral excluded",
        "experiment": metadata["config"]["name"],
    }
    if args.baseline:
        baseline, meta = restore_run(args.baseline, device)
        if (
            meta["data_fingerprint"] != data.fingerprint
            or meta["config"]["model"]["name"] != "text"
        ):
            raise ValueError("Baseline must be a text-only model trained on the same prepared data")
        base = predict(baseline, Subset(dataset, [index]), 1, device)[0]
        result["text_baseline"] = {
            "prediction": names[base["prediction"]],
            "probabilities": {"negative": base["prob_negative"], "positive": base["prob_positive"]},
        }
    return result


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_data(args.source, args.output, args.sha256)
        elif args.command == "prepare-csd":
            from .csd_prepare import prepare_csd
            from .video_pipeline import check_tools

            tools = check_tools(args.tools, native=False)
            from .io import read_json

            if read_json(args.tools.parent / "feature-manifest.json") != tools["manifest"]:
                raise ValueError("Feature manifest differs from the verified tool bundle")
            result = prepare_csd(
                args.source,
                args.output,
                args.tools.parent / "feature-manifest.json",
                args.splits,
                model_path=Path(tools["paths"]["text"]),
                device=choose_device(args.device),
                max_clips=args.max_clips,
            )
            if result["feature_manifest"] != tools["manifest"]:
                raise ValueError("Prepared feature manifest differs from the tool bundle")
        elif args.command == "doctor":
            from .video_pipeline import check_tools, ensure_compatible

            tools = check_tools(args.tools)
            if args.run:
                _, metadata = restore_run(args.run)
                ensure_compatible(metadata, tools["manifest"])
            result = {"ready": True, "feature_fingerprint": tools["fingerprint"]}
        elif args.command in {"video", "serve"}:
            from .video_pipeline import VideoClassifier

            classifier = VideoClassifier(args.run, args.tools, args.device)
            if args.command == "video":
                result = classifier.classify(args.input)
                if args.output:
                    from .io import write_json

                    write_json(args.output, result)
            else:
                from .web_demo import build_demo

                build_demo(classifier).launch(
                    server_name=args.host,
                    server_port=args.port,
                    share=False,
                    show_api=False,
                    max_file_size=200_000_000,
                )
                return 0
        elif args.command == "train":
            config = load_config(args.config)
            overrides = {
                k: getattr(args, k)
                for k in ("seed", "epochs", "device", "train_fraction")
                if getattr(args, k) is not None
            }
            config = replace(config, training=replace(config.training, **overrides))
            trained = train_run(load_data(args.data), config, args.output, resume=args.resume)
            result = {
                "run": str(args.output),
                "best_epoch": trained["best_epoch"],
                "best_validation": trained["best_validation"],
            }
        elif args.command == "evaluate":
            result = evaluate_run(
                load_data(args.data),
                args.run,
                args.split,
                args.robustness,
                args.device,
                args.batch_size,
            )
        elif args.command == "report":
            summary = summarize_runs(args.runs, args.output, args.bootstrap_samples, args.split)
            result = {"output": str(args.output), "summary_rows": len(summary["summary"])}
        else:
            result = _demo(args)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0
