"""Run the shared demo inside its pinned Linux container."""

import argparse
import json
from pathlib import Path


def relocate_tools(assets: Path, work: Path) -> Path:
    """Relocate model files without changing the checkpoint's feature contract."""
    assets = Path(assets).resolve()
    source = assets / "tools/video-tools.json"
    record = json.loads(source.read_text())
    for section in ("text", "speech"):
        relative = Path(record["paths"][section])
        resolved = (source.parent / relative).resolve()
        if relative.is_absolute() or not resolved.is_relative_to(assets):
            raise ValueError("Model paths must be relative and stay inside the demo bundle")
        if not resolved.is_dir():
            raise ValueError(f"Missing bundled {section} model; run ./demo.sh setup again")
        record["paths"][section] = str(resolved)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    output = work / "video-tools.json"
    output.write_text(json.dumps(record, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["doctor", "serve"])
    parser.add_argument("--home", type=Path, required=True)
    args = parser.parse_args()
    import torch

    from mosei.train import restore_run
    from mosei.video_pipeline import VideoClassifier, check_tools, ensure_compatible

    torch.set_num_threads(4)
    work = args.home / "work"
    tools = relocate_tools(args.home / "assets", work)
    run = args.home / "assets/run"
    if args.action == "doctor":
        record = check_tools(tools)
        _, metadata = restore_run(run, "cpu")
        ensure_compatible(metadata, record["manifest"])
        print("Demo ready: checkpoint, model files, feature code and extractor images verified.")
        return
    classifier = VideoClassifier(run, tools, device="cpu", temp_root=work)
    from mosei.web_demo import build_demo

    # All interfaces inside the container; the launcher publishes host loopback only.
    build_demo(classifier).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=[str(work)],
        max_file_size="200mb",
        inbrowser=False,
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Demo setup error: {exc}") from exc
