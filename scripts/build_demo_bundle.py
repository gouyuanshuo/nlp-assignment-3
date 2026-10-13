"""Export a configured author's demo, without datasets, credentials or training state."""

import argparse
import gzip
import json
import shutil
import subprocess
import tarfile
from pathlib import Path

from mosei.io import fingerprint, read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def stage_assets(tools: Path, run: Path, target: Path):
    record = read_json(tools)
    manifest = record["manifest"]
    if fingerprint(manifest) != record["fingerprint"]:
        raise ValueError("Tool manifest fingerprint mismatch")
    if read_json(run / "run.json")["feature_manifest"] != manifest:
        raise ValueError("Checkpoint and tools have different feature manifests")
    target.mkdir(parents=True, exist_ok=False)
    for section, folder in [("text", "bert"), ("speech", "whisper")]:
        source = Path(record["paths"][section])
        destination = target / "models" / folder
        destination.mkdir(parents=True)
        for name, digest in manifest[section]["files_sha256"].items():
            if Path(name).name != name or name in {".", ".."}:
                raise ValueError("Invalid model filename in manifest")
            if sha256_file(source / name) != digest:
                raise ValueError(f"Model checksum mismatch: {section}/{name}")
            shutil.copy2(source / name, destination / name)
        record["paths"][section] = f"../models/{folder}"
    (target / "run").mkdir()
    for name in ("best.pt", "run.json"):
        shutil.copy2(run / name, target / "run" / name)
    write_json(target / "tools/video-tools.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/demo-v1")
    parser.add_argument("--tools", type=Path, default=ROOT / "data/tools/video-tools.json")
    parser.add_argument("--run", type=Path, default=ROOT / "runs/colab/selected-run")
    parser.add_argument("--image", default="mosei-demo:20261013")
    args = parser.parse_args()
    from mosei.train import restore_run
    from mosei.video_pipeline import check_tools, ensure_compatible

    record = check_tools(args.tools)
    _, metadata = restore_run(args.run, "cpu")
    ensure_compatible(metadata, record["manifest"])
    args.output.mkdir(parents=True, exist_ok=False)
    assets = args.output / "assets"
    stage_assets(args.tools, args.run, assets)
    with tarfile.open(args.output / "demo-assets.tar.gz", "w:gz", compresslevel=1) as archive:
        archive.add(assets, arcname="assets")
    app_id = subprocess.check_output(
        ["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    images = [
        args.image,
        record["manifest"]["audio"]["image"],
        record["manifest"]["vision"]["image"],
    ]
    write_json(
        args.output / "demo-release.json",
        {
            "schema_version": 1,
            "image": app_id,
            "platform": "linux/amd64",
            "feature_fingerprint": record["fingerprint"],
        },
    )
    compressed = args.output / "demo-images.tar.gz"
    print("Exporting pinned Docker images…", flush=True)
    with gzip.open(compressed, "wb", compresslevel=1) as output:
        process = subprocess.Popen(["docker", "save", *images], stdout=subprocess.PIPE)
        try:
            shutil.copyfileobj(process.stdout, output, length=1024 * 1024)
        finally:
            process.stdout.close()
        if process.wait():
            raise RuntimeError("docker save failed")
    # Each GitHub release asset must be smaller than 2 GiB.
    with compressed.open("rb") as source:
        part = 0
        while block := source.read(16 * 1024 * 1024):
            with (args.output / f"demo-images.tar.gz.part-{part:03d}").open("wb") as out:
                out.write(block)
                remaining = 1500 * 1024 * 1024 - len(block)
                while remaining and (block := source.read(min(remaining, 16 * 1024 * 1024))):
                    out.write(block)
                    remaining -= len(block)
            part += 1
    files = [
        args.output / "demo-assets.tar.gz",
        args.output / "demo-release.json",
        *sorted(args.output.glob("demo-images.tar.gz.part-*")),
    ]
    (args.output / "SHA256SUMS").write_text("".join(f"{sha256_file(p)}  {p.name}\n" for p in files))
    print(
        json.dumps(
            {
                "output": str(args.output),
                "assets": len(files),
                "download_bytes": sum(p.stat().st_size for p in files),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
