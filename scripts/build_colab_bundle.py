"""Build a small source/tool-manifest archive for Colab, excluding data and secrets."""

import argparse
import json
import zipfile
from pathlib import Path

from mosei.io import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def build(output: Path):
    files = [ROOT / "pyproject.toml", ROOT / "README.md"]
    for directory, suffixes in [
        ("src/mosei", {".py"}),
        ("configs", {".toml", ".json"}),
        ("scripts", {".py"}),
    ]:
        files.extend(
            p for p in sorted((ROOT / directory).iterdir()) if p.suffix in suffixes and p.is_file()
        )
    files += [
        ROOT / "data/tools/video-tools.json",
        ROOT / "data/tools/feature-manifest.json",
        ROOT / "data/csd/splits.json",
    ]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing bundle input; run video tool setup first: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): sha256_file(p) for p in files}
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(ROOT)))
        archive.writestr("bundle-manifest.json", json.dumps(hashes, indent=2))
    print(f"Colab source bundle: {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/mosei-colab-source.zip")
    build(parser.parse_args().output)
