"""Fetch and verify the pinned official video folds without executing SDK code."""

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

from mosei.csd import parse_sdk_splits
from mosei.io import write_json

SDK_REVISION = "4f2eadcd7e7b9e20e83b868cdad385b41830285c"
SDK_SHA256 = "4e13a2f538cf02ab56144eb2a91ca48a4024ce64faaae790ab9ff1c3f399b473"
SDK_URL = (
    "https://raw.githubusercontent.com/CMU-MultiComp-Lab/CMU-MultimodalSDK/"
    f"{SDK_REVISION}/mmsdk/mmdatasdk/dataset/standard_datasets/CMU_MOSEI/cmu_mosei_std_folds.py"
)


def download_splits(output: Path):
    with urlopen(SDK_URL, timeout=60) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != SDK_SHA256:
        raise ValueError("Official split source checksum mismatch")
    splits = parse_sdk_splits(source.decode("utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, {"source_url": SDK_URL, "sha256": SDK_SHA256, "splits": splits})
    print(f"Verified official source-video folds: { {k: len(v) for k, v in splits.items()} }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/csd/splits.json"))
    download_splits(parser.parse_args().output)
