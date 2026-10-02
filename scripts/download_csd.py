"""Download the selected public Kaggle CSD release with resumable ranges."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

from mosei.download import download_file
from mosei.io import read_json, sha256_file, write_json

FILES = {
    "audio": ("CMU-MOSEI/acoustics/CMU_MOSEI_COVAREP.csd", 11585596442),
    "vision": ("CMU-MOSEI/visuals/CMU_MOSEI_VisualOpenFace2.csd", 16718477556),
    "words": ("CMU-MOSEI/languages/CMU_MOSEI_TimestampedWords.csd", 37943951),
    "labels": ("CMU-MOSEI/labels/CMU_MOSEI_Labels.csd", 23254352),
    "glove": ("CMU-MOSEI/languages/CMU_MOSEI_TimestampedWordVectors.csd", 1550778982),
    "phones": ("CMU-MOSEI/languages/CMU_MOSEI_TimestampedPhones.csd", 59563650),
    "facet": ("CMU-MOSEI/visuals/CMU_MOSEI_VisualFacet42.csd", 1656499074),
}


def download_csd(output: Path, files: list[str] | None = None) -> dict:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    selected = files or ["words", "labels", "audio", "vision"]
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(FILES):
        raise ValueError("Select unique known CSD streams")
    with urlopen(
        "https://www.kaggle.com/api/v1/datasets/list/samarwarsi/cmu-mosei", timeout=30
    ) as response:
        current = {f["name"]: f["totalBytes"] for f in json.load(response)["datasetFiles"]}
    if any(current.get(name) != size for name, size in FILES.values()):
        raise ValueError(
            "Kaggle catalogue differs from pinned version 1; inspect before downloading"
        )
    write_json(
        output / "source.json",
        {"dataset": "samarwarsi/cmu-mosei", "version": 1, "selected": selected},
    )

    def fetch(key):
        name, size = FILES[key]
        target = output / Path(name).name
        record = target.with_suffix(".csd.download.json")
        expected = read_json(record)["sha256"] if record.exists() else None
        url = (
            "https://www.kaggle.com/api/v1/datasets/download/samarwarsi/cmu-mosei/"
            + quote(name, safe="")
            + "?datasetVersionNumber=1"
        )
        print(f"Downloading {key}: {name} ({size / 1e9:.2f} GB)", flush=True)
        download_file(
            url, target, expected, workers=4, chunk_bytes=32 * 1024 * 1024, expected_size=size
        )
        result = {
            "source": url,
            "version": 1,
            "file": target.name,
            "bytes": size,
            "sha256": sha256_file(target),
            "checksum_origin": "calculated locally; no independently published checksum",
        }
        write_json(record, result)
        print(f"Completed {key}: {target}", flush=True)
        return key, result

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = dict(executor.map(fetch, selected))
    write_json(output / "downloads.json", results)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/csd"))
    parser.add_argument(
        "--files", nargs="+", choices=FILES, default=["words", "labels", "audio", "vision"]
    )
    args = parser.parse_args()
    download_csd(args.output, args.files)


if __name__ == "__main__":
    main()
