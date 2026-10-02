"""Resumable public dataset download with a mandatory published checksum."""

import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import Request, urlopen

from .data import MOSEI_SHA256
from .io import read_json, sha256_file, write_json

KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/neffow/cmu-mosei/aligned_50.pkl?datasetVersionNumber=1"
AUTHORS_URL = "https://drive.usercontent.google.com/download?id=16LfoCDw8LncJMVHaCOhN3H3_jjiKBmPu&export=download&confirm=t"


def download_file(
    url: str,
    output: Path,
    expected_sha256: str = MOSEI_SHA256,
    workers: int = 4,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> Path:
    output = Path(output)
    if workers < 1 or chunk_bytes < 1:
        raise ValueError("Download workers and chunk size must be positive")
    if output.exists():
        if sha256_file(output) != expected_sha256:
            raise ValueError(f"Existing file SHA-256 differs from the trusted checksum: {output}")
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as response:
        content_range = response.headers.get("Content-Range", "")
        if response.status != 206 or not content_range.startswith("bytes 0-0/"):
            raise ValueError(
                "Source did not return byte ranges; check the dataset download URL/access"
            )
        total = int(content_range.rsplit("/", 1)[1])
        resolved_url = response.geturl()
    parts = output.with_suffix(output.suffix + ".parts")
    info = {"total_bytes": total, "expected_sha256": expected_sha256, "chunk_bytes": chunk_bytes}
    parts.mkdir(exist_ok=True)
    if (parts / "download.json").exists() and read_json(parts / "download.json") != info:
        raise ValueError(
            "Existing download chunks use a different size/checksum; use a new output path"
        )
    write_json(parts / "download.json", info)
    ranges = [
        (index, low, min(low + chunk_bytes, total) - 1)
        for index, low in enumerate(range(0, total, chunk_bytes))
    ]

    def fetch(entry):
        index, low, high = entry
        path = parts / f"{index:06d}"
        if path.exists() and path.stat().st_size == high - low + 1:
            return
        for attempt in range(3):
            try:
                with urlopen(
                    Request(resolved_url, headers={"Range": f"bytes={low}-{high}"}), timeout=90
                ) as response:
                    if (
                        response.status != 206
                        or response.headers.get("Content-Range") != f"bytes {low}-{high}/{total}"
                    ):
                        raise ValueError("Server returned an unexpected byte range")
                    with path.with_suffix(".tmp").open("wb") as stream:
                        shutil.copyfileobj(response, stream, 1024 * 1024)
                if path.with_suffix(".tmp").stat().st_size != high - low + 1:
                    raise ValueError("Incomplete dataset download chunk")
                path.with_suffix(".tmp").replace(path)
                return
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for completed, future in enumerate(
            as_completed([pool.submit(fetch, r) for r in ranges]), 1
        ):
            future.result()
            print(f"Dataset chunks: {completed}/{len(ranges)}", flush=True)
    staging = output.with_suffix(output.suffix + ".assembling")
    with staging.open("wb") as target:
        for index, _, _ in ranges:
            with (parts / f"{index:06d}").open("rb") as source:
                shutil.copyfileobj(source, target, 8 * 1024 * 1024)
    digest = sha256_file(staging)
    if digest != expected_sha256:
        raise ValueError(f"Downloaded SHA-256 mismatch: {digest}; archive has not been accepted")
    staging.replace(output)
    shutil.rmtree(parts)
    return output
