"""Recoverable transfer of verified prepared caches between Colab and Drive."""

import shutil
import tarfile
import tempfile
import uuid
from pathlib import Path

from .data import load_data, window_cache_files
from .io import read_json, sha256_file, write_json


def _validate(cache):
    data = load_data(cache)
    if data.audit["cache_version"] != 2:
        raise ValueError("Colab video training requires the content-verified CSD cache")
    return data.audit


def _record(archive):
    marker = archive.with_suffix(archive.suffix + ".json")
    if not archive.is_file() or not marker.is_file():
        return None
    try:
        record = read_json(marker)
        if archive.stat().st_size != record["bytes"]:
            return None
        if not isinstance(record["sha256"], str) or not record["data_fingerprint"]:
            return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def restore_cache(cache: Path, archive: Path) -> bool:
    """Reuse verified local content, or atomically restore a complete archive."""
    cache, archive = Path(cache), Path(archive)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        try:
            _validate(cache)
            return True
        except (OSError, ValueError, KeyError, EOFError):
            preserved = cache.with_name(cache.name + "-interrupted-" + uuid.uuid4().hex)
            cache.rename(preserved)
            print(f"Incomplete or changed runtime cache preserved at {preserved}", flush=True)
    record = _record(archive)
    if record is None:
        return False
    with tempfile.TemporaryDirectory(prefix=".cache-restore-", dir=cache.parent) as temporary:
        stage = Path(temporary)
        local_archive = stage / "prepared.tar"
        shutil.copy2(archive, local_archive)
        if sha256_file(local_archive) != record["sha256"]:
            print("Drive archive is incomplete or changed; rebuilding from CSD source.", flush=True)
            return False
        with tarfile.open(local_archive) as tar:
            expected = {"prepared-csd/" + name for name in ["audit.json", *window_cache_files()]}
            members = tar.getmembers()
            if (
                {m.name for m in members} != expected
                or len(members) != len(expected)
                or any(not m.isfile() for m in members)
            ):
                raise ValueError("Prepared archive has missing or unexpected content")
            tar.extractall(stage, filter="data")
        restored = stage / "prepared-csd"
        audit = _validate(restored)
        if audit["fingerprint"] != record["data_fingerprint"]:
            raise ValueError("Prepared archive data fingerprint mismatch")
        restored.rename(cache)
    return True


def persist_cache(cache: Path, archive: Path) -> dict:
    """Publish archive then completion record; retry even when local data exists."""
    cache, archive = Path(cache), Path(archive)
    audit = _validate(cache)
    record = _record(archive)
    if (
        record is not None
        and record["data_fingerprint"] == audit["fingerprint"]
        and sha256_file(archive) == record["sha256"]
    ):
        return record
    archive.parent.mkdir(parents=True, exist_ok=True)
    transfer = archive.with_name(archive.name + ".uploading-" + uuid.uuid4().hex)
    try:
        with tempfile.TemporaryDirectory(prefix=".cache-publish-", dir=cache.parent) as temporary:
            local_archive = Path(temporary) / "prepared.tar"
            with tarfile.open(local_archive, "w") as tar:
                # Export only known dataset content, never arbitrary extra local files.
                for name in ["audit.json", *window_cache_files()]:
                    tar.add(cache / name, arcname="prepared-csd/" + name, recursive=False)
            record = {
                "sha256": sha256_file(local_archive),
                "bytes": local_archive.stat().st_size,
                "data_fingerprint": audit["fingerprint"],
            }
            shutil.copy2(local_archive, transfer)
            if sha256_file(transfer) != record["sha256"]:
                raise ValueError("Prepared cache transfer checksum mismatch")
            transfer.replace(archive)
            write_json(archive.with_suffix(archive.suffix + ".json"), record)
    finally:
        transfer.unlink(missing_ok=True)
    return record
