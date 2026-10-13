import json
import os
import runpy
import socket
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def runtime():
    path = ROOT / "scripts/demo_runtime.py"
    assert path.is_file(), "Portable demo runtime has not been implemented"
    return runpy.run_path(str(path))


def test_relocated_assets_preserve_manifest_and_resolve_model_paths(tmp_path):
    assets = tmp_path / "classmate project" / "assets"
    (assets / "tools").mkdir(parents=True)
    (assets / "models/bert").mkdir(parents=True)
    (assets / "models/whisper").mkdir(parents=True)
    record = {
        "manifest": {"schema_version": 1, "software": {"numpy": "test-version"}},
        "fingerprint": "unchanged-training-fingerprint",
        "paths": {"text": "../models/bert", "speech": "../models/whisper"},
    }
    source = assets / "tools/video-tools.json"
    source.write_text(json.dumps(record))
    output = runtime()["relocate_tools"](assets, tmp_path / "work")
    relocated = json.loads(output.read_text())
    assert relocated["manifest"] == record["manifest"]
    assert relocated["fingerprint"] == record["fingerprint"]
    assert Path(relocated["paths"]["text"]) == assets / "models/bert"
    assert Path(relocated["paths"]["speech"]) == assets / "models/whisper"
    assert json.loads(source.read_text()) == record


@pytest.mark.parametrize("bad_path", ["/home/author/models", "../../../outside"])
def test_relocation_rejects_model_paths_outside_bundle(tmp_path, bad_path):
    assets = tmp_path / "assets"
    (assets / "tools").mkdir(parents=True)
    (assets / "tools/video-tools.json").write_text(
        json.dumps(
            {
                "manifest": {},
                "fingerprint": "x",
                "paths": {"text": bad_path, "speech": "../models/whisper"},
            }
        )
    )
    with pytest.raises(ValueError, match="bundle|relative"):
        runtime()["relocate_tools"](assets, tmp_path / "work")


def test_launcher_reports_missing_bundle_before_starting_demo(tmp_path):
    launcher = ROOT / "demo.sh"
    assert launcher.is_file(), "Classmate launcher has not been implemented"
    result = subprocess.run(
        ["bash", str(launcher), "doctor"],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "MOSEI_DEMO_HOME": str(tmp_path / "missing")},
    )
    assert result.returncode != 0
    assert "setup" in result.stderr.lower()


def test_export_omits_training_data_and_private_model_cache(tmp_path):
    builder = ROOT / "scripts/build_demo_bundle.py"
    assert builder.is_file(), "Demo bundle builder has not been implemented"
    stage = runpy.run_path(str(builder))["stage_assets"]
    tools = tmp_path / "tools.json"
    model = tmp_path / "author-model"
    model.mkdir()
    (model / "weights.bin").write_bytes(b"fixed model")
    (model / "private-token").write_text("must not ship")
    from mosei.io import fingerprint, sha256_file

    manifest = {
        "text": {"files_sha256": {"weights.bin": sha256_file(model / "weights.bin")}},
        "speech": {"files_sha256": {"weights.bin": sha256_file(model / "weights.bin")}},
    }
    tools.write_text(
        json.dumps(
            {
                "manifest": manifest,
                "fingerprint": fingerprint(manifest),
                "paths": {"text": str(model), "speech": str(model)},
            }
        )
    )
    run = tmp_path / "run"
    run.mkdir()
    (run / "best.pt").write_bytes(b"checkpoint")
    (run / "run.json").write_text(json.dumps({"feature_manifest": manifest}))
    (run / "last.pt").write_bytes(b"optimizer and training state")
    target = tmp_path / "staged"
    stage(tools, run, target)
    files = {str(p.relative_to(target)) for p in target.rglob("*") if p.is_file()}
    assert files == {
        "tools/video-tools.json",
        "models/bert/weights.bin",
        "models/whisper/weights.bin",
        "run/best.pt",
        "run/run.json",
    }
    assert json.loads((target / "tools/video-tools.json").read_text())["paths"] == {
        "text": "../models/bert",
        "speech": "../models/whisper",
    }


@pytest.mark.parametrize("engine", ["rootless", "desktop"])
def test_launcher_uses_daemon_socket_and_correct_user_namespace(tmp_path, engine):
    # Docker is the external process boundary. Exercise actual shell argument assembly.
    home = tmp_path / "demo home"
    home.mkdir()
    (home / "demo-release.json").write_text(json.dumps({"image": "sha256:" + "a" * 64}))
    docker = tmp_path / "docker"
    log = tmp_path / "arguments.json"
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(tmp_path / "docker.sock"))
    docker.write_text(
        f"#!{sys.executable}\n"
        + """import json, os, sys
from pathlib import Path
if sys.argv[1] == "info":
    if "{{json .DriverStatus}}" in sys.argv:
        print('[["driver-type", "io.containerd.snapshotter.v1"]]')
    elif "{{.OperatingSystem}}" in sys.argv:
        print("Docker Desktop" if os.environ["DOCKER_TEST_ENGINE"] == "desktop" else "Ubuntu")
    else:
        print('["name=rootless"]' if os.environ["DOCKER_TEST_ENGINE"] == "rootless" else '[]')
elif sys.argv[1] == "run":
    if "--entrypoint" in sys.argv:
        print("123")
    else:
        Path(os.environ["DOCKER_TEST_LOG"]).write_text(json.dumps(sys.argv[2:]))
"""
    )
    docker.chmod(0o755)
    try:
        result = subprocess.run(
            ["bash", str(ROOT / "demo.sh"), "doctor"],
            env={
                **os.environ,
                "PATH": str(tmp_path) + ":/usr/bin:/bin",
                "MOSEI_DEMO_HOME": str(home),
                "DOCKER_TEST_LOG": str(log),
                "DOCKER_TEST_ENGINE": engine,
                "DOCKER_HOST": "unix://" + str(tmp_path / "docker.sock"),
            },
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr
        args = json.loads(log.read_text())
        assert args[args.index("--user") + 1] == (
            "0:0" if engine == "rootless" else f"{os.getuid()}:{os.getgid()}"
        )
        if engine == "desktop":
            assert "type=bind,source=/var/run/docker.sock.raw,target=/var/run/docker.sock" in args
            assert args[args.index("--group-add") + 1] == "123"
        assert "--publish" not in args
        assert f"type=bind,source={home}/work,target={home}/work" in args
    finally:
        sock.close()


def test_setup_rejects_unverified_image_parts_before_docker_load(tmp_path):
    import hashlib

    downloads = tmp_path / "downloads"
    downloads.mkdir()
    checksums = []
    for name in ["demo-assets.tar.gz", "demo-release.json", "demo-images.tar.gz.part-000"]:
        data = b"fixture"
        (downloads / name).write_bytes(data)
        checksums.append(f"{hashlib.sha256(data).hexdigest()}  {name}\n")
    (downloads / "SHA256SUMS").write_text("".join(checksums))
    (downloads / "demo-images.tar.gz.part-001").write_bytes(b"unverified extra data")
    docker = tmp_path / "docker"
    docker.write_text(
        "#!/bin/sh\nif [ \"$1\" = info ]; then printf 'io.containerd.snapshotter.v1'; exit 0; fi\nprintf 'UNEXPECTED_DOCKER_LOAD' >&2\nexit 99\n"
    )
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "demo.sh"), "setup", str(downloads)],
        env={
            **os.environ,
            "PATH": str(tmp_path) + ":/usr/bin:/bin",
            "MOSEI_DEMO_HOME": str(tmp_path / "home"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "checksum file list" in result.stderr.lower()
    assert "UNEXPECTED_DOCKER_LOAD" not in result.stderr


def test_setup_rejects_classic_image_store_before_downloading(tmp_path):
    docker = tmp_path / "docker"
    docker.write_text("#!/bin/sh\nprintf '[]'\n")
    docker.chmod(0o755)
    result = subprocess.run(
        ["bash", str(ROOT / "demo.sh"), "setup", str(tmp_path / "downloads")],
        env={
            **os.environ,
            "PATH": str(tmp_path) + ":/usr/bin:/bin",
            "MOSEI_DEMO_HOME": str(tmp_path / "home"),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "containerd" in result.stderr
    assert not (tmp_path / "downloads").exists()
