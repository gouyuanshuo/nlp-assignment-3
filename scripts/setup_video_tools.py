"""Download pinned local models and build isolated, fingerprinted video extractors."""

import argparse
import importlib.metadata
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.request import urlopen

from mosei.io import fingerprint, read_json, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
LPC_REVISION = "e4fc6cfd222d6be50429301ec2ea91b26f9367b1"


def command(args, **kwargs):
    return subprocess.run([str(a) for a in args], check=True, **kwargs)


def image_id(name):
    result = subprocess.run(
        ["docker", "image", "inspect", name, "--format", "{{.Id}}"], capture_output=True, text=True
    )
    return result.stdout.strip() if result.returncode == 0 else None


def replace_once(path, old, new):
    source = path.read_text()
    if source.count(old) != 1:
        raise ValueError(f"Pinned COVAREP source changed at compatibility patch: {path.name}")
    path.write_text(source.replace(old, new))


def covarep_image(directory, revision):
    existing = image_id("mosei-covarep:prototype-v1")
    if existing:
        return existing
    source = directory / "covarep"
    if not source.exists():
        command(["git", "clone", "https://github.com/covarep/covarep.git", source])
        command(["git", "checkout", revision], cwd=source)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
    if (
        actual != revision
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("COVAREP source is not the clean pinned checkout")
    if not image_id("mosei-covarep:local"):
        command(
            [
                "docker",
                "build",
                "-t",
                "mosei-covarep:local",
                "-f",
                ROOT / "tools/video/Dockerfile.covarep",
                ROOT / "tools/video",
            ]
        )
    with tempfile.TemporaryDirectory(prefix="covarep-build-", dir=directory) as temporary:
        context = Path(temporary)
        shutil.copytree(source, context / "covarep", ignore=shutil.ignore_patterns(".git"))
        shutil.copytree(ROOT / "tools/video", context / "compat")
        with urlopen(
            f"https://raw.githubusercontent.com/gnu-octave/octave-signal/{LPC_REVISION}/inst/lpc.m",
            timeout=60,
        ) as response:
            (context / "compat/lpc.m").write_bytes(response.read())
        for filename, old, new in [
            ("env_te.m", "ifft(A,[],2,'symmetric')", "real(ifft(A,[],2))"),
            ("hspec2fwcep.m", "ifft(log(abs(Cwrap)),'symmetric')", "real(ifft(log(abs(Cwrap))))"),
        ]:
            replace_once(context / "covarep/envelope" / filename, old, new)
        extractor = context / "covarep/feature_extraction/COVAREP_feature_extraction.m"
        code = extractor.read_text()
        first = code.index("        % Creaky voice detection")
        last = code.index("        % Spectral envelope parameterisation", first)
        extractor.write_text(
            code[:first] + "        % Omitted: MATLAB-only creaky-voice classifier.\n" + code[last:]
        )
        for old, new in [
            ("'Rd_conf','creak','MCEP_0'", "'Rd_conf','MCEP_0'"),
            ("Rd(:) Rd_conf(:) creak_pp(:) MCEP_int", "Rd(:) Rd_conf(:) MCEP_int"),
            (
                "save([in_dir filesep basename '.mat'],'features','names')",
                "intervals=[(0:length(feature_sampling)-1)'*sample_rate, (1:length(feature_sampling))'*sample_rate];\n        save('-mat7-binary',[in_dir filesep basename '.mat'],'features','names','intervals')",
            ),
            (
                "warning(['An error occurred while analysing ' basename ': ' getReport(err)])",
                "rethrow(err)",
            ),
        ]:
            replace_once(extractor, old, new)
        command(
            [
                "docker",
                "build",
                "-t",
                "mosei-covarep:prototype-v1",
                "-f",
                ROOT / "tools/video/Dockerfile.runtime",
                context,
            ]
        )
    return image_id("mosei-covarep:prototype-v1")


def setup(output, models):
    from huggingface_hub import snapshot_download

    output, models = Path(output).resolve(), Path(models).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(ROOT / "configs/video_features.json")
    paths = {}
    for section, folder, patterns in [
        (
            "text",
            "bert",
            [
                "config.json",
                "model.safetensors",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.txt",
                "special_tokens_map.json",
            ],
        ),
        (
            "speech",
            "whisper",
            [
                "config.json",
                "model.bin",
                "tokenizer.json",
                "vocabulary.txt",
                "preprocessor_config.json",
            ],
        ),
    ]:
        spec = manifest[section]
        path = models / folder
        snapshot_download(
            spec["repo"],
            revision=spec["revision"],
            local_dir=path,
            allow_patterns=patterns,
            max_workers=2,
        )
        spec["files_sha256"] = {
            p.name: sha256_file(p)
            for p in sorted(path.iterdir())
            if p.is_file() and p.name in patterns
        }
        paths[section] = str(path)
    manifest["audio"]["image"] = covarep_image(output, manifest["audio"]["covarep_commit"])
    if not image_id("mosei-openface:prototype"):
        command(
            [
                "docker",
                "build",
                "-t",
                "mosei-openface:prototype",
                "-f",
                ROOT / "tools/video/Dockerfile.openface",
                ROOT / "tools/video",
            ]
        )
    manifest["vision"]["image"] = image_id("mosei-openface:prototype")
    manifest["audio"]["octave_lpc_revision"] = LPC_REVISION
    manifest["audio"]["compatibility"] = (
        "Voicebox signed peaks; linear interp1q; real symmetric IFFT; modern audioread; MATLAB-only creak excluded"
    )
    manifest["software"] = {
        p: importlib.metadata.version(p)
        for p in ("transformers", "tokenizers", "faster-whisper", "ctranslate2", "numpy", "scipy")
    }
    manifest["ffmpeg_version"] = subprocess.check_output(
        ["ffmpeg", "-version"], text=True
    ).splitlines()[0]
    manifest["preprocessing_sha256"] = {
        name: sha256_file(ROOT / "src/mosei" / name)
        for name in (
            "video_features.py",
            "text_features.py",
            "video_extractors.py",
            "video_media.py",
            "csd_prepare.py",
        )
    }
    record = {"manifest": manifest, "fingerprint": fingerprint(manifest), "paths": paths}
    write_json(output / "video-tools.json", record)
    write_json(output / "feature-manifest.json", manifest)
    print(
        f"Video tools: {output / 'video-tools.json'}\nFeature fingerprint: {record['fingerprint']}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/tools")
    parser.add_argument("--models", type=Path, default=ROOT / "data/models")
    args = parser.parse_args()
    setup(args.output, args.models)
