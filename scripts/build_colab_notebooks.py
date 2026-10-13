"""Generate portable Colab notebooks with explicit, inspectable execution stages."""

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": dedent(text).strip() + "\n",
    }


POST_INSTALL = """
# Editable-install .pth files are not discovered again by a running notebook kernel.
sys.path.insert(0, str(PROJECT / 'src'))
stale_packages = {}
for distribution, expected in tools['manifest']['software'].items():
    loaded = sys.modules.get(distribution.replace('-', '_'))
    current = getattr(loaded, '__version__', None)
    if current is not None and current != expected:
        stale_packages[distribution] = {'loaded': current, 'installed': expected}
if stale_packages:
    raise RuntimeError(
        'Restart the Colab session (Runtime > Restart session), then rerun the setup cells. '
        'The kernel still holds packages from before installation: ' + str(stale_packages)
    )
print('Installed the source bundle and matching feature-library versions.')
"""


EXTRACT = (
    """
import hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

PROJECT = Path('/content/mosei-project')
PROJECT.mkdir(exist_ok=True)
with zipfile.ZipFile(BUNDLE) as archive:
    for item in archive.infolist():
        target = (PROJECT / item.filename).resolve()
        if not target.is_relative_to(PROJECT.resolve()):
            raise ValueError('Unsafe source archive path')
    archive.extractall(PROJECT)
hashes = json.loads((PROJECT / 'bundle-manifest.json').read_text())
for name, expected in hashes.items():
    if hashlib.sha256((PROJECT / name).read_bytes()).hexdigest() != expected:
        raise ValueError(f'Source bundle checksum mismatch: {name}')
os.chdir(PROJECT)
os.environ.update(USE_TF='0', USE_FLAX='0', TOKENIZERS_PARALLELISM='false')
tools = json.loads((PROJECT / 'data/tools/video-tools.json').read_text())
versions = [f'{name}=={version}' for name, version in tools['manifest']['software'].items()]
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', '.[video]', *versions], check=True)
"""
    + POST_INSTALL
)


def build():
    download = [
        markdown("""
        # CMU-MOSEI · 1. Download the selected CSD streams

        Use a **CPU runtime** for this notebook to conserve your Colab compute units.
        It downloads the four required streams from `samarwarsi/cmu-mosei`, version 1
        (about 28.4 GB), into your own Google Drive. Alternative FACET/GloVe/phone
        streams are not inputs to this model. The CSD package contains features,
        words and labels, rather than raw videos.

        First build `dist/mosei-colab-source.zip` in the local project with
        `.venv/bin/python scripts/build_colab_bundle.py`. Upload that small archive
        below. No dataset or model weights are inside it.

        Drive and Colab authentication are performed interactively by you. Nothing
        is shared publicly. Downloads resume from saved chunks if a session ends.
        """),
        code("""
        from google.colab import drive, files
        from pathlib import Path
        drive.mount('/content/drive')
        STORE = Path('/content/drive/MyDrive/NLP_Assignment_3')
        STORE.mkdir(parents=True, exist_ok=True)
        uploaded = files.upload()  # Choose mosei-colab-source.zip only.
        if len(uploaded) != 1:
            raise ValueError('Upload exactly one source bundle')
        name, content = next(iter(uploaded.items()))
        BUNDLE = STORE / 'mosei-colab-source.zip'
        BUNDLE.write_bytes(content)
        del uploaded, content
        """),
        code(EXTRACT),
        code("""
        source = STORE / 'csd'
        subprocess.run([sys.executable, 'scripts/download_csd.py', '--output', str(source)], check=True)
        shutil.copy2(PROJECT / 'data/csd/splits.json', source / 'splits.json')
        print('Download complete. Open 02_train_colab.ipynb in an A100 GPU runtime when available.')
        """),
        markdown("""
        Once the download completes, disconnect this CPU runtime. The files remain
        in `MyDrive/NLP_Assignment_3/csd`. If interrupted, rerun the cells; complete
        files are checked and partial downloads resume. Your source bundle also
        remains in that Drive folder for the training notebook.
        """),
    ]
    train = [
        markdown("""
        # CMU-MOSEI · 2. Train on Colab and export the video classifier

        Select **Runtime → Change runtime type → GPU → A100**, when available.
        Another CUDA GPU can run the same protocol; this notebook prints the GPU
        actually allocated. Complete notebook 1 first.

        Your 200 compute units are a runtime budget, not a fixed number of hours.
        Read the current units/hour rate in Colab's resource panel and enter it in
        the timing cell below. [Colab's FAQ](https://research.google.com/colaboratory/faq.html)
        explains that GPU availability and session limits vary.

        Native video tools remain on the local demo computer. Colab reads the
        already extracted COVAREP/OpenFace streams, regenerates frozen BERT words,
        and trains the fusion classifier. It preserves the exact local feature
        manifest so exported checkpoints can be verified before video inference.
        """),
        code("""
        from google.colab import drive
        from pathlib import Path
        drive.mount('/content/drive')
        STORE = Path('/content/drive/MyDrive/NLP_Assignment_3')
        BUNDLE = STORE / 'mosei-colab-source.zip'
        if not BUNDLE.is_file():
            raise FileNotFoundError('Complete notebook 1 first.')
        """),
        code(EXTRACT),
        code("""
        from huggingface_hub import snapshot_download
        subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], check=True)
        subprocess.run([sys.executable, '-c', 'import torch; assert torch.cuda.is_available(), "Select a GPU runtime"; print(torch.cuda.get_device_name(0))'], check=True)
        spec = tools['manifest']['text']
        model_path = Path('/content/bert-base-uncased')
        snapshot_download(spec['repo'], revision=spec['revision'], local_dir=model_path,
                          allow_patterns=list(spec['files_sha256']), max_workers=4)
        tools['paths']['text'] = str(model_path)
        (PROJECT / 'data/tools/video-tools.json').write_text(json.dumps(tools, indent=2))
        print('Pinned BERT snapshot ready. The feature manifest itself is unchanged.')
        """),
        code("""
        from mosei.colab_cache import persist_cache, restore_cache
        from mosei.data import load_data

        cache = Path('/content/prepared-csd')
        cache_archive = STORE / ('prepared-' + tools['fingerprint'][:16] + '.tar')
        if not restore_cache(cache, cache_archive):
            if shutil.disk_usage('/content').free < 55_000_000_000:
                raise RuntimeError('Allow at least 55 GB of free runtime disk for source and staging files.')
            source = Path('/content/csd')
            source.mkdir(exist_ok=True)
            for filename in tools['manifest']['files'].values():
                for name in (filename, filename + '.download.json'):
                    original, local = STORE / 'csd' / name, source / name
                    if not original.is_file():
                        raise FileNotFoundError(f'Complete notebook 1: {original}')
                    if not local.is_file() or local.stat().st_size != original.stat().st_size:
                        print('Copying', name, flush=True)
                        shutil.copy2(original, local)
            subprocess.run([sys.executable, '-m', 'mosei', 'prepare-csd', '--source', str(source),
                            '--output', str(cache), '--tools', 'data/tools/video-tools.json',
                            '--splits', 'data/csd/splits.json', '--device', 'cuda'], check=True)
        audit = load_data(cache).audit  # Verify every array and clip metadata file before reuse.
        if audit['feature_manifest'] != tools['manifest']:
            raise ValueError('Cache does not match this source bundle; use a fresh runtime.')
        if audit['smoke_max_clips_per_split'] is not None or 'smoke' in audit['folds_source'].get('purpose', '').lower():
            raise ValueError('Full experiments require the complete source folds, not an integration subset.')
        # Always finish publishing, even if an earlier session prepared the local cache.
        persist_cache(cache, cache_archive)
        RUNS = STORE / ('runs-csd-' + audit['fingerprint'][:16])
        print({s: audit['splits'][s]['eligible_count'] for s in ('train', 'valid', 'test')})
        print('Checkpoints will be saved to:', RUNS)
        """),
        markdown("""
        ## Measure before the full matrix

        This separate one-epoch run estimates throughput and does not enter the
        research results. Enter the **current** Colab rate below. The estimate
        covers training and validation, and excludes download, BERT preparation,
        held-out robustness evaluation, and idle time. Early stopping can shorten
        training. These are estimates, not a compute-unit spending cap.
        """),
        code("""
        UNITS_PER_HOUR = None  # Enter the rate displayed by your current Colab GPU session.
        AVAILABLE_UNITS = 200  # Replace with your remaining balance.
        INCLUDE_ABLATIONS = True  # 6 configurations x 3 seeds; False runs the 3 core configurations.
        timing = STORE / ('timing-' + audit['fingerprint'][:16])
        subprocess.run([sys.executable, '-m', 'mosei', 'train', '--data', str(cache),
                        '--config', 'configs/transformer.toml', '--output', str(timing),
                        '--epochs', '1', '--device', 'cuda', '--resume'], check=True)
        measured = json.loads((timing / 'run.json').read_text())
        epoch_seconds = json.loads((timing / 'history.json').read_text())[0]['seconds']
        max_runs = (6 if INCLUDE_ABLATIONS else 3) * 3
        estimated_hours = epoch_seconds * 30 * max_runs / 3600
        print(f'Measured Transformer epoch: {epoch_seconds:.1f} seconds on {measured["device_name"]}')
        print(f'Conservative training-only estimate using that pace for every model: {estimated_hours:.2f} hours')
        if UNITS_PER_HOUR is not None:
            print(f'Estimated training units: {estimated_hours * UNITS_PER_HOUR:.1f}; budget hours: {AVAILABLE_UNITS / UNITS_PER_HOUR:.2f}')
        """),
        code("""
        matrix = [sys.executable, 'scripts/run_experiments.py', '--data', str(cache),
                  '--output', str(RUNS), '--device', 'cuda', '--resume']
        if INCLUDE_ABLATIONS:
            matrix.append('--include-ablations')
        subprocess.run([*matrix, '--train-only'], check=True)
        print('Training finished. best.pt is selected only by validation macro F1 / loss.')
        """),
        markdown("""
        If the runtime disconnects, reconnect and rerun the setup/cache cells and
        the same matrix cell. Completed runs are reused; interrupted runs restore
        the last completed epoch, optimizer and RNG state. A partial epoch repeats.
        Changing the configuration or data requires a different output directory.
        Exact numerical reproducibility also depends on hardware/software.

        Freeze all modeling choices before running the held-out evaluation cell.
        """),
        code("""
        subprocess.run(matrix, check=True)
        print('Clean / matched-corruption results and paired source-video bootstrap intervals:', RUNS / 'report')
        """),
        code("""
        from google.colab import files
        candidates = []
        for directory in RUNS.glob('transformer-seed*'):
            result = json.loads((directory / 'run.json').read_text())
            candidates.append((result['best_validation']['macro_f1'], -result['best_validation']['loss'], directory))
        selected = max(candidates, key=lambda x: x[:2])[2]  # Never select by test scores.
        export = Path('/content/mosei-video-checkpoint.zip')
        with zipfile.ZipFile(export, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in ['best.pt', 'config.json', 'run.json', 'history.json', 'data_audit.json']:
                archive.write(selected / name, 'selected-run/' + name)
        shutil.copy2(export, STORE / export.name)
        print('Selected by validation:', selected.name)
        files.download(str(export))
        """),
        markdown("""
        Extract the export into the local project's `runs/colab/` directory, then run:

        ```bash
        .venv/bin/python -m mosei doctor --run runs/colab/selected-run --tools data/tools/video-tools.json
        .venv/bin/python -m mosei serve --run runs/colab/selected-run --tools data/tools/video-tools.json
        ```

        Open http://127.0.0.1:7860. Keep the matching local source/tool bundle;
        feature changes require new preparation/training. Download the `report`
        folder separately for the assignment. Disconnect and delete the Colab
        runtime after saving outputs to avoid consuming units while idle.
        """),
    ]
    folder = ROOT / "notebooks"
    folder.mkdir(exist_ok=True)
    for filename, cells, accelerator in [
        ("01_download_csd.ipynb", download, None),
        ("02_train_colab.ipynb", train, "GPU"),
    ]:
        metadata = {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "colab": {"name": filename, "provenance": []},
        }
        if accelerator:
            metadata["accelerator"] = accelerator
        notebook = {"cells": cells, "metadata": metadata, "nbformat": 4, "nbformat_minor": 5}
        for index, cell in enumerate(cells):
            cell["id"] = f"cell-{index:02d}"
        (folder / filename).write_text(json.dumps(notebook, indent=2) + "\n")
        print(folder / filename)
    subprocess.run([sys.executable, "-m", "ruff", "check", "--fix", str(folder)], check=True)
    subprocess.run([sys.executable, "-m", "ruff", "format", str(folder)], check=True)


if __name__ == "__main__":
    build()
