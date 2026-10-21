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
        ## Choose the matrix, then measure each configuration

        The default remains the original 3 models and 3 Transformer ablations,
        each with 3 seeds. Set `INCLUDE_HANLIN_MODELS = True` to add the integrated
        **ALMT** and **Graph_MFN** binary classifiers (6 extra runs). They use the
        same prepared data and training/evaluation protocol, not the separate
        legacy scripts or a missing `mosei_dataset.py`. You can instead edit
        `CONFIG_PATHS` to run only the configurations you want.

        A fresh one-epoch calibration for **each selected configuration** measures
        its own training/validation pace on this runtime; no model inherits the
        Transformer's timing. Calibrations consume compute but are separate from
        research results. Rerunning the timing cell recalibrates rather than
        reusing measurements from an earlier GPU session.

        Enter the **current** Colab rate below. The matrix estimate extrapolates
        each configuration to its epoch limit and excludes calibration, downloads,
        BERT preparation, held-out robustness evaluation, and idle time. Early
        stopping can shorten training, while later epochs and Drive writes may
        vary. This is neither an upper bound nor a compute-unit spending cap.
        Choose `EXPORT_MODEL` before inspecting test results; the selected seed
        is chosen only by validation scores within that model.
        """),
        code("""
        UNITS_PER_HOUR = None  # Enter the rate displayed by your current Colab GPU session.
        AVAILABLE_UNITS = 200  # Replace with your remaining balance.
        INCLUDE_ABLATIONS = True  # The original 3 Transformer ablations.
        INCLUDE_HANLIN_MODELS = False  # Explicit opt-in: add ALMT and Graph_MFN.
        SEEDS = [1111, 1112, 1113]
        EXPORT_MODEL = 'transformer'  # Or 'almt' / 'graph_mfn' if included in CONFIG_PATHS.
        CONFIG_PATHS = [f'configs/{name}.toml' for name in ('text', 'concat', 'transformer')]
        if INCLUDE_ABLATIONS:
            CONFIG_PATHS += [f'configs/{name}.toml' for name in ('no_modality', 'no_positions', 'mean_pooling')]
        if INCLUDE_HANLIN_MODELS:
            CONFIG_PATHS += ['configs/almt.toml', 'configs/graph_mfn.toml']
        print(f'Selected {len(CONFIG_PATHS)} configurations x {len(SEEDS)} seeds.')
        print(CONFIG_PATHS)
        """),
        code("""
        import time
        from mosei.config import load_config

        if not CONFIG_PATHS or not SEEDS:
            raise ValueError('Choose at least one configuration and one seed.')
        if UNITS_PER_HOUR is not None and UNITS_PER_HOUR <= 0:
            raise ValueError('UNITS_PER_HOUR must be positive or None.')
        # Each invocation gets a fresh directory, so old GPU timings cannot be reused.
        timing = STORE / ('timing-' + audit['fingerprint'][:16]) / str(time.time_ns())
        estimated_hours = 0.0
        for config_path in CONFIG_PATHS:
            config = load_config(Path(config_path))
            measured_run = timing / config.name
            subprocess.run([sys.executable, '-m', 'mosei', 'train', '--data', str(cache),
                            '--config', config_path, '--output', str(measured_run),
                            '--epochs', '1', '--seed', str(SEEDS[0]), '--device', 'cuda'], check=True)
            measured = json.loads((measured_run / 'run.json').read_text())
            epoch_seconds = json.loads((measured_run / 'history.json').read_text())[0]['seconds']
            model_hours = epoch_seconds * config.training.epochs * len(SEEDS) / 3600
            estimated_hours += model_hours
            print(f'{config.name}: {epoch_seconds:.1f} seconds/epoch on {measured["device_name"]}; '
                  f'{len(SEEDS)} seeds x {config.training.epochs} epochs: {model_hours:.2f} hours')
        print(f'Matrix training/validation extrapolation: {estimated_hours:.2f} hours (not an upper bound)')
        if UNITS_PER_HOUR is not None:
            print(f'Estimated matrix units: {estimated_hours * UNITS_PER_HOUR:.1f}; '
                  f'budget hours at the entered rate: {AVAILABLE_UNITS / UNITS_PER_HOUR:.2f}')
        """),
        code("""
        matrix = [sys.executable, 'scripts/run_experiments.py', '--data', str(cache),
                  '--output', str(RUNS), '--device', 'cuda', '--resume',
                  '--configs', *CONFIG_PATHS, '--seeds', *[str(seed) for seed in SEEDS]]
        subprocess.run([*matrix, '--train-only'], check=True)
        print('Training finished. best.pt is selected only by validation macro F1 / loss.')
        """),
        markdown("""
        If the runtime disconnects, reconnect and rerun the setup/cache cells and
        the same matrix cell. Completed runs are reused; interrupted runs restore
        the last completed epoch, optimizer and RNG state. A partial epoch repeats.
        Changing `CONFIG_PATHS`, seeds, configurations, or data requires a different
        `RUNS` output directory; resume deliberately rejects a changed matrix.
        Exact numerical reproducibility also depends on hardware/software.

        Freeze all modeling choices before running the held-out evaluation cell.
        """),
        code("""
        subprocess.run(matrix, check=True)
        print('Clean / matched-corruption results and paired source-video bootstrap intervals:', RUNS / 'report')
        """),
        code("""
        from dataclasses import replace
        from google.colab import files
        from mosei.config import config_from_dict, load_config

        manifest_path = RUNS / 'manifest.json'
        if not manifest_path.is_file():
            raise ValueError('Matrix manifest is missing. Run the complete frozen training matrix before export.')
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('status') not in {'trained', 'evaluated'}:
            raise ValueError('Matrix training is incomplete. Resume the frozen matrix before export.')
        frozen_configs = [config_from_dict(value) for value in manifest['configs']]
        frozen_seeds = manifest['seeds']
        if not frozen_configs or not frozen_seeds or len(set(frozen_seeds)) != len(frozen_seeds):
            raise ValueError('Matrix manifest must contain configurations and unique seeds.')
        if sum(config.name == EXPORT_MODEL for config in frozen_configs) != 1:
            raise ValueError(f'{EXPORT_MODEL} is not a unique experiment in the frozen matrix. '
                             'Choose an included EXPORT_MODEL, or train a new matrix with different CONFIG_PATHS.')
        current_configs = []
        for config_path in CONFIG_PATHS:
            config = load_config(Path(config_path))
            current_configs.append(replace(config, training=replace(config.training, device='cuda')))
        if current_configs != frozen_configs or SEEDS != frozen_seeds:
            raise ValueError('Current CONFIG_PATHS/configuration contents or SEEDS differ from the frozen matrix. '
                             'Restore those choices, or train a new RUNS directory.')
        if audit['fingerprint'] != manifest['data_fingerprint']:
            raise ValueError('Current data differs from the frozen matrix. Restore the matching prepared cache.')

        # run.json is the trainer's completion marker; validate every frozen run,
        # not just whichever seed directories happen to exist for the export model.
        export_files = ['best.pt', 'config.json', 'run.json', 'history.json', 'data_audit.json']
        candidates = []
        for config in frozen_configs:
            for seed in frozen_seeds:
                directory = RUNS / f'{config.name}-seed{seed}'
                missing = [name for name in export_files if not (directory / name).is_file()]
                if missing:
                    raise ValueError(f'Incomplete frozen run {directory.name}: missing {", ".join(missing)}. '
                                     'Resume the matrix before export.')
                expected = replace(config, training=replace(config.training, seed=seed))
                try:
                    result = json.loads((directory / 'run.json').read_text())
                    saved_config = json.loads((directory / 'config.json').read_text())
                    history = json.loads((directory / 'history.json').read_text())
                    saved_audit = json.loads((directory / 'data_audit.json').read_text())
                    if (config_from_dict(result['config']) != expected
                            or config_from_dict(saved_config) != expected
                            or result['data_fingerprint'] != manifest['data_fingerprint']
                            or saved_audit['fingerprint'] != manifest['data_fingerprint']):
                        raise ValueError('configuration, seed, or data does not match the frozen matrix')
                    if (not 1 <= result['epochs_completed'] <= expected.training.epochs
                            or result['epochs_completed'] != len(history)
                            or not 1 <= result['best_epoch'] <= result['epochs_completed']):
                        raise ValueError('completed epoch metadata and history do not match')
                    score = (result['best_validation']['macro_f1'], -result['best_validation']['loss'])
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    raise ValueError(f'Invalid completed run {directory.name}: {exc}') from exc
                if config.name == EXPORT_MODEL:
                    candidates.append((*score, directory))
        selected = max(candidates, key=lambda x: x[:2])[2]  # Never select by test scores.
        export = Path('/content/mosei-video-checkpoint.zip')
        with zipfile.ZipFile(export, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for name in export_files:
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
