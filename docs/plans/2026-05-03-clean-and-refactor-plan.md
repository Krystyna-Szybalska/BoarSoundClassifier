# Plan: Clean and refactor BoarSoundClassifier into a testable package

**Status**: ready to execute (Phase 1 not yet started). All decisions locked.
**Last updated**: 2026-05-03

## Context

The repo is a research-style binary audio classifier (boar / not-boar from 4-second clips) that has been rewritten once. Both generations still sit on disk: a broken legacy Keras pipeline ([Model/main.py](../../Model/main.py), [Model/utils.py](../../Model/utils.py)) and the live PyTorch pipeline ([Model/pytorch_model_changes.ipynb](../../Model/pytorch_model_changes.ipynb)). Five training-snapshot notebooks (`_best`, `_saved`, `_good_weight_decay_1st`, `_nice_training`, plus the root `pytorch_model.ipynb`) clutter the tree, an empty stub script exists ([Model/SoundData/RenameFiles.py](../../Model/SoundData/RenameFiles.py)), several pipeline scripts hardcode stale `C:\Users\Krysia\Desktop\...` paths, and there is no `pyproject.toml`, `requirements.txt`, or `tests/` directory. `Model/main.py` has three demonstrable bugs (line 94 references undefined `n_timesteps`; line 96 uses `Conv2D` while only `Conv1D` is imported; line 127 has bad indentation) that prove it has never executed.

**Goal**: produce a small, testable `boar_sound_classifier` package containing the live pipeline; archive dead code (do not hard-delete unless the file is provably broken); preserve every architectural invariant CLAUDE.md calls out (split-on-`OriginalFileName`, the `(1, 128, 173)` mel input contract, the deliberate `stop` guard-rail cells, Polish + English in strings/comments).

**Outcome**: a future contributor can `pip install -e .` the package, run `pytest`, train via a thin `notebooks/train.ipynb`, run inference via `notebooks/infer.ipynb`, and inspect via `notebooks/visualize.ipynb` — none of which duplicate code; all import from `boar_sound_classifier`.

## Hard constraints (do not violate)

- **Do not commit or push** anything ([CLAUDE.md](../../CLAUDE.md)). Every step ends in `git add` at most; the user pushes.
- **Do not delete data files**: never remove `*.csv`, `*.pth`, `*.pkl`, `*.wav`, `*.mp3`, `*.ogg`, `*.m4a`, `*.opus`, `*.txt`. These are gitignored locally but represent hours of curation.
- **Do not split data on `FileName`** — only on `OriginalFileName`. The leakage rule is the project's most important invariant.
- **Do not change the mel input contract** (`SAMPLE_RATE=22050`, `DURATION=4`, `N_FFT=2048`, `HOP_LENGTH=512`, `N_MELS=128`, `INPUT_MEL_WIDTH=173`). Old `.pth` checkpoints will silently fail to load if any of these shift.
- **Preserve mixed Polish + English** strings and comments verbatim during extraction. Do not translate.
- **Preserve the `stop` NameError guard cells** in any notebook that retains them. They are intentional Run-All breakers, not bugs.
- **Deletions require user approval before each phase that deletes** — confirm at every gate. Auto mode does not waive this.

## Locked decisions (already agreed with user)

- Snapshots get **archived**, not hard-deleted. Move to `archive/`.
- `Model/test_data/` (the boar-oink + cat-meow WAVs) gets **promoted to `tests/fixtures/`**.
- The repo will keep three orchestrator notebooks under `notebooks/`: **train**, **infer**, and (optionally) **visualize** — all thin, all importing from `boar_sound_classifier`, none duplicating code.
- Pipeline notebooks (codec augmentation, dataset split) become thin orchestrators that import from `boar_sound_classifier.pipeline.*`.
- CI: **pre-commit hooks (local) + GitHub Actions (cloud)**. GH Actions runs on `ubuntu-latest` only, against **Python 3.11** only.

## Phased approach

The plan is six phases. Each one ends in a stable, runnable state, so the user can stop the refactor at any phase boundary and still have a working repo.

### Phase 1 — Triage: archive snapshots, delete provably dead code

Move (don't delete) the five training-snapshot notebooks into a new `archive/` directory at repo root:

- [Model/pytorch_model_changes_best.ipynb](../../Model/pytorch_model_changes_best.ipynb) → `archive/`
- [Model/pytorch_model_changes_saved.ipynb](../../Model/pytorch_model_changes_saved.ipynb) → `archive/`
- [Model/pytorch_model_changes_good_weight_decay_1st.ipynb](../../Model/pytorch_model_changes_good_weight_decay_1st.ipynb) → `archive/`
- [Model/pytorch_model_changes_nice_training.ipynb](../../Model/pytorch_model_changes_nice_training.ipynb) → `archive/`
- [pytorch_model.ipynb](../../pytorch_model.ipynb) (the MFCC baseline) → `archive/`

Move-not-delete preserves training history (loss curves, accuracy timelines) the user has invested time producing. If the user later confirms none of these are referenced, a follow-up `git rm` is trivial.

Delete (with explicit confirmation) the provably broken / empty files:

- [Model/main.py](../../Model/main.py) — three confirmed bugs, never callable, references `Conv2D` not imported.
- [Model/utils.py](../../Model/utils.py) — incomplete stub with wrong column names (`fold`, `slice_file_name`, `class`) that don't exist in any current CSV.
- [Model/SoundData/RenameFiles.py](../../Model/SoundData/RenameFiles.py) — 382 bytes, only placeholder variable assignments.

Phase 1 leaves the live notebook untouched and only moves/deletes things demonstrably dead.

### Phase 2 — Package skeleton (no code extraction yet)

Create a `src/`-layout Python package and supporting scaffolding. Nothing deleted, notebook still runs:

```
BoarSoundClassifier/
├── pyproject.toml          # NEW — modern packaging, declares deps + entry points
├── requirements.txt        # NEW — pinned versions for reproducibility
├── requirements-dev.txt    # NEW — pytest, ruff, etc.
├── README.md               # EXPAND — currently 21 bytes
├── src/
│   └── boar_sound_classifier/
│       ├── __init__.py     # NEW — exports public API
│       └── config.py       # NEW — placeholder, populated in Phase 3
├── tests/
│   ├── __init__.py
│   ├── conftest.py         # NEW — synthetic 4 s WAV fixture
│   └── fixtures/           # NEW — promoted from Model/test_data/
│       ├── boar_oink.wav   # MOVED from Model/test_data/
│       └── cat_meow.wav    # MOVED from Model/test_data/
└── docs/
    └── plans/              # already exists (this file lives here)
```

Pin in `requirements.txt` (deduced from notebook imports + CLAUDE.md): `torch`, `torchaudio`, `librosa`, `audiomentations`, `albumentations`, `soundfile`, `pydub`, `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`, `tqdm`. Pin to whatever the user's working notebook environment currently has — read versions out of the live env in this phase, do not guess.

Add `[tool.pytest.ini_options]` and `[tool.ruff]` blocks to `pyproject.toml`. Set `pythonpath = ["src"]` so tests find the package without an install.

CI scaffolding (`.github/workflows/`, `.pre-commit-config.yaml`) is added in Phase 5, not Phase 2 — Phase 2 is just the package skeleton and dependency pinning.

### Phase 3 — Extract the live pipeline into modules

Map each cell of [Model/pytorch_model_changes.ipynb](../../Model/pytorch_model_changes.ipynb) into a module. Boundaries are dictated by the cells themselves (already verified by grep):

| Module | Source (notebook line) | Contents |
|---|---|---|
| `boar_sound_classifier/config.py` | cells around lines 122-128, 772-777, 998 | `SAMPLE_RATE=22050`, `DURATION=4`, `N_FFT=2048`, `HOP_LENGTH=512`, `N_MELS=128`, `INPUT_MEL_WIDTH=173`, `NUM_CLASSES=2`, `calculate_num_frames(...)` |
| `boar_sound_classifier/augment.py` | cells at lines 48 and 86 | `apply_augmentations(audio, sr)`, `augment_spectrogram(spec)` |
| `boar_sound_classifier/data.py` | cell at line 136 | `class WildBoarDataset(Dataset)` |
| `boar_sound_classifier/model.py` | cell at line 779 | `class AudioClassifier(nn.Module)` |
| `boar_sound_classifier/train.py` | cell at line 863 | `train_model(...)`, plus `evaluate(...)` if it exists in a later cell, plus the `combine_histories(...)` helper |
| `boar_sound_classifier/infer.py` | NEW — extracted from "predict on single file" cells | `load_model(checkpoint_path)`, `predict(model, audio_path) -> (label, prob)` |

The notebook's symbol-defining cells get replaced with `from boar_sound_classifier.X import Y`. The notebook becomes an orchestrator: load checkpoint → call `train_model` → plot. The deliberate `stop` NameError cells stay where they are.

The canonical training notebook moves to `notebooks/train.ipynb`. Two more orchestrator notebooks join it:

- `notebooks/infer.ipynb` — uses `boar_sound_classifier.infer.load_model` + `predict`. Runs against a single WAV (e.g., one of the `tests/fixtures/` files) and prints the predicted label + probability.
- `notebooks/visualize.ipynb` (optional) — uses helpers in `boar_sound_classifier.viz` (NEW small module): plot mel-spec for a clip, plot training history from a `.pkl`, plot confusion matrix on the test set.

Pipeline scripts get their own subpackage:

```
src/boar_sound_classifier/pipeline/
├── __init__.py
├── chunk_audio.py        # from SplitWavAudioIn4SekChunks.py + split_files.py
├── pad_audio.py          # from pad_audio.py — strip hardcoded paths, take CLI args
├── codec_augment.py      # extracted from compression_augmentation.ipynb
├── quality_filter.py     # from RemoveLowQuality.py + SplitCSV.py + change_label_in_csv.py
└── split_dataset.py      # extracted from split_data_into_datasets.ipynb
```

Each pipeline module exposes a `main(args)` and an `if __name__ == "__main__"` block, with `argparse` replacing every `C:\Users\Krysia\Desktop\...` literal. The two run-once data-pipeline notebooks ([Model/compression_augmentation.ipynb](../../Model/compression_augmentation.ipynb), [Model/split_data_into_datasets.ipynb](../../Model/split_data_into_datasets.ipynb)) get reduced to thin notebooks under `notebooks/` that import and call those modules. [Model/add_old_q0_files.ipynb](../../Model/add_old_q0_files.ipynb) gets reviewed; if it's a one-off maintenance script, it goes to `archive/`.

After Phase 3 the notebook MUST still produce identical mel-spec outputs given identical seeds. Verification step at the end of Phase 3: run a single training epoch on a 4-sample subset before and after extraction, compare loss to 6 decimal places. If they diverge, extraction broke something.

### Phase 4 — Tests

Tests live in `tests/`, use `pytest`, and target the boundaries that matter:

- `tests/test_config.py` — `calculate_num_frames(22050, 4, 2048, 512) == 173`. Locks in the input contract.
- `tests/test_model.py` — `AudioClassifier()(torch.randn(2, 1, 128, 173)).shape == (2, 2)`. Locks the FC layer wiring; if anyone changes mel dims, this fails loudly.
- `tests/test_data.py` — load `tests/fixtures/boar_oink.wav`, instantiate `WildBoarDataset` against a one-row in-memory DataFrame, assert returned tensor shape `(1, 128, 173)` and label dtype `torch.long`.
- `tests/test_augment.py` — `augment=False` path is deterministic (same input → same output). `augment=True` path produces a different output.
- `tests/test_split_leakage.py` — generate a fake metadata DataFrame with `OriginalFileName` lineage, run `split_dataset.split(...)`, assert that no `OriginalFileName` appears in more than one of train/val/test. **This is the most important test** — it codifies the architectural invariant.
- `tests/test_pipeline_chunk.py` — feed `tests/fixtures/cat_meow.wav` (or a synthetic 9 s WAV from `conftest`) to `chunk_audio.chunk(...)`, assert the expected number of 4 s chunks are produced.
- `tests/test_infer.py` — load a small checkpoint (or mock the model with a fixed-output stub), call `predict(model, fixtures/boar_oink.wav)`, assert it returns `(label_int, prob_float)` with sensible types.

Fixtures: the two existing test WAVs ([Model/test_data/](../../Model/test_data/)) move to `tests/fixtures/`. `conftest.py` additionally generates a synthetic boar-shaped (low-frequency growl approximation via sine sweep + noise) WAV at session scope when more variety is needed.

### Phase 5 — Developer ergonomics + CI

- `Makefile`: `make test` (pytest), `make lint` (ruff check), `make format` (ruff format), `make data-prep` (chained pipeline modules), `make train` (jupyter run on `notebooks/train.ipynb`).
- `.pre-commit-config.yaml` with:
  - `ruff` (lint + format) — fast feedback under 1s.
  - `nbstripout` — strips notebook outputs on commit so diffs don't bloat with cell outputs.
  - `check-yaml`, `end-of-file-fixer`, `trailing-whitespace` (basic stdlib hooks).
- `.github/workflows/tests.yml`: GitHub Actions, triggered on `push` and `pull_request`. Runs on `ubuntu-latest` against **Python 3.11 only**. Steps: checkout → setup-python → cache pip → `pip install -e ".[dev]"` → `ruff check` → `pytest -v`. Cache key on `requirements*.txt` for speed. No coverage reporting in v1 (can be added later).
- `README.md` rewrite: install instructions, how to run the pipeline end-to-end, where data lives, how to run tests, link to [CLAUDE.md](../../CLAUDE.md) and this plan for deeper context.
- Update `.gitignore` if needed — current one is good, but verify `archive/` is not accidentally excluded.

### Phase 6 — Validation gate

Before declaring done, run end-to-end:

1. `pip install -e .[dev]` from a fresh venv.
2. `pytest -v` — all tests green.
3. `python -m boar_sound_classifier.pipeline.chunk_audio --help` — CLI works.
4. Open `notebooks/train.ipynb`, run cells 0 through "load checkpoint", confirm a saved `.pth` still loads cleanly into the new `AudioClassifier`.
5. Open `notebooks/infer.ipynb`, run all cells, confirm prediction on a fixture WAV.
6. Train one epoch on a small subset, compare final loss to a value recorded before refactor. Must match within numerical noise.
7. `git status` — only intended files touched. No accidentally committed `.csv` / `.pth` / `.wav`.

Any of 4-6 failing means the extraction is wrong; revert the extraction commit and re-do.

## Files at a glance

**Critical files this plan modifies or moves:**

- [Model/pytorch_model_changes.ipynb](../../Model/pytorch_model_changes.ipynb) → `notebooks/train.ipynb` (slimmed to orchestrator)
- [Model/compression_augmentation.ipynb](../../Model/compression_augmentation.ipynb) → `notebooks/codec_augment.ipynb` (slimmed)
- [Model/split_data_into_datasets.ipynb](../../Model/split_data_into_datasets.ipynb) → `notebooks/split_dataset.ipynb` (slimmed)
- All `Model/SoundData/*.py` → `src/boar_sound_classifier/pipeline/*.py` (with hardcoded paths replaced by argparse)
- [Model/test_data/](../../Model/test_data/) WAVs → `tests/fixtures/`
- [Model/main.py](../../Model/main.py), [Model/utils.py](../../Model/utils.py), [Model/SoundData/RenameFiles.py](../../Model/SoundData/RenameFiles.py) → deleted (with confirmation)
- All `pytorch_model_changes_*.ipynb` snapshots + root `pytorch_model.ipynb` → `archive/`

**Files this plan creates:**

- `pyproject.toml`, `requirements.txt`, `requirements-dev.txt`, `Makefile`, `.pre-commit-config.yaml`
- `src/boar_sound_classifier/{__init__,config,augment,data,model,train,infer,viz}.py`
- `src/boar_sound_classifier/pipeline/{__init__,chunk_audio,pad_audio,codec_augment,quality_filter,split_dataset}.py`
- `tests/{conftest,test_config,test_model,test_data,test_augment,test_split_leakage,test_pipeline_chunk,test_infer}.py`
- `tests/fixtures/` (populated from `Model/test_data/`)
- `notebooks/{train,infer,visualize,codec_augment,split_dataset}.ipynb`
- `archive/` (catch-all for snapshots)

**Files explicitly left alone:**

- [CLAUDE.md](../../CLAUDE.md) (re-optimized in this same iteration; will need a path-update pass once the refactor lands)
- [Model/SoundData/RawData/](../../Model/SoundData/RawData/), `Model/SoundData/PreparedData/`, `Model/SoundData/PreparedDataRecompressed/` (data, untouchable)
- `Model/SoundData/Quality_0_Files/` (data, untouchable)
- [Model/SoundData/DisplaySound.ipynb](../../Model/SoundData/DisplaySound.ipynb), [Model/SoundData/Playsound.ipynb](../../Model/SoundData/Playsound.ipynb) (utilities; consider folding into `notebooks/visualize.ipynb` or leave as is)
- All `*.pth`, `*.pkl`, `*.csv`, `*.txt`, `*.wav` (gitignored data)
- `venv/` (stale; user replaces it themselves at Phase 2)

## Verification (end-to-end test of the refactor)

After Phase 6 the user should be able to do, from a clean shell:

```bash
python -m venv .venv && .\.venv\Scripts\activate
pip install -e ".[dev]"
pytest -v                              # all green
python -m boar_sound_classifier.pipeline.split_dataset --metadata Model\SoundData\Metadata_All_Normalized.csv --out-dir Model\
jupyter notebook notebooks\train.ipynb # runs end-to-end, loads val_acc_0.9502.pth
jupyter notebook notebooks\infer.ipynb # predicts on tests/fixtures/boar_oink.wav
```

If all five commands work and `pytest` covers the leakage rule + input contract + model wiring, the refactor is done.

## Open decisions

None — all resolved. Ready to execute Phase 1.
