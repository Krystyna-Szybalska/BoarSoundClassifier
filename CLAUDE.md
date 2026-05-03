# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Hard rules (non-negotiable)

- **DO NOT COMMIT OR PUSH** any code. The user is the only one who pushes to the remote.
- **DO NOT DELETE DATA FILES.** Anything matching `*.csv`, `*.pth`, `*.pkl`, `*.txt`, `*.wav`, `*.mp3`, `*.ogg`, `*.m4a`, `*.opus` is curated input, model weights, or training history — hours of work. These are gitignored locally; do not `git add` them either.

## What this project is

Binary audio classifier that decides whether a 4-second clip contains a wild boar (`Label = 1`) or not (`Label = 0`). Trained on a curated mix of BBC, Freesound, Fesliyanstudios, etc. recordings. Cornell and xeno-canto are present in raw data but deliberately excluded during dataset assembly (`EXCLUDE_KEYWORDS` in [Model/split_data_into_datasets.ipynb](Model/split_data_into_datasets.ipynb)).

## Active refactor — read the plan first

A clean-and-restructure refactor is in flight. Before making structural changes, read [docs/plans/2026-05-03-clean-and-refactor-plan.md](docs/plans/2026-05-03-clean-and-refactor-plan.md) — it covers the target package layout (`src/boar_sound_classifier/`), the orchestrator notebooks (`notebooks/{train,infer,visualize}.ipynb`), what gets archived vs. deleted, and the verification gates.

Until the refactor lands, the live training pipeline is [Model/pytorch_model_changes.ipynb](Model/pytorch_model_changes.ipynb). The legacy Keras files ([Model/main.py](Model/main.py), [Model/utils.py](Model/utils.py)), the empty stub [Model/SoundData/RenameFiles.py](Model/SoundData/RenameFiles.py), and the snapshot notebooks (`*_best`, `*_saved`, `*_good_weight_decay_1st`, `*_nice_training`, root `pytorch_model.ipynb`) are slated for archive or deletion — do not extend them.

## Architectural invariants (never violate)

### The leakage rule (most important)

Splits **must** be made on the *original recording*, before chunking and before codec re-encoding. A single 30-second BBC clip becomes ~7 segments × ~10 codec variants ≈ 70 samples; if any of those 70 land in train and the rest in val, validation accuracy is meaningless. The chain `OriginalFileName` (recording-level) → `FileName` (segment-level, e.g. `NHU05040109_0` or `NHU05040109_0_mp3_64k`) preserves this lineage. The `*_original_names.csv` files in [Model/](Model/) are the only safe way to filter `Metadata_All_Normalized.csv` into train/val/test.

If you build a new metadata CSV or split, preserve `OriginalFileName` and split on it.

### The mel input contract

```
SAMPLE_RATE = 22050    DURATION = 4 s
N_FFT = 2048           HOP_LENGTH = 512    N_MELS = 128
→ Mel-spectrogram tensor shape (1, 128, 173)
```

`INPUT_MEL_WIDTH = 173` is baked into the FC layer's input size in `AudioClassifier`. Changing `DURATION`, `HOP_LENGTH`, `N_FFT`, or `SAMPLE_RATE` requires recomputing via `calculate_num_frames(...)`, updating `INPUT_MEL_WIDTH`, and **retraining from scratch** — old `.pth` checkpoints will not load.

`NUM_CLASSES = 2`. The head is `nn.CrossEntropyLoss` over class indices, **not** one-hot. The dead Keras files use one-hot — do not carry that pattern into new code.

### The `stop` guard cells

Cells containing a bare `stop` (deliberate `NameError`) appear repeatedly in `pytorch_model_changes.ipynb`. They are intentional Run-All breakers. Do not "fix" them.

### Mixed Polish + English

Variable and column names are English (`train_df`, `OriginalFileName`). Log messages, docstrings, and many comments are Polish (`treningowy`, `walidacyjny`, `Wczytywanie`, `Błąd`). Preserve this when editing — do not translate wholesale.

## The data pipeline (multi-stage, run-once)

Raw audio → trainable dataset is a chain. Each metadata CSV is produced by a different stage:

1. **Source files** in [Model/SoundData/RawData/](Model/SoundData/RawData/) under per-source subfolders (`BBC/`, `Freesound/`, `Fesliyanstudios/`, ...).
2. **Segment to 4-second chunks**: [Model/SoundData/split_files.py](Model/SoundData/split_files.py) + [Model/SoundData/SplitWavAudioIn4SekChunks.py](Model/SoundData/SplitWavAudioIn4SekChunks.py) → `<basename>_<i>.wav` in `Model/SoundData/PreparedData/`.
3. **Pad short tails** to exactly 4 s: [Model/SoundData/pad_audio.py](Model/SoundData/pad_audio.py). Reads from `To pad.txt` / `To pad and 0.txt`.
4. **Manual quality annotation** in `Model/SoundData/Metadata.csv` (semicolon-separated). `Quality = 0` = unusable, `Quality = 1` = keep. [Model/SoundData/RemoveLowQuality.py](Model/SoundData/RemoveLowQuality.py) physically moves `Quality=0` files into `Quality_0_Files/`.
5. **Codec augmentation** ([Model/compression_augmentation.ipynb](Model/compression_augmentation.ipynb)): re-encodes each kept WAV at lossy bitrates (`mp3` 128/64/32k, `ogg` qa3/qa0/qa6, `m4a` 96/64k, `opus` 64/32k) into `PreparedDataRecompressed/`, emitting `Metadata_All.csv`. A second cell normalizes `FilePath` to absolute paths → `Metadata_All_Normalized.csv`.
6. **Dataset split** ([Model/split_data_into_datasets.ipynb](Model/split_data_into_datasets.ipynb)) on **original recording names** → `train_original_names.csv`, `val_original_names.csv`, `test_original_names.csv` in [Model/](Model/). The training notebook joins these against `Metadata_All_Normalized.csv` via the `OriginalFileName` column.

## Augmentations

Two layers, both train-only (val/test datasets use `augment=False`):

- **Audio-level**, before mel computation: `AddGaussianNoise`, `PitchShift ±2 semitones`, `TimeStretch 0.8–1.2x`, `Gain ±7 dB`, each at `p=0.5`.
- **Spectrogram-level**: `ShiftScaleRotate` and `CoarseDropout` (frequency/time masking) — partially commented in `apply_augmentations`; check `augment_spectrogram` before assuming it's active.
- **Codec augmentation** is *baked into the dataset* (step 5 above), not applied per-batch. So even with `augment=False`, val/test sets contain codec-degraded samples — intentional, part of the project's robustness story.

## Training and checkpoints

- Driver: `train_model(...)` in `pytorch_model_changes.ipynb`. `Adam(lr=5e-5, weight_decay=1e-3)`, `ReduceLROnPlateau(factor=0.1, patience=1, min_lr=1e-6)` on val loss.
- Checkpoint files in [Model/](Model/) are named after the validation accuracy they achieved: `val_acc_0.8754.pth`, `val_acc_0.9502.pth`. **The filename IS the metadata** — there is no separate manifest. The training loop also writes `best_audio_classifier_.pth` (live "best so far").
- Resumed training: `model.load_state_dict(...)` precedes a `train_model(...)` call; history is pickled to `training_history<from>-<to>.pkl`. Multiple histories stitch via `combine_histories(...)` for a single plot.

## Environment

- The committed `venv/` is **stale** (TF 2.13 / Keras 2.13, no PyTorch). Current notebooks need a separate environment with `torch torchaudio librosa audiomentations albumentations soundfile pydub seaborn tqdm matplotlib pandas scikit-learn`. A pinned `requirements.txt` is part of the planned refactor — until then, install ad hoc.
- `pydub` requires `ffmpeg` on PATH (codec augmentation step uses it).
- Many CSVs and legacy scripts contain absolute paths under `C:\Users\Krysia\Desktop\...`. The repo lives at `C:\Users\Krysia\Documents\Moje dokumenty\Projekty\Machine Learning\BoarSoundClassifier\`. These paths are **historical** — `Metadata_All_Normalized.csv`'s `FilePath` column is rebuilt by the normalization cell in `compression_augmentation.ipynb` (re-run that cell rather than hand-editing).

## Common operations

- **Run a training cell after a fresh kernel**: launch Jupyter with CWD = `Model/` (notebook-relative paths like `'SoundData/PreparedData'` resolve from there). Cells 0–9 of `pytorch_model_changes.ipynb` define imports/dataset/model; cell 12 instantiates model+optimizer; cell 14 loads a checkpoint and runs `train_model(...)`. To skip from-scratch training, run 0–12 then jump to cell 24+ to load `val_acc_0.9502.pth` and evaluate.
- **Re-build the codec-augmented dataset**: in `compression_augmentation.ipynb`, set paths in cell 5, run `perform_format_augmentation(...)` (cell 8), then re-run the `Metadata_All_Normalized.csv` normalization cell.
- **Re-build the train/val/test splits**: in `split_data_into_datasets.ipynb`, run cells 0–9. Output is the three `*_original_names.csv` files in [Model/](Model/).
