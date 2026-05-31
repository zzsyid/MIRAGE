The code will upload after acceptance
# MIRAGE
Official code release for **MIRAGE**, a multi-evidence audio-visual forgery detector designed for hard fake-audio and cross-modal mismatch detection.
This repository contains the **final training and evaluation pipeline** used for the released MIRAGE model, together with the cleaned protocol construction utilities and the final checkpoint.

## Highlights
- Multi-evidence AV forgery detection with three evidence sources:
  - **Artifact evidence** for local visual forgery traces
  - **Inconsistency evidence** for cross-modal mismatch detection
  - **Identity evidence** for global face-voice consistency
- Fine-grained phoneme-level inconsistency modeling with scale-adaptive routing
- Training protocol code for the final pseudo-audio-swap intervention setting
- Released final checkpoint:
  - `checkpoints/mirage_best.pt`

## Repository Layout
- `scripts/`: training, evaluation, and protocol preparation entry points
- `model/`, `dataset/`, `utils/`: runtime dependencies for MIRAGE
- `metadata/favc/test_split.csv`: local FakeAVCeleb split metadata used by evaluation
- `third_party/av_hubert/`: AVHuBERT dependency used for phoneme-sync feature precomputation
- `checkpoints/mirage_best.pt`: released full-model checkpoint

## What Is Included
This release includes the code for the **final training protocol** used by MIRAGE, including:
- pseudo pair preparation: `scripts/prepare_pseudo_audio_swap_v3_pairs.py`
- XTTS generation wrapper: `scripts/run_generate_pseudo_audio_swap_v3_xtts.sh`
- XTTS generation core: `scripts/generate_pseudo_audio_swap_v3_xtts.py`
- pseudo dataset construction: `scripts/build_pseudo_audio_swap_v3_dataset.py`
- intervention manifest merge: `scripts/merge_intervention_manifests.py`
- identity cache precomputation: `scripts/precompute_global_identity_cache.py`
- phoneme cache precomputation: `scripts/precompute_phoneme_sync_cache.py`
- test identity cache precomputation: `scripts/precompute_test_identity_cache.py`

You will need to prepare the datasets locally and point the scripts to your own dataset roots.

## Environment
A starting environment file is provided:
- `conda_env.yml`

Create and activate an environment, for example:

```bash
conda env create -f conda_env.yml
conda activate mirage
```

Depending on your machine and CUDA stack, you may still need to adjust PyTorch / torchvision / torchaudio versions.

## Expected Data Layout
By default, the cleaned scripts assume repo-relative paths where possible. If your datasets live elsewhere, pass them through environment variables.

Recommended dataset roots:
- `AVDF_ROOT=/path/to/AV-Deepfake1M`
- `FAVC_ROOT=/path/to/FakeAVCeleb_v1.2`
- `OUTPUT_ROOT=/path/to/output_dir`

The released FakeAVCeleb split file is already included at:
- `metadata/favc/test_split.csv`
