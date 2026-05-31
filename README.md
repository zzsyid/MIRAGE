The train code will upload after acceptance
# MIRAGE Eval-Only Release

This directory is a clean **evaluation-only** repository scaffold for the MIRAGE/Synchformer-style dual-evidence detector.
It is designed to help you publish inference and benchmark reproduction code **without releasing training code**.

## Public-release boundary
- The runtime model code is provided in `scripts/dual_evidence_runtime.py`.

## What is included
- Dual-evidence model definition needed for inference only
- Evaluation scorers and summary scripts
- LAV-DF and FakeAVCeleb manifest preparation helpers
- Evaluation wrappers for LAV-DF, FakeAVCeleb, and generic manifest-based scoring

## Directory layout
- `scripts/`: evaluation scripts only
- `checkpoints/`: put released checkpoints here
- `manifests/`: optional manifests for generic evaluation
- `outputs/`: generated scores and summaries

## Required external assets
You still need to provide these paths yourself:
- `CKPT`: released MIRAGE checkpoint, e.g. `checkpoints/mirage_best.pt`
- `PHONEME_CACHE_CKPT`: AV-HuBERT checkpoint used for phoneme cache extraction
- `AVHUBERT_ROOT`: AV-HuBERT code root
- dataset roots / metadata paths for the target benchmark

## Third-party runtime dependencies
This eval-only repo assumes the following are available:
- `ffmpeg` in `PATH`
- `speechbrain` for the speaker encoder
- `insightface` for the face encoder
- an AV-HuBERT checkout for phoneme-cache extraction

## Quick start
### 0. AVDF
Prepare an AVDF diagnostic manifest first (for public release, we assume you provide `manifests/avdf_manifest.csv`).
```bash
cd /data1/zzsyid/MIRAGE/eval_only
CKPT=$PWD/checkpoints/mirage_best.pt MANIFEST_CSV=$PWD/manifests/avdf_manifest.csv PHONEME_CACHE_CKPT=/path/to/avhubert/self_large_vox_433h.pt AVHUBERT_ROOT=/path/to/av_hubert bash scripts/run_dual_evidence_avdf_eval.sh
```

### 1. LAV-DF
```bash
cd /data1/zzsyid/MIRAGE/eval_only
CKPT=$PWD/checkpoints/mirage_best.pt LAVDF_ROOT=/path/to/LAV-DF METADATA_JSON=/path/to/LAV-DF/metadata.min.json PHONEME_CACHE_CKPT=/path/to/avhubert/self_large_vox_433h.pt AVHUBERT_ROOT=/path/to/av_hubert bash scripts/run_dual_evidence_lavdf_eval.sh
```

### 2. FakeAVCeleb
```bash
cd /data1/zzsyid/MIRAGE/eval_only
CKPT=$PWD/checkpoints/mirage_best.pt FAVC_ROOT=/path/to/FakeAVCeleb_v1.2 SPLIT_CSV=/path/to/test_split.csv PHONEME_CACHE_CKPT=/path/to/avhubert/self_large_vox_433h.pt AVHUBERT_ROOT=/path/to/av_hubert bash scripts/run_dual_evidence_fakeavceleb_eval.sh
```

### 3. Generic manifest evaluation
Your manifest should contain at least `id,group,path,label` (or `expected_anomaly`).
```bash
cd /data1/zzsyid/MIRAGE/eval_only
CKPT=$PWD/checkpoints/mirage_best.pt MANIFEST_CSV=$PWD/manifests/eval_manifest.csv THRESHOLD_GROUP=real_test REQUIRE_GROUPS=real_test,fake_audio,fake_video,fake_av bash scripts/run_manifest_eval.sh
```
