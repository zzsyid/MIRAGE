import csv
import json
import logging
import os
import random
from pathlib import Path

import torch

from dataset.dataset_utils import get_fixed_offsets, get_video_and_audio, subsample_dataset


class AVDeepfakeSync(torch.utils.data.Dataset):

    def __init__(
        self,
        split,
        vids_dir,
        transforms=None,
        splits_path="./data",
        seed=1337,
        load_fixed_offsets_on=("valid", "test"),
        vis_load_backend="read_video",
        size_ratio=None,
        attr_annot_path=None,
        max_attr_per_vid=None,
        meta_dir="./data/avdeepfake",
        split2meta=None,
        split2vids_dir=None,
        path_keys=None,
        path_join_keys=None,
        path_join_sep="/",
        path_prefix="",
        path_strip_prefix="",
        real_only=True,
        real_field_candidates=None,
        real_values=None,
        fake_values=None,
        allow_missing_files=False,
        dataset_name="avdeepfake",
        max_clip_len_sec=11,
        assume_all_real_if_label_missing=False,
        max_retry_per_sample=10,
        min_video_frames=None,
        min_audio_frames=None,
    ):
        super().__init__()
        self.split = split
        self.base_vids_dir = vids_dir
        self.transforms = transforms
        self.splits_path = splits_path
        self.seed = seed
        self.load_fixed_offsets_on = [] if load_fixed_offsets_on is None else list(load_fixed_offsets_on)
        self.vis_load_backend = vis_load_backend
        self.size_ratio = size_ratio
        self.meta_dir = meta_dir
        self.path_prefix = path_prefix
        self.path_strip_prefix = path_strip_prefix
        self.real_only = real_only
        self.allow_missing_files = allow_missing_files
        self.dataset_name = dataset_name
        self.max_clip_len_sec = max_clip_len_sec
        self.assume_all_real_if_label_missing = assume_all_real_if_label_missing
        self.max_retry_per_sample = max_retry_per_sample
        self.min_video_frames = min_video_frames
        self.min_audio_frames = min_audio_frames
        self.path_keys = path_keys or [
            "path",
            "video_path",
            "file",
            "filepath",
            "relative_path",
            "video",
        ]
        self.path_join_keys = path_join_keys or []
        self.path_join_sep = path_join_sep
        self.real_field_candidates = real_field_candidates or [
            "modify_type",
            "modality",
            "label",
            "manipulation_type",
            "type",
            "is_fake",
        ]
        self.real_values = {str(v).lower() for v in (real_values or ["real", "pristine", "original", "authentic", "0", "false"])}
        self.fake_values = {str(v).lower() for v in (fake_values or [
            "fake",
            "audio_fake",
            "video_fake",
            "av_fake",
            "1",
            "true",
            "audio_modified",
            "visual_modified",
            "video_modified",
            "both_modified",
            "manipulated",
        ])}
        self.split2meta = split2meta or {
            "train": "train.json",
            "valid": "val.json",
            "test": "test.json",
        }
        self.split2vids_dir = split2vids_dir or {}
        self.vids_dir = self._resolve_vids_dir(split)

        records = self._load_records(split)
        if self.real_only:
            records = [r for r in records if self._is_real_record(r)]
        records = self._filter_by_min_length(records)
        clip_paths = []
        for record in records:
            path = self._resolve_path(record)
            if path is None:
                continue
            clip_paths.append(path)

        if split in self.load_fixed_offsets_on:
            logging.info(f"Using fixed offset for {split}")
            self.vid2offset_params = get_fixed_offsets(transforms, split, splits_path, dataset_name)

        self.dataset = subsample_dataset(clip_paths, size_ratio, shuffle=split == "train")
        logging.info(f"{dataset_name}::{split} has {len(self.dataset)} items")

    def _filter_by_min_length(self, records):
        if self.min_video_frames is None and self.min_audio_frames is None:
            return records

        kept_records = []
        dropped = 0
        for record in records:
            if self.min_video_frames is not None:
                video_frames = record.get("video_frames")
                if video_frames is not None and int(video_frames) < int(self.min_video_frames):
                    dropped += 1
                    continue
            if self.min_audio_frames is not None:
                audio_frames = record.get("audio_frames")
                if audio_frames is not None and int(audio_frames) < int(self.min_audio_frames):
                    dropped += 1
                    continue
            kept_records.append(record)

        logging.info(
            f"{self.dataset_name}::{self.split} filtered by min length: "
            f"kept {len(kept_records)} / {len(records)} (dropped {dropped})"
        )
        return kept_records

    def _resolve_vids_dir(self, split):
        split_dir = self.split2vids_dir.get(split)
        if split_dir is None:
            return vids_dir_to_str(self.base_vids_dir)
        split_dir = Path(split_dir)
        if not split_dir.is_absolute():
            split_dir = Path(self.base_vids_dir) / split_dir
        return str(split_dir.resolve())

    def _load_records(self, split):
        meta_name = self.split2meta.get(split)
        if meta_name is None:
            raise ValueError(f"No metadata entry configured for split={split}.")
        meta_path = Path(meta_name)
        if not meta_path.is_absolute():
            meta_path = Path(self.meta_dir) / meta_name
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        if meta_path.suffix.lower() == ".json":
            with open(meta_path, "r", encoding="utf-8") as infile:
                data = json.load(infile)
        elif meta_path.suffix.lower() == ".csv":
            with open(meta_path, "r", encoding="utf-8") as infile:
                return list(csv.DictReader(infile))
        elif meta_path.suffix.lower() == ".txt":
            with open(meta_path, "r", encoding="utf-8") as infile:
                return [{"path": line.strip()} for line in infile if line.strip()]
        else:
            raise ValueError(f"Unsupported metadata suffix: {meta_path.suffix}")

        if isinstance(data, dict):
            for key in ["data", "items", "videos", "samples", "metadata"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            raise ValueError(f"Unsupported metadata dict format in {meta_path}")
        if isinstance(data, list):
            return data
        raise ValueError(f"Unsupported metadata format in {meta_path}")

    def _resolve_path(self, record):
        raw_path = None
        if self.path_join_keys:
            parts = []
            for key in self.path_join_keys:
                value = record.get(key)
                if value is None or str(value).strip() == "":
                    parts = []
                    break
                parts.append(str(value).strip())
            if parts:
                raw_path = self.path_join_sep.join(parts)
        for key in self.path_keys:
            if raw_path is not None:
                break
            if key in record and record[key]:
                raw_path = record[key]
                break
        if raw_path is None:
            raise KeyError(f"Could not find a video path in record keys: {list(record.keys())}")

        raw_path = str(raw_path)
        if self.path_strip_prefix and raw_path.startswith(self.path_strip_prefix):
            raw_path = raw_path[len(self.path_strip_prefix):].lstrip("/\\")
        if self.path_prefix:
            raw_path = os.path.join(self.path_prefix, raw_path)

        path = Path(raw_path)
        if not path.is_absolute():
            path = Path(self.vids_dir) / path
        path = path.resolve()

        if not path.exists():
            if self.allow_missing_files:
                logging.warning(f"Skipping missing file: {path}")
                return None
            raise FileNotFoundError(path)
        return str(path)

    def _is_real_record(self, record):
        for key in self.real_field_candidates:
            if key not in record:
                continue
            value = record[key]
            if isinstance(value, bool):
                return value is False if key == "is_fake" else value is True
            value_norm = str(value).lower()
            if value_norm in self.real_values:
                return True
            if value_norm in self.fake_values:
                return False
            if value_norm in {"orig", "original_video", "real_video", "clean"}:
                return True
            if any(tag in value_norm for tag in ["modified", "manipulated", "deepfake"]):
                return False
            if "fake" in value_norm:
                return False
        if self.assume_all_real_if_label_missing:
            return True
        raise ValueError(
            "Could not determine whether a record is real. "
            f"Checked keys: {self.real_field_candidates}; record keys: {list(record.keys())}; "
            f"record={record}"
        )

    def __getitem__(self, index):
        curr_index = index
        last_exc = None
        for _attempt in range(self.max_retry_per_sample):
            path = self.dataset[curr_index]
            try:
                rgb, audio, meta = get_video_and_audio(path, get_meta=True, end_sec=self.max_clip_len_sec)

                item = {
                    "video": rgb,
                    "audio": audio,
                    "meta": meta,
                    "path": path,
                    "targets": {},
                    "split": self.split,
                }

                if self.split in self.load_fixed_offsets_on:
                    rel_path = str(Path(path).relative_to(Path(self.vids_dir))).replace("\\", "/")
                    unique_id = rel_path.replace(".mp4", "")
                    offset_params = self.vid2offset_params[unique_id]
                    item["targets"]["offset_sec"] = offset_params["offset_sec"]
                    item["targets"]["v_start_i_sec"] = offset_params["v_start_i_sec"]
                    if "oos_target" in offset_params:
                        item["targets"]["offset_target"] = {
                            "oos": offset_params["oos_target"],
                            "offset": item["targets"]["offset_sec"],
                        }

                if self.transforms is not None:
                    item = self.transforms(item)

                return item
            except (AssertionError, FileNotFoundError, RuntimeError, ValueError) as exc:
                last_exc = exc
                logging.warning(f"Skipping problematic sample at {path}: {exc}")
                if self.split == "train":
                    curr_index = random.randrange(len(self.dataset))
                else:
                    curr_index = (curr_index + 1) % len(self.dataset)

        raise RuntimeError(
            f"Failed to fetch a valid sample after {self.max_retry_per_sample} attempts. "
            f"Last exception: {last_exc}"
        )

    def __len__(self):
        return len(self.dataset)


def vids_dir_to_str(path_like):
    return str(Path(path_like).resolve())
