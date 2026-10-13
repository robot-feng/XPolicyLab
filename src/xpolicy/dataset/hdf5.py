"""Replay one XPL trajectory using the existing RGB image decoder."""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from XPolicyLab.utils.process_data import decode_image_bit
from xpolicy.core.contracts import DatasetProfile, Observation


class Hdf5Dataset:
    def __init__(self, path: Path, profile: DatasetProfile):
        if profile.decoder_id != "xpl_hdf5_v1" or profile.hdf5 is None:
            raise ValueError("unsupported dataset decoder; expected xpl_hdf5_v1")
        self.profile, self.layout = profile, profile.hdf5
        self._file = h5py.File(path, "r")
        try:
            self._validate()
        except BaseException:
            self.close()
            raise

    def _dataset(self, key):
        value = self._file.get(key)
        if not isinstance(value, h5py.Dataset):
            raise ValueError(f"missing HDF5 dataset: {key}")
        return value

    def _validate(self):
        lengths = []
        for field in self.layout.state.values():
            data = self._dataset(field.key)
            if data.ndim != 2 or field.column >= data.shape[1]:
                raise ValueError(f"{field.key} requires a [T,D] source with column {field.column}")
            lengths.append(data.shape[0])
        for camera in self.profile.cameras:
            data = self._dataset(camera.source_key)
            if data.ndim < 1:
                raise ValueError(f"camera {camera.name} has no frame axis")
            lengths.append(data.shape[0])
        if not lengths or min(lengths) < 1 or len(set(lengths)) != 1:
            raise ValueError("camera/state streams must have the same nonzero frame count")
        self.length = lengths[0]
        for field in self.layout.action.values():
            data = self._dataset(field.key)
            if data.ndim != 2 or data.shape[0] != self.length or field.column >= data.shape[1]:
                raise ValueError(f"action source {field.key} differs from the declared layout")
        if self.layout.timestamp_key is None:
            self.timestamps = np.arange(self.length, dtype=np.float64) / self.profile.sample_hz
            self.timebase_source = "nominal_index_rate"
        else:
            self.timestamps = np.asarray(self._dataset(self.layout.timestamp_key)[()])
            if self.timestamps.shape != (self.length,) or self.timestamps.dtype.kind not in "iuf" or not np.isfinite(self.timestamps).all():
                raise ValueError("timestamps must be a finite [T] real array")
            self.timebase_source = "stored"
        if np.any(self.timestamps[1:] <= self.timestamps[:-1]):
            raise ValueError("recorded timestamps must be strictly increasing")
        if self.layout.frame_id_key is None:
            self.frame_ids = np.arange(self.length, dtype=np.int64)
        else:
            self.frame_ids = np.asarray(self._dataset(self.layout.frame_id_key)[()])
            if self.frame_ids.shape != (self.length,) or self.frame_ids.dtype.kind not in "iu" or np.any(self.frame_ids < 0):
                raise ValueError("frame IDs must be a non-negative [T] integer array")
            if any(int(right) <= int(left) for left, right in zip(self.frame_ids, self.frame_ids[1:])):
                raise ValueError("frame IDs must be strictly increasing")
        instruction = self._dataset(self.layout.instruction_key)
        if self.layout.instruction_mode == "per_frame" and instruction.shape != (self.length,):
            raise ValueError("per_frame instructions must have shape [T]")
        if self.layout.instruction_mode == "constant" and instruction.shape != ():
            raise ValueError("constant instructions must be a scalar string or JSON list")

    def _instruction(self, index):
        source = self._dataset(self.layout.instruction_key)
        value = source.asstr()[index] if self.layout.instruction_mode == "per_frame" else source.asstr()[()]
        if self.layout.instruction_mode == "constant":
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = value
            if isinstance(parsed, list):
                if self.layout.instruction_index >= len(parsed):
                    raise ValueError("instruction_index exceeds the recorded alternatives")
                value = parsed[self.layout.instruction_index]
        if not isinstance(value, str):
            raise ValueError("dataset instruction must be a string")
        return value

    def observe(self, index: int, episode_id: str) -> Observation:
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < self.length:
            raise IndexError(f"dataset row out of range: {index}")
        images = {camera.name: decode_image_bit(self._dataset(camera.source_key)[index]) for camera in self.profile.cameras}
        state = np.array([self._dataset(self.layout.state[field.name].key)[index, self.layout.state[field.name].column] for field in self.profile.state_fields])
        return Observation(
            images=images, state=state,
            state_names=tuple(field.name for field in self.profile.state_fields),
            state_units=tuple(field.unit for field in self.profile.state_fields),
            instruction=self._instruction(index), timestamp_s=float(self.timestamps[index]),
            clock_domain=self.profile.clock_domain, frame_id=int(self.frame_ids[index]), episode_id=episode_id,
        )

    def labels(self, index: int, count: int):
        if not self.layout.action:
            return None
        values = np.column_stack([self._dataset(self.layout.action[field.name].key)[index:index + count, self.layout.action[field.name].column] for field in self.profile.action_fields])
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("action labels must contain finite physical values")
        return values

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
