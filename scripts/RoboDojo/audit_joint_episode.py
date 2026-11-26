"""Cross-check episode 0 of the RoboDojo v3 joint export against its native HDF5."""

import argparse
import json
from pathlib import Path

import av
import cv2
import h5py
import numpy as np
import pyarrow.parquet as pq

from XPolicyLab.utils.process_data import decode_image_bit

CAMERAS = {
    "cam_high": "cam_head",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}
JOINTS = (
    "left_arm_joint_states", "left_ee_joint_states",
    "right_arm_joint_states", "right_ee_joint_states",
)


def thumbnail(rgb):
    # Mean intensity isolates timing from JPEG/MP4 color encoding differences.
    return cv2.resize(rgb, (160, 120), interpolation=cv2.INTER_AREA).mean(axis=2, dtype=np.float32)


def compare_camera(path, start, fps, reference, next_frame):
    native = np.stack([thumbnail(decode_image_bit(bits)) for bits in reference])
    images = []
    with av.open(str(path)) as video:
        for frame in video.decode(video=0):
            index = (float(frame.pts * frame.time_base) - start) * fps
            if index < -0.5:
                continue
            if abs(index - len(images)) > 0.01:
                raise ValueError(f"unexpected video timestamp at {path}, frame {len(images)}")
            images.append(thumbnail(frame.to_ndarray(format="rgb24")))
            if len(images) == len(native):
                break
    if len(images) != len(native):
        raise ValueError(f"missing video timestamps at {path}")
    images = np.stack(images)
    offsets = (-1, 0, 1)
    interior = np.arange(1, len(images) - 1)
    errors = np.stack([
        np.abs(images[interior] - native[interior + offset]).mean(axis=(1, 2)) for offset in offsets
    ], axis=1)
    motion = np.abs(native[2:] - native[:-2]).mean(axis=(1, 2))
    moving = np.argsort(motion)[-max(1, len(interior) // 4):]
    result = {
        "interior_frames": len(interior),
        "mae_by_native_offset_minus1_zero_plus1": errors.mean(axis=0).tolist(),
        "most_motion_quartile_mae_by_offset": errors[moving].mean(axis=0).tolist(),
        "last_video_vs_native_last_mae": float(np.abs(images[-1] - native[-1]).mean()),
    }
    if next_frame is not None:
        result["last_video_vs_next_native_first_mae"] = float(np.abs(images[-1] - next_frame).mean())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="LeRobot root containing meta/, data/ and videos/")
    parser.add_argument("episode0", type=Path, help="native arrange_largest_number episode 0000000")
    parser.add_argument("--episode1", type=Path, help="native episode 0000001 for the boundary check")
    args = parser.parse_args()
    info = json.loads((args.root / "meta/info.json").read_text())
    fps = info["fps"]
    meta_path = args.root / "meta/episodes/chunk-000/file-000.parquet"
    columns = [col for col in pq.read_schema(meta_path).names if not col.startswith("stats/")]
    episodes = pq.read_table(meta_path, columns=columns)
    ep = episodes.filter(episodes["episode_index"].to_numpy() == 0).to_pylist()[0]
    data_path = info["data_path"].format(
        chunk_index=ep["data/chunk_index"], file_index=ep["data/file_index"]
    )
    data = pq.read_table(args.root / data_path, filters=[("episode_index", "=", 0)])
    n = ep["length"]
    np.testing.assert_array_equal(data["frame_index"].to_numpy(), np.arange(n))
    np.testing.assert_allclose(data["timestamp"].to_numpy(), np.arange(n) / fps, atol=2e-6, rtol=0)
    report = {"episode": 0, "frames": n, "fps": fps, "cameras": {}, "metadata_episodes_checked": len(episodes)}
    span_mismatch = np.zeros(len(episodes), dtype=bool)
    with h5py.File(args.episode0) as h5:
        report["native_fix"] = str(h5.attrs["fix_obs_shift_rgb"])
        for group, key in (("state", "observation.state"), ("action", "action")):
            native = np.concatenate([h5[f"{group}/{joint}"][:] for joint in JOINTS], axis=1)
            actual = np.array(data[key].to_pylist(), dtype=np.float32)
            np.testing.assert_array_equal(actual, native.astype(np.float32))
            report[f"{group}_max_abs_error_before_float32_cast"] = float(np.max(np.abs(actual - native)))
        for camera, name in CAMERAS.items():
            key = f"observation.images.{camera}"
            prefix = f"videos/{key}"
            start = ep[f"{prefix}/from_timestamp"]
            path = args.root / info["video_path"].format(
                video_key=key, chunk_index=ep[f"{prefix}/chunk_index"], file_index=ep[f"{prefix}/file_index"]
            )
            colors = f"vision/{name}/colors"
            if len(h5[colors]) != n:
                raise ValueError(f"native camera length mismatch: {name}")
            next_frame = None
            if args.episode1:
                with h5py.File(args.episode1) as following:
                    next_frame = thumbnail(decode_image_bit(following[colors][0]))
            result = compare_camera(path, start, fps, h5[colors], next_frame)
            result["metadata_video_span_frames"] = round((ep[f"{prefix}/to_timestamp"] - start) * fps)
            before, same, after = result["most_motion_quartile_mae_by_offset"]
            result["aligned"] = result["metadata_video_span_frames"] == n and same < min(before, after)
            if next_frame is not None and result["last_video_vs_next_native_first_mae"] < result["last_video_vs_native_last_mae"]:
                result["aligned"] = False
            spans = episodes[f"{prefix}/to_timestamp"].to_numpy() - episodes[f"{prefix}/from_timestamp"].to_numpy()
            mismatches = np.rint(spans * fps) != episodes["length"].to_numpy()
            result["metadata_episodes_with_span_mismatch"] = int(mismatches.sum())
            span_mismatch |= mismatches
            report["cameras"][camera] = result
    report["metadata_episodes_with_any_span_mismatch"] = int(span_mismatch.sum())
    report["episode_passed"] = all(item["aligned"] for item in report["cameras"].values())
    print(json.dumps(report, indent=2))
    return 0 if report["episode_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
