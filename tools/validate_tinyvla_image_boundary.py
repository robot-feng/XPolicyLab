"""Compare TinyVLA's actual image-loading methods against a retained Git baseline.

Uses real Torch/HDF5 and synthetic camera fixtures. Constructor, augmentation,
language backbone, checkpoint inference and physical evaluation are not run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import subprocess
import sys

import cv2
import h5py
import numpy as np

from XPolicyLab.policy.TinyVLA.image_preprocessing import read_camera_images
from XPolicyLab.utils.process_data import decode_image_bit, encode_image_bit
from xpolicy.core.artifacts import file_sha256, source_revision, write_json


ROOT = Path(__file__).resolve().parents[1]
LOADERS = [Path("policy/TinyVLA/tinyvla/data_utils") / name for name in ("datasets.py", "utils.py")]


def _methods(source, name, torch):
    tree = ast.parse(source)
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "EpisodicDataset")
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in {"__getitem__", "_locate_transition"}]
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"torch": torch, "np": np, "h5py": h5py, "cv2": cv2, "read_camera_images": read_camera_images}
    exec(compile(module, name, "exec"), namespace)
    return namespace["EpisodicDataset"]


def _fixture(path, layout):
    frame = np.empty((24, 32, 3), np.uint8)
    frame[:] = [218, 43, 12]
    if layout.startswith("legacy"):
        # Retained failure fixture only, not a training-data conversion/export.
        success, value = cv2.imencode(".jpg", frame)
        if not success:
            raise ValueError("unable to construct historical fixture")
        payload = value.tobytes()
    else:
        payload = encode_image_bit(frame)
    expected = frame if layout == "raw" else decode_image_bit(payload)
    imsize = len(payload) if layout.endswith("matrix") else 32
    if imsize != expected.shape[1]:
        expected = cv2.resize(expected, (320, 180), interpolation=cv2.INTER_LINEAR)
    with h5py.File(path, "w") as file:
        file.attrs["source_kind"] = "synthetic_regression_fixture"
        file.attrs["sim"], file.attrs["compress"] = True, layout != "raw"
        file.create_dataset("language_raw", data=np.asarray([b"synthetic image boundary fixture"]))
        for key in ("action", "observations/qpos", "observations/qvel"):
            file.create_dataset(key, data=np.zeros((2, 2), np.float32))
        for name in ("left", "right", "head"):
            key = f"observations/images/{name}"
            if layout == "raw":
                file.create_dataset(key, data=np.stack([frame, frame]))
            elif layout.endswith("matrix"):
                row = np.asarray(bytearray(payload), dtype=np.uint8)[None]
                file.create_dataset(key, data=np.stack([row, row]))
            else:
                field = file.create_dataset(key, (2,), dtype=h5py.vlen_dtype(np.dtype("uint8")))
                field[0] = field[1] = np.asarray(bytearray(payload), dtype=np.uint8)
    return expected, imsize


def _run(cls, path, imsize, expected, torch):
    dataset = cls()
    dataset.dataset_path_list, dataset.episode_ids = [str(path)], [0]
    dataset.episode_len, dataset.cumulative_len = np.array([2]), np.array([2])
    dataset.max_episode_len, dataset.chunk_size = 2, 2
    dataset.camera_names, dataset.imsize = ("left", "right", "head"), imsize
    dataset.transformations, dataset.augment_images, dataset.policy_class = [], False, "ACT"
    dataset.norm_stats = {"qpos_mean": torch.zeros(2), "qpos_std": torch.ones(2), "action_mean": torch.zeros(2), "action_std": torch.ones(2)}
    try:
        images, _, _, _ = dataset[0]
        reference = torch.from_numpy(np.stack([expected] * 3)).permute(0, 3, 1, 2) / 255.
        shape_equal = images.shape == reference.shape
        equal = shape_equal and torch.equal(images, reference)
        channels = images[0].mean(dim=(1, 2)).tolist()
        return {"status": "PASS" if equal else "FAIL", "shape": list(images.shape), "expected_shape": list(reference.shape),
                "dtype": str(images.dtype), "exact_pixels_equal": equal,
                "channel_means_rgb": channels, "red_dominant_fixture_preserved": channels[0] > channels[2] + .4,
                "first_divergence": None if equal else "TinyVLA.training.images"}
    except Exception as exc:
        return {"status": "FAIL", "error": {"type": type(exc).__name__, "message": str(exc)},
                "first_divergence": "TinyVLA.training.images"}


def validate(output, baseline_ref="c378d3f04d68f41681077d97ea68b632598f9cd6"):
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    sources = {str(path): file_sha256(ROOT / path) for path in [*LOADERS, Path("policy/TinyVLA/image_preprocessing.py"), Path("utils/process_data.py"), Path("tools/validate_tinyvla_image_boundary.py")]}
    report = {"schema_version": 1, "scope": "TinyVLA_image_loader_methods_before_after", "status": "RUNNING",
              "fixture_source_kind": "synthetic", "fixture_usage": "retained_failure_counterexamples_not_training_exports",
              "provenance": {**source_revision(), "source_sha256": sources}, "baseline_ref": baseline_ref, "cases": [],
              "not_run": ["loader_constructor", "augmentation", "language_backbone", "checkpoint_inference", "full_training", "debug_eval", "simulator_or_hardware"]}
    write_json(output / "IMAGE_BOUNDARY_REPORT.json", report)
    try:
        import torch
        report["runtime"] = {"python": sys.version, "executable": sys.executable, "torch": torch.__version__, "cv2": cv2.__version__, "h5py": h5py.__version__}
        report["baseline_git_sha"] = subprocess.check_output(["git", "rev-parse", baseline_ref], cwd=ROOT, text=True).strip()
        for path in LOADERS:
            old_source = subprocess.check_output(["git", "show", f"{baseline_ref}:{path.as_posix()}"], cwd=ROOT, text=True)
            old = _methods(old_source, f"{baseline_ref}:{path}", torch)
            current = _methods((ROOT / path).read_text(), str(path), torch)
            for layout in ("raw", "marked_vector", "legacy_vector", "marked_matrix", "legacy_matrix"):
                fixture = output / f"{path.stem}-{layout}.hdf5"
                expected, imsize = _fixture(fixture, layout)
                report["cases"].append({"loader": path.as_posix(), "storage": layout, "fixture": fixture.name,
                                        "baseline_source_sha256": hashlib.sha256(old_source.encode()).hexdigest(),
                                        "baseline": _run(old, fixture, imsize, expected, torch),
                                        "current": _run(current, fixture, imsize, expected, torch)})
        report["baseline_failures_retained"] = sum(case["baseline"]["status"] == "FAIL" for case in report["cases"])
        report["counterexamples_reproduced"] = {
            "encoded_vector_shape_failure": all(case["baseline"].get("error", {}).get("type") == "IndexError" for case in report["cases"] if case["storage"].endswith("vector")),
            "legacy_red_blue_reversal": all(case["baseline"].get("red_dominant_fixture_preserved") is False for case in report["cases"] if case["storage"] == "legacy_matrix"),
        }
        report["status"] = "PASS" if (all(case["current"]["status"] == "PASS" for case in report["cases"]) and
                                         all(case["baseline"]["status"] == "PASS" for case in report["cases"] if case["storage"] == "raw") and
                                         all(report["counterexamples_reproduced"].values())) else "FAIL"
    except (ImportError, subprocess.SubprocessError) as exc:
        report["status"] = "BLOCKED"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    except (OSError, ValueError, TypeError) as exc:
        report["status"] = "FAIL"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["provenance"]["source_sha256_after"] = {name: file_sha256(ROOT / name) for name in sources}
        report["source_stable"] = sources == report["provenance"]["source_sha256_after"]
        if not report["source_stable"]:
            report["status"] = "FAIL"
        report["evidence_sha256"] = {path.name: file_sha256(path) for path in output.iterdir() if path.is_file() and path.name != "IMAGE_BOUNDARY_REPORT.json"}
        write_json(output / "IMAGE_BOUNDARY_REPORT.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ref", default="c378d3f04d68f41681077d97ea68b632598f9cd6")
    args = parser.parse_args(argv)
    report = validate(args.output, args.baseline_ref)
    print(report["status"])
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
