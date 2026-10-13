"""Hash-verified numeric golden samples and ordered boundary comparison.

These are diagnostic records, not robot deployment profiles. Unknown physical
units may be recorded as null, but cannot thereby become verified calibration.
No model, video decoder, or device SDK is imported here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from xpolicy.core.artifacts import file_sha256, write_json
from xpolicy.core.contracts import Name, Schema, Sha256


LAYERS = ("O1", "O2", "A0", "A1", "A2", "A3", "A4", "A5")


class GoldenArray(Schema):
    key: Name
    shape: tuple[Annotated[int, Field(gt=0, strict=True)], ...]
    dtype: Name
    sha256: Sha256
    # Labels for axes such as camera pixel channel or state/action joint.
    axes: dict[str, tuple[Name, ...]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_axes(self):
        for axis, labels in self.axes.items():
            if not axis.isdecimal() or str(int(axis)) != axis or int(axis) >= len(self.shape):
                raise ValueError("axis labels require a canonical nonnegative dimension index")
            if len(labels) != self.shape[int(axis)] or len(set(labels)) != len(labels):
                raise ValueError("axis labels must be unique and match the declared dimension")
        return self


class GoldenLayer(Schema):
    name: Literal["O1", "O2", "A0", "A1", "A2", "A3", "A4", "A5"]
    metadata: dict[str, Any]
    arrays: dict[Name, GoldenArray] = Field(min_length=1)


class GoldenManifest(Schema):
    scope: Name
    sample: dict[str, Any]
    provenance: dict[str, Any]
    arrays_sha256: Sha256
    layers: tuple[GoldenLayer, ...]

    @model_validator(mode="after")
    def ordered_unique_layers(self):
        names = [layer.name for layer in self.layers]
        if not names or names != sorted(set(names), key=LAYERS.index):
            raise ValueError("golden layers must be nonempty, unique and in boundary order")
        keys = [array.key for layer in self.layers for array in layer.arrays.values()]
        if len(keys) != len(set(keys)):
            raise ValueError("each golden array requires a unique storage key")
        # Also reject NaN, non-JSON objects and unbounded language-specific values.
        json.dumps(self.model_dump(mode="json"), allow_nan=False)
        return self


def _numeric(value):
    array = np.asarray(value)
    if not array.ndim or not array.size or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
        raise ValueError("golden arrays must be nonempty finite bool/integer/float arrays")
    return np.array(array, copy=True, order="C")


def _digest(array):
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def save_bundle(path: Path, *, scope: str, sample: dict, provenance: dict, layers: list[dict]):
    """Write a new manifest plus NPZ; publish the manifest after the arrays.

    Layer input has name/metadata/arrays and optional axes per array. NPZ never
    contains Python objects. A failed write leaves no valid manifest to load.
    """
    values, declared = {}, []
    for layer in layers:
        arrays = {}
        for index, (name, value) in enumerate(layer["arrays"].items()):
            array = _numeric(value)
            key = f"{layer['name']}_{index}"
            values[key] = array
            arrays[name] = {"key": key, "shape": array.shape, "dtype": array.dtype.str,
                            "sha256": _digest(array), "axes": layer.get("axes", {}).get(name, {})}
        declared.append({"name": layer["name"], "metadata": layer["metadata"], "arrays": arrays})
    initial = GoldenManifest(scope=scope, sample=sample, provenance=provenance,
                             arrays_sha256="0" * 64, layers=declared)
    path = path.expanduser().resolve()
    path.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(path / "arrays.npz", **values)
    manifest = initial.model_copy(update={"arrays_sha256": file_sha256(path / "arrays.npz")})
    write_json(path / "manifest.json", manifest.model_dump(mode="json"))
    return path


def _json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate golden JSON key: {key}")
        result[key] = value
    return result


def load_bundle(path: Path):
    path = path.expanduser().resolve()
    manifest = GoldenManifest.model_validate(json.loads((path / "manifest.json").read_text(encoding="utf-8"), object_pairs_hook=_json_object))
    if file_sha256(path / "arrays.npz") != manifest.arrays_sha256:
        raise ValueError(f"golden NPZ SHA256 mismatch: {path}")
    values = {}
    with np.load(path / "arrays.npz", allow_pickle=False) as storage:
        expected = {item.key for layer in manifest.layers for item in layer.arrays.values()}
        if set(storage.files) != expected or len(storage.files) != len(expected):
            raise ValueError("golden NPZ keys differ from the manifest")
        for layer in manifest.layers:
            for item in layer.arrays.values():
                array = _numeric(storage[item.key])
                if array.shape != item.shape or array.dtype.str != item.dtype or _digest(array) != item.sha256:
                    raise ValueError(f"golden array content differs from the manifest: {layer.name}/{item.key}")
                array.setflags(write=False)
                values[item.key] = array
    return manifest, values


def _metadata_diff(expected, observed, field):
    if type(expected) is not type(observed):
        return {"field": field, "kind": "metadata_type", "expected": expected, "observed": observed}
    if isinstance(expected, dict):
        if expected.keys() != observed.keys():
            return {"field": field, "kind": "metadata_keys", "expected": sorted(expected), "observed": sorted(observed)}
        for key in expected:
            difference = _metadata_diff(expected[key], observed[key], f"{field}.{key}")
            if difference:
                return difference
    elif isinstance(expected, list):
        if len(expected) != len(observed):
            return {"field": field, "kind": "metadata_length", "expected": len(expected), "observed": len(observed)}
        for index, (left, right) in enumerate(zip(expected, observed)):
            difference = _metadata_diff(left, right, f"{field}[{index}]")
            if difference:
                return difference
    elif expected != observed:
        return {"field": field, "kind": "metadata_value", "expected": expected, "observed": observed}
    return None


def compare_bundles(expected: Path, observed: Path, *, layers=None, atol=0.0, rtol=0.0):
    """Stop at the first divergent boundary. Integers/pixels remain exact.

    A PASS describes only the requested layers; it is never an acceptance of
    physical units, model forward behavior, or device operation.
    """
    for name, value in (("atol", atol), ("rtol", rtol)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite nonnegative number")
    expected, observed = (Path(path).expanduser().resolve() for path in (expected, observed))
    baseline, left = load_bundle(expected)
    candidate, right = load_bundle(observed)
    selected = list(layers) if layers is not None else [item.name for item in baseline.layers]
    if not selected or selected != sorted(set(selected), key=lambda name: LAYERS.index(name)):
        raise ValueError("comparison layers must be nonempty, unique and in boundary order")
    report = {"schema_version": 1, "scope": "golden_boundary_comparison", "status": "PASS",
              "compared_layers": selected, "omitted_layers": [name for name in LAYERS if name not in selected],
              "tolerance": {"atol": atol, "rtol": rtol, "integer_pixel_comparison": "exact"},
              "expected_manifest_sha256": file_sha256(Path(expected) / "manifest.json"),
              "observed_manifest_sha256": file_sha256(Path(observed) / "manifest.json"),
              "not_verified": ["physical_calibration", "checkpoint_forward", "device_behavior"],
              "checks": [], "first_divergence": None}

    def fail(layer, difference):
        report.update(status="FAIL", first_divergence={"layer": layer, **difference})
        return report

    difference = _metadata_diff(baseline.sample, candidate.sample, "sample")
    if difference:
        return fail("sample_identity", difference)
    before, after = {x.name: x for x in baseline.layers}, {x.name: x for x in candidate.layers}
    if layers is None and before.keys() != after.keys():
        return fail("layer_layout", {"field": "layers", "kind": "layer_names", "expected": list(before), "observed": list(after)})
    for name in selected:
        if name not in before or name not in after:
            return fail(name, {"field": "layer", "kind": "missing_layer", "expected": name in before, "observed": name in after})
        one, two = before[name], after[name]
        difference = _metadata_diff(one.metadata, two.metadata, "metadata")
        if difference:
            return fail(name, difference)
        if one.arrays.keys() != two.arrays.keys():
            return fail(name, {"field": "arrays", "kind": "array_names", "expected": sorted(one.arrays), "observed": sorted(two.arrays)})
        for field, declaration in one.arrays.items():
            other = two.arrays[field]
            for attribute in ("shape", "dtype", "axes"):
                if getattr(declaration, attribute) != getattr(other, attribute):
                    return fail(name, {"field": field, "kind": attribute, "expected": getattr(declaration, attribute), "observed": getattr(other, attribute)})
            first, second = left[declaration.key], right[other.key]
            with np.errstate(over="ignore", invalid="ignore"):
                matched = np.isclose(second, first, atol=atol, rtol=rtol) if first.dtype.kind == "f" else second == first
            if not matched.all():
                index = tuple(int(x) for x in np.argwhere(~matched)[0])
                return fail(name, {"field": field, "kind": "values", "index": list(index),
                                   "labels": {axis: labels[index[int(axis)]] for axis, labels in declaration.axes.items()},
                                   "expected": first[index].item(), "observed": second[index].item(),
                                   "mismatched_elements": int(np.count_nonzero(~matched))})
            report["checks"].append({"layer": name, "field": field, "status": "PASS", "elements": int(first.size)})
    return report
