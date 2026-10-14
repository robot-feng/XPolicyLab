"""Local asset evidence inventory; never infer licensing from the root LICENSE.

This tool records files and explicit declarations. It does not interpret license
terms, authorize redistribution, import model code, or execute install scripts.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import Field, model_validator

from xpolicy.core.artifacts import artifact_sha256, file_sha256
from xpolicy.core.config import read_yaml
from xpolicy.core.contracts import ArtifactIdentity, Name, Schema, Sha256


RESOURCE_KINDS = {
    **dict.fromkeys((".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".bin"), "model_artifact_candidate"),
    **dict.fromkeys((".urdf", ".stl", ".obj", ".glb", ".dae", ".ply", ".usd", ".usda", ".usdc"), "robot_asset_candidate"),
    **dict.fromkeys((".pkl", ".pickle", ".npy", ".npz", ".h5", ".hdf5", ".parquet", ".arrow", ".zip", ".gz", ".zst"), "data_asset_candidate"),
    **dict.fromkeys((".png", ".jpg", ".jpeg", ".gif", ".webp", ".mp4", ".webm", ".mov", ".wav", ".mp3", ".pdf", ".svg"), "media_asset_candidate"),
}
ARTIFACT_EXTENSIONS = set(RESOURCE_KINDS)
ADAPTER_SUPPORT = {"scripts", "tests", "configs", "config", "data", "datasets", "results", "checkpoints", "inspect", "assets", "utils"}


class SourceDeclaration(Schema):
    uri: Name | None = None
    revision: Name | None = None


class AssetDeclaration(Schema):
    asset_id: Name
    kind: Literal["source", "checkpoint", "dataset", "sdk", "robot_asset", "model_artifact", "documentation_asset"]
    path: Path
    expected_sha256: Sha256 | None = None
    # A subset records only those files, never claims the entire directory.
    include: tuple[Name, ...] = ()
    source: SourceDeclaration = Field(default_factory=SourceDeclaration)
    declared_license: Name | None = None
    license_evidence: tuple[ArtifactIdentity, ...] = ()
    review_evidence: ArtifactIdentity | None = None
    notes: Name | None = None

    @model_validator(mode="after")
    def unique_files(self):
        if len(set(self.include)) != len(self.include):
            raise ValueError("asset include filenames must be unique")
        for name in self.include:
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or path.as_posix() != name or not path.parts:
                raise ValueError("asset include requires canonical relative filenames without traversal or globs")
            if any(char in name for char in "*?[]"):
                raise ValueError("asset include requires explicit filenames, not globs")
        if len({item.uri for item in self.license_evidence}) != len(self.license_evidence):
            raise ValueError("license evidence filenames must be unique")
        return self


class AssetCatalog(Schema):
    assets: tuple[AssetDeclaration, ...] = ()

    @model_validator(mode="after")
    def unique_assets(self):
        if len({item.asset_id for item in self.assets}) != len(self.assets):
            raise ValueError("catalog asset IDs must be unique")
        return self


def _license_name(path):
    name = path.name.casefold()
    return name.startswith(("license", "licence", "copying", "copyright", "notice", "third_party_notice", "third-party-notice"))


def _digest_rows(rows):
    """Identity of selected materialized files, explicitly distinct from a full tree."""
    digest = hashlib.sha256(b"XPL-SELECTED-FILES-v1\n")
    for row in sorted(rows, key=lambda item: item["path"]):
        name = row["path"].encode("utf-8")
        digest.update(len(name).to_bytes(8, "big")); digest.update(name)
        digest.update(bytes.fromhex(row["sha256"]))
    return digest.hexdigest()


def _file(path, *, label):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"materialized regular file required: {label}")
    # Reject symlinks in parent directories as well as at the leaf.
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ValueError(f"materialize symlink path before inventory: {label}")
    digest = file_sha256(path)
    after = path.stat()
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError(f"file changed while hashing: {label}")
    return {"path": label, "sha256": digest, "bytes": info.st_size}


def _tracked_file(root, name):
    path = root / name
    if not path.is_symlink():
        return {**_file(path, label=name), "status": "RECORDED"}
    target = os.readlink(path)
    normalized = Path(os.path.abspath(path.parent / target))
    # Record a relative, single-hop in-repository link without granting a
    # general symlink-following capability. External/chained links stay blocked.
    record = {"path": name, "link_sha256": hashlib.sha256(os.fsencode(target)).hexdigest()}
    if Path(target).is_absolute() or not normalized.is_relative_to(root):
        return {**record, "status": "BLOCKED", "finding": "EXTERNAL_OR_ABSOLUTE_SYMLINK_NOT_FOLLOWED"}
    try:
        materialized = _file(normalized, label=normalized.relative_to(root).as_posix())
    except (OSError, ValueError):
        return {**record, "status": "BLOCKED", "finding": "SYMLINK_TARGET_NOT_MATERIALIZED_REGULAR_FILE"}
    return {**record, "status": "RECORDED_LINK", "target": materialized}


def _tracked(root):
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=root, capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        raw = subprocess.run(["git", "ls-files", "--stage", "-z"], cwd=root, capture_output=True, check=True, timeout=10).stdout
    except subprocess.SubprocessError as exc:
        raise ValueError("repository index could not be read; use a local Git checkout with no unmerged entries") from exc
    if Path(top).resolve() != root:
        raise ValueError("asset inventory root must be the repository root")
    result = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, oid, stage = metadata.decode().split()
        if stage != "0":
            raise ValueError("resolve unmerged index entries before asset inventory")
        name = os.fsdecode(name)
        if not name or Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("invalid tracked filename")
        result.append({"path": name, "git_mode": mode, "git_object_id": oid})
    return sorted(result, key=lambda item: item["path"])


def _origin(readme, root):
    if not readme.is_file() or any(item.is_symlink() for item in (readme, *readme.parents)):
        return []
    candidates = []
    for number, line in enumerate(readme.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if "Original code:" not in line:
            continue
        for match in re.finditer(r"https?://[^\s<>\])|]+", line.split("Original code:", 1)[1]):
            try:
                parsed = urlsplit(match.group().rstrip(".,;"))
            except ValueError:
                continue
            if not parsed.hostname or parsed.username is not None or parsed.password is not None:
                continue
            # Never publish query/fragment credentials found in a source claim.
            uri = urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))
            candidates.append({"uri": uri, "file": readme.relative_to(root).as_posix(), "line": number,
                               "status": "LOCAL_README_CLAIM_NOT_REMOTE_VERIFIED"})
    return candidates


def _declaration(item, catalog_path):
    # Do not resolve symlinks before checking them: doing so hides their presence.
    raw = item.path.expanduser()
    path = raw if raw.is_absolute() else catalog_path.parent / raw
    path = Path(os.path.abspath(path))
    result = {"asset_id": item.asset_id, "kind": item.kind, "path": str(path),
              "declaration": item.model_dump(mode="json"), "status": "BLOCKED", "findings": [], "license_evidence": []}
    findings = result["findings"]
    if item.source.uri is None:
        findings.append({"code": "SOURCE_UNDECLARED"})
    if item.source.revision is None:
        findings.append({"code": "SOURCE_REVISION_UNDECLARED"})
    if item.declared_license is None:
        findings.append({"code": "LICENSE_UNDECLARED"})
    if not item.license_evidence:
        findings.append({"code": "LICENSE_EVIDENCE_MISSING"})
    if item.review_evidence is None:
        findings.append({"code": "REVIEW_EVIDENCE_MISSING"})
    try:
        if any(part.is_symlink() for part in (path, *path.parents)):
            raise ValueError("materialize asset symlinks before inventory")
        if not path.exists():
            findings.append({"code": "ASSET_NOT_MATERIALIZED"})
        else:
            if item.include:
                if not path.is_dir():
                    raise ValueError("include filenames require a directory asset")
                rows = [_file(path / name, label=name) for name in item.include]
                actual = _digest_rows(rows)
                result["identity"] = {"algorithm": "XPL-SELECTED-FILES-v1", "scope": "explicit_subset_only", "sha256": actual, "files": rows}
            else:
                actual = artifact_sha256(path) if path.is_dir() else _file(path, label=str(path))["sha256"]
                result["identity"] = {"algorithm": "XPL-DIR-SHA256-v1" if path.is_dir() else "SHA256", "scope": "full_materialized_asset", "sha256": actual}
            result["identity"]["expected_status"] = "NOT_DECLARED" if item.expected_sha256 is None else "PASS" if actual == item.expected_sha256 else "FAIL"
            if item.expected_sha256 is None:
                findings.append({"code": "EXPECTED_IDENTITY_UNDECLARED"})
            elif actual != item.expected_sha256:
                findings.append({"code": "ASSET_SHA256_MISMATCH"})
        for category, identities in (("license", item.license_evidence), ("review", (item.review_evidence,) if item.review_evidence else ())):
            for identity in identities:
                parsed = urlsplit(identity.uri)
                if parsed.scheme:
                    raise ValueError("evidence must be a local materialized file")
                evidence = Path(identity.uri).expanduser()
                evidence = evidence if evidence.is_absolute() else catalog_path.parent / evidence
                record = _file(Path(os.path.abspath(evidence)), label=identity.uri)
                if not record["bytes"]:
                    raise ValueError(f"empty {category} evidence file")
                record["status"] = "PASS" if record["sha256"] == identity.sha256 else "FAIL"
                result.setdefault(f"{category}_evidence", []).append(record)
                if record["status"] != "PASS":
                    findings.append({"code": f"{category.upper()}_SHA256_MISMATCH"})
        if "identity" in result:
            actual_after = (_digest_rows([_file(path / name, label=name) for name in item.include]) if item.include else
                            artifact_sha256(path) if path.is_dir() else _file(path, label=str(path))["sha256"])
            result["identity"]["sha256_after"] = actual_after
            if result["identity"]["sha256"] != actual_after:
                findings.append({"code": "ASSET_CHANGED_DURING_INVENTORY"})
    except (OSError, ValueError) as exc:
        findings.append({"code": "ASSET_EVIDENCE_ERROR", "type": type(exc).__name__, "message": str(exc)})
    errors = {"ASSET_SHA256_MISMATCH", "LICENSE_SHA256_MISMATCH", "REVIEW_SHA256_MISMATCH", "ASSET_EVIDENCE_ERROR", "ASSET_CHANGED_DURING_INVENTORY"}
    result["status"] = "FAIL" if any(finding["code"] in errors for finding in findings) else "BLOCKED" if findings else "DOCUMENTED"
    return result


def inventory_assets(root: Path, catalog_path: Path | None = None):
    root = root.expanduser().resolve()
    tracked = _tracked(root)
    before_index = tracked
    entries, errors = {}, []
    for item in tracked:
        name, mode = item["path"], item["git_mode"]
        path = Path(name)
        if not (mode == "160000" or name in {"LICENSE", "NOTICE"} or (path.parts[0] == "policy" and
                (path.suffix.lower() in ARTIFACT_EXTENSIONS or _license_name(path) or len(path.parts) <= 3))):
            continue
        if mode == "160000":
            entries[name] = {**item, "status": "GITLINK_NOT_AUDITED"}
            continue
        try:
            entries[name] = {**item, **_tracked_file(root, name)}
        except (OSError, ValueError) as exc:
            entries[name] = {**item, "status": "FAIL", "error": {"type": type(exc).__name__, "message": str(exc)}}
            errors.append(name)
    names = [item["path"] for item in tracked]
    adapters = sorted(Path(name).parent.as_posix() for name in names if len(Path(name).parts) == 3 and Path(name).name == "model.py" and Path(name).parts[0] == "policy")
    components = {}
    for name in names:
        path = Path(name)
        if len(path.parts) > 3 and path.parts[0] == "policy" and path.parts[2] not in ADAPTER_SUPPORT and not path.parts[2].startswith((".", "_")):
            component = Path(*path.parts[:3]).as_posix()
            components.setdefault(component, "DIRECT_CHILD_SOURCE_CANDIDATE")
        if path.parts[0] == "policy" and _license_name(path) and len(path.parts) > 3:
            components.setdefault(path.parent.as_posix(), "NESTED_LICENSE_SCOPE_CANDIDATE")
    candidate_rows = []
    for component, discovery in sorted(components.items()):
        licenses = [entries[name] for name in entries if Path(name).parent.as_posix() == component and _license_name(Path(name))]
        adapter = Path(*Path(component).parts[:2]).as_posix()
        candidate_rows.append({"path": component, "discovery": discovery, "license_files_at_scope_root": licenses,
            "origin_claims": _origin(root / adapter / "README.md", root),
            "findings": ["COMPONENT_SCOPE_AND_PROVENANCE_REVIEW_REQUIRED"] + ([] if licenses else ["NO_LICENSE_FILE_AT_CANDIDATE_ROOT"]),
            "status": "BLOCKED"})
    resources = [{**entries[name], "kind": RESOURCE_KINDS[Path(name).suffix.lower()],
                 "status": "BLOCKED", "findings": ["RESOURCE_ORIGIN_AND_TERMS_UNDECLARED"]}
                for name in sorted(entries) if Path(name).suffix.lower() in ARTIFACT_EXTENSIONS]
    declarations, catalog_identity = [], None
    if catalog_path is not None:
        catalog_path = catalog_path.expanduser().absolute()
        catalog_identity = _file(catalog_path, label=str(catalog_path))
        catalog = AssetCatalog.model_validate(read_yaml(catalog_path))
        declarations = [_declaration(item, catalog_path) for item in catalog.assets]
        if file_sha256(catalog_path) != catalog_identity["sha256"]:
            errors.append("CATALOG_CHANGED_DURING_INVENTORY")
    documented = {str(Path(row["path"])): row for row in declarations if row["status"] == "DOCUMENTED" and
                  row.get("identity", {}).get("scope") == "full_materialized_asset"}
    for row in candidate_rows + resources:
        declaration = documented.get(str(root / row["path"]))
        accepted = {"source"} if "discovery" in row else {
            "model_artifact_candidate": {"model_artifact", "checkpoint"},
            "robot_asset_candidate": {"robot_asset"}, "data_asset_candidate": {"dataset", "model_artifact"},
            "media_asset_candidate": {"documentation_asset"},
        }[row["kind"]]
        if declaration is not None and declaration["kind"] in accepted:
            row.update(status="DOCUMENTED", findings=[], documented_by=declaration["asset_id"])
    # Re-read every selected file rather than assuming git SHA describes bytes.
    stable = before_index == _tracked(root)
    for name, item in entries.items():
        if item["status"] in {"RECORDED", "RECORDED_LINK", "BLOCKED"}:
            try:
                current = _tracked_file(root, name)
                stable = all(item.get(key) == value for key, value in current.items()) and stable
            except (OSError, ValueError):
                stable = False
    if not stable:
        errors.append("TRACKED_EVIDENCE_CHANGED_DURING_INVENTORY")
    blockers = ([{"path": row["path"], "kind": "source_candidate", "findings": row["findings"]} for row in candidate_rows if row["findings"]] +
                [{"path": row["path"], "kind": row["kind"], "findings": row["findings"]} for row in resources if row["findings"]] +
                [{"asset_id": row["asset_id"], "path": row["path"], "kind": row["kind"], "findings": row["findings"]} for row in declarations if row["status"] != "DOCUMENTED"])
    missing_root = [name for name in ("LICENSE", "NOTICE") if name not in entries or entries[name]["status"] not in {"RECORDED", "RECORDED_LINK"} or
                    not entries[name].get("bytes", entries[name].get("target", {}).get("bytes", 0))]
    if missing_root:
        blockers.append({"path": ".", "findings": [f"ROOT_{name}_MISSING" for name in missing_root]})
    for item in entries.values():
        if item["status"] == "GITLINK_NOT_AUDITED":
            blockers.append({"path": item["path"], "findings": ["GITLINK_NOT_MATERIALIZED_OR_AUDITED"]})
        elif item["status"] == "BLOCKED":
            blockers.append({"path": item["path"], "findings": [item["finding"]]})
    repository_sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
    return {"schema_version": 1, "scope": "local_asset_evidence_inventory", "status": "FAIL" if errors or any(row["status"] == "FAIL" for row in declarations) else "BLOCKED" if blockers else "DOCUMENTED",
            "root": str(root), "selection": "git_index_tracked_files_plus_explicit_catalog_assets",
            "repository_sha": repository_sha.stdout.strip() if repository_sha.returncode == 0 else None,
            "audit_implementation_sha256": file_sha256(Path(__file__)),
            "index_selection_sha256": hashlib.sha256(json.dumps(tracked, sort_keys=True).encode()).hexdigest(),
            "selected_evidence_stable": stable, "catalog": catalog_identity,
            "summary": {"adapters": len(adapters), "source_candidates": len(candidate_rows), "resource_candidates": len(resources),
                        "resource_kind_counts": dict(Counter(item["kind"] for item in resources)),
                        "license_evidence_files": sum(_license_name(Path(name)) for name in entries), "declared_assets": len(declarations),
                        "declared_status_counts": dict(Counter(item["status"] for item in declarations)), "blockers": len(blockers)},
            "adapters": adapters, "root_attribution": [entries[name] for name in ("LICENSE", "NOTICE") if name in entries],
            "source_candidates": candidate_rows, "resource_candidates": resources, "declared_assets": declarations,
            "selected_file_evidence": list(entries.values()), "blockers": blockers, "errors": errors,
            "not_verified": ["legal_permission_or_redistribution", "license_scope_inheritance", "review_adequacy", "remote_origin_authenticity",
                             "undeclared_external_or_untracked_assets", "unmaterialized_submodule_contents", "EVA_assets", "package_environment_compatibility"]}
