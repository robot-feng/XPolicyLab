import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from xpolicy.core.artifacts import artifact_sha256, file_sha256
from xpolicy.diagnostics.assets import AssetCatalog, AssetDeclaration, inventory_assets


def git(root, *args):
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def write(root, name, content):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    git(root, "init", "-q")
    write(root, "LICENSE", "synthetic root license evidence")
    write(root, "NOTICE", "synthetic attribution")
    write(root, "policy/sample/model.py", "raise RuntimeError('must never import model')")
    write(root, "policy/sample/README.md", "**Original code:** https://example.com/source?secret=ignored#token\n")
    write(root, "policy/sample/vendor/code.py", "# synthetic vendor implementation\n")
    write(root, "policy/sample/vendor/LICENSE", "synthetic vendor license; no actual permission asserted")
    write(root, "policy/sample/vendor/nested/LICENSE", "synthetic nested terms")
    write(root, "policy/sample/vendor/weights.pt", "synthetic binary bytes")
    write(root, "policy/sample/scripts/helper.py", "# adapter support is not vendor discovery\n")
    write(root, "policy/sample/install.sh", "exit 99 # must not execute\n")
    git(root, "add", ".")
    return root


def catalog(path, rows):
    path.write_text(yaml.safe_dump({"schema_version": 1, "assets": rows}))
    return path


def documented(root, path, asset_id):
    return {"asset_id": asset_id, "kind": "source" if path.is_dir() else "model_artifact", "path": str(path),
            "expected_sha256": artifact_sha256(path),
            "source": {"uri": "https://example.com/source", "revision": "synthetic-revision"},
            "declared_license": "synthetic terms for tests only",
            "license_evidence": [{"uri": str(root / "policy/sample/vendor/LICENSE"), "sha256": file_sha256(root / "policy/sample/vendor/LICENSE")}],
            "review_evidence": {"uri": str(root / "NOTICE"), "sha256": file_sha256(root / "NOTICE")}}


def test_inventory_is_readonly_and_does_not_inherit_root_license(repository):
    report = inventory_assets(repository)
    assert report["status"] == "BLOCKED"
    assert report["selected_evidence_stable"]
    assert report["summary"]["adapters"] == 1
    assert report["summary"]["resource_candidates"] == 1
    assert len(report["source_candidates"]) == 2
    claim = report["source_candidates"][0]["origin_claims"][0]
    assert claim["uri"] == "https://example.com/source"
    assert all(row["status"] == "BLOCKED" for row in report["source_candidates"])
    assert len(report["root_attribution"]) == 2
    assert not (repository / "policy/sample/vendor/__pycache__").exists()


def test_missing_nested_terms_and_binary_origin_are_not_covered_by_parent(repository, tmp_path):
    path = catalog(tmp_path / "catalog.yaml", [documented(repository, repository / "policy/sample/vendor", "parent")])
    report = inventory_assets(repository, path)
    parent = next(row for row in report["source_candidates"] if row["path"].endswith("/vendor"))
    assert parent["status"] == "DOCUMENTED"
    assert report["status"] == "BLOCKED"
    assert any(row["path"].endswith("nested") for row in report["blockers"])
    assert report["resource_candidates"][0]["status"] == "BLOCKED"


def test_fully_pinned_documents_resolve_only_their_explicit_scopes(repository, tmp_path):
    paths = [repository / "policy/sample/vendor", repository / "policy/sample/vendor/nested", repository / "policy/sample/vendor/weights.pt"]
    path = catalog(tmp_path / "catalog.yaml", [documented(repository, item, f"scope-{index}") for index, item in enumerate(paths)])
    report = inventory_assets(repository, path)
    assert report["status"] == "DOCUMENTED" and report["blockers"] == []
    assert "legal_permission_or_redistribution" in report["not_verified"]


def test_subset_identity_never_claims_complete_component(repository, tmp_path):
    row = documented(repository, repository / "policy/sample/vendor", "subset")
    row["include"] = ["LICENSE"]
    row.pop("expected_sha256")
    path = catalog(tmp_path / "catalog.yaml", [row])
    report = inventory_assets(repository, path)
    item = report["declared_assets"][0]
    assert item["identity"]["scope"] == "explicit_subset_only"
    assert item["identity"]["algorithm"] == "XPL-SELECTED-FILES-v1"
    assert item["identity"]["expected_status"] == "NOT_DECLARED"
    assert all(row["status"] == "BLOCKED" for row in report["source_candidates"])


@pytest.mark.parametrize("include", [["../secret"], ["/absolute"], ["code.py", "code.py"], ["./code.py"], ["*.py"], ["folder\\file"], ["."]])
def test_invalid_subset_paths_rejected(include):
    with pytest.raises(ValueError):
        AssetDeclaration(asset_id="bad", kind="dataset", path="source", include=include)


def test_hash_mismatch_fails_without_hiding_other_assets(repository, tmp_path):
    row = documented(repository, repository / "policy/sample/vendor", "wrong")
    row["license_evidence"][0]["sha256"] = "0" * 64
    path = catalog(tmp_path / "catalog.yaml", [row, {"asset_id": "missing", "kind": "sdk", "path": str(tmp_path / "absent")}])
    report = inventory_assets(repository, path)
    assert report["status"] == "FAIL"
    assert report["declared_assets"][1]["status"] == "BLOCKED"
    assert any(item["code"] == "LICENSE_SHA256_MISMATCH" for item in report["declared_assets"][0]["findings"])


def test_symlink_is_recorded_as_failure_without_following_it(repository, tmp_path):
    outside = write(tmp_path, "outside", "private content must not be hashed through link")
    target = repository / "policy/sample/vendor/linked.pt"
    target.symlink_to(outside)
    git(repository, "add", "policy/sample/vendor/linked.pt")
    report = inventory_assets(repository)
    item = next(row for row in report["selected_file_evidence"] if row["path"].endswith("linked.pt"))
    assert report["status"] == item["status"] == "BLOCKED"
    assert "sha256" not in item
    assert item["finding"] == "EXTERNAL_OR_ABSOLUTE_SYMLINK_NOT_FOLLOWED"


def test_internal_license_alias_records_binding_and_target_separately(repository):
    alias = repository / "policy/sample/vendor/COPYING"
    alias.symlink_to("LICENSE")
    git(repository, "add", "policy/sample/vendor/COPYING")
    report = inventory_assets(repository)
    item = next(row for row in report["selected_file_evidence"] if row["path"].endswith("COPYING"))
    assert item["status"] == "RECORDED_LINK"
    assert item["target"]["sha256"] == file_sha256(repository / "policy/sample/vendor/LICENSE")
    assert item["link_sha256"] != item["target"]["sha256"]
    assert report["selected_evidence_stable"]


def test_external_asset_full_hash_is_materialized_without_import(repository, tmp_path):
    external = write(tmp_path, "external-checkpoint.pt", "synthetic external checkpoint")
    row = {"asset_id": "external", "kind": "checkpoint", "path": str(external), "expected_sha256": file_sha256(external)}
    path = catalog(tmp_path / "catalog.yaml", [row])
    report = inventory_assets(repository, path)
    assert report["declared_assets"][0]["identity"]["expected_status"] == "PASS"
    assert report["declared_assets"][0]["status"] == "BLOCKED"


def test_duplicate_catalog_ids_rejected():
    row = {"asset_id": "same", "kind": "sdk", "path": "missing"}
    with pytest.raises(ValueError, match="unique"):
        AssetCatalog(assets=[row, row])


def test_cli_keeps_blocked_inventory_and_has_no_model_dependencies(repository, tmp_path):
    output = tmp_path / "inventory.json"
    result = subprocess.run([sys.executable, "-m", "xpolicy", "assets", "inventory", "--root", str(repository), "--output", str(output)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2, result.stderr
    assert json.loads(output.read_text())["status"] == "BLOCKED"
    assert "secret=ignored" not in output.read_text()


def test_cli_invalid_root_still_writes_a_failure_report(tmp_path):
    output = tmp_path / "failed.json"
    result = subprocess.run([sys.executable, "-m", "xpolicy", "assets", "inventory", "--root", str(tmp_path), "--output", str(output)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2
    assert json.loads(output.read_text())["status"] == "FAIL"


def test_nested_gitlink_is_retained_without_materialization(repository):
    name = "policy/sample/vendor/submodule"
    git(repository, "update-index", "--add", "--cacheinfo", "160000", "1" * 40, name)
    report = inventory_assets(repository)
    row = next(item for item in report["selected_file_evidence"] if item["path"] == name)
    assert row["status"] == "GITLINK_NOT_AUDITED" and row["git_object_id"] == "1" * 40
    assert any(item["path"] == name for item in report["blockers"])


def test_selected_evidence_change_cannot_look_stable(repository, monkeypatch):
    from xpolicy.diagnostics import assets
    original, calls = assets._tracked, 0
    def track(root):
        nonlocal calls
        calls += 1
        if calls == 2:
            (root / "policy/sample/README.md").write_text("changed source declaration during inventory")
        return original(root)
    monkeypatch.setattr(assets, "_tracked", track)
    report = inventory_assets(repository)
    assert report["status"] == "FAIL" and not report["selected_evidence_stable"]
    assert "TRACKED_EVIDENCE_CHANGED_DURING_INVENTORY" in report["errors"]


def test_resource_kind_cannot_silently_document_source_scope(repository, tmp_path):
    row = documented(repository, repository / "policy/sample/vendor", "wrong-kind")
    row["kind"] = "dataset"
    path = catalog(tmp_path / "catalog.yaml", [row])
    report = inventory_assets(repository, path)
    assert report["declared_assets"][0]["status"] == "DOCUMENTED"
    assert all(item["status"] == "BLOCKED" for item in report["source_candidates"])


def test_empty_or_wrong_asset_identity_is_a_failure(repository, tmp_path):
    empty = write(tmp_path, "empty-license", "")
    row = documented(repository, repository / "policy/sample/vendor", "invalid")
    row["expected_sha256"] = "0" * 64
    row["license_evidence"] = [{"uri": str(empty), "sha256": file_sha256(empty)}]
    report = inventory_assets(repository, catalog(tmp_path / "catalog.yaml", [row]))
    assert report["status"] == "FAIL"
    assert report["declared_assets"][0]["identity"]["expected_status"] == "FAIL"


def test_asset_cli_imports_no_model_or_hardware_framework():
    code = "import sys; import xpolicy.diagnostics.assets; assert not any(name.split('.')[0] in {'torch', 'transformers', 'rospy', 'gdk'} for name in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_nonregular_external_asset_is_rejected_without_opening_fifo(repository, tmp_path):
    import os
    fifo = tmp_path / "not-a-checkpoint"
    os.mkfifo(fifo)
    path = catalog(tmp_path / "catalog.yaml", [{"asset_id": "fifo", "kind": "checkpoint", "path": str(fifo)}])
    report = inventory_assets(repository, path)
    assert report["status"] == "FAIL"
    assert any(item["code"] == "ASSET_EVIDENCE_ERROR" for item in report["declared_assets"][0]["findings"])


def test_cli_cannot_overwrite_a_previous_audit(repository, tmp_path):
    path = write(tmp_path, "old.json", '{"status":"FAIL","reason":"earlier evidence"}')
    previous = path.read_bytes()
    result = subprocess.run([sys.executable, "-m", "xpolicy", "assets", "inventory", "--root", str(repository), "--output", str(path)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 2 and path.read_bytes() == previous
