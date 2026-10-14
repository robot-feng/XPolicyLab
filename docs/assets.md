# Asset and attribution evidence inventory

The `assets` CLI inventories local evidence for blueprint A05. It reads tracked
files and an optional explicit external-asset catalog. It never imports model code,
loads a pickle/checkpoint, runs install scripts, contacts a model host, copies vendor
resources, or treats the repository's Apache license as all third-party licenses.

```bash
xpolicy assets inventory --root . \
  --catalog configs/assets/catalog.yaml \
  --output runs/asset-inventory/repository.json
```

Exit code 0 means the selected scopes have complete pinned declarations
(`DOCUMENTED`). Exit code 2 retains a `BLOCKED` or `FAIL` report. This command is
an evidence audit, not a runtime admission change or an authorization to redistribute.
It does not change model installation, checkpoint loading or EVA behavior.
The output must be a new file; previous audit results are never overwritten on retry.

## Tracked repository discovery

The inventory root must be a Git repository root. Discovery uses the index's tracked
file list and records its identity; it does not recursively collect caches, secrets,
virtual environments, downloaded checkpoints or untracked runs.

| Evidence | Treatment |
| --- | --- |
| Root `LICENSE` / `NOTICE` | Hash and retain; neither is inherited by third-party candidates |
| `policy/*/model.py` | Count adapter directories without importing them |
| Direct source directories / nested license scopes | Record heuristic candidates requiring explicit scope/provenance review |
| Local `Original code:` README URLs | Record the file/line claim; strip credentials-bearing URLs and query/fragment data; no remote authenticity claim |
| License/notice/copyright files | Record path, bytes and SHA256 |
| Model/data/robot/media resource extensions | Record resource candidates and hashes; a `.pt` is not assumed to be a trained policy checkpoint |
| Relative one-hop internal file links | Record link binding and target file hash separately |
| External, absolute, chained or missing symlink targets | BLOCKED without reading the target |
| Gitlinks/submodules | Keep Git object identity and BLOCKED; nested contents are not claimed to have been audited |

Adapter support directories such as `scripts`, `tests` and `configs` are not labelled
as vendor trees solely because they contain Python files. Discovery is a candidate
inventory, not a universal ownership classifier. Known model/data/mesh/media suffixes
are defined in `RESOURCE_KINDS`; undeclared external and untracked assets remain
unverified. `selected_evidence_stable` checks the selected files and index before/after,
not every unselected vendor source file. The report separately identifies the audit
implementation bytes and repository HEAD; HEAD alone does not describe dirty files.

## Explicit asset declarations

`configs/assets/catalog.yaml` demonstrates a pinned local license-file subset. It
intentionally remains BLOCKED: historical source revision, full source identity and
review evidence have not been established. External assets use the same schema:

```yaml
schema_version: 1
assets:
  - asset_id: local_policy_weights
    kind: checkpoint
    path: /path/to/run/final_model/pytorch_model.pt
    source:
      uri: /path/to/original/training-run
    notes: Supply original source revision and asset-specific terms from evidence.
```

Supported kinds are source, checkpoint, dataset, sdk, robot_asset, model_artifact and
documentation_asset. Paths are relative to the catalog. Missing paths, source claims,
revisions, expected hashes, license declarations, license evidence and review evidence
remain visible findings. URI values are declarations, not proof of remote identity.

`expected_sha256` pins the asset identity. `license_evidence` is a list of local
`{uri, sha256}` file references; `review_evidence` is one such reference. Empty evidence,
changed hashes and nonregular evidence files fail. The tool checks bytes and declared
scope, not legal permission, license compatibility or the adequacy of a review record.
No review record is generated or approved automatically.

For a file, identity is SHA256. A full materialized directory uses the existing
`XPL-DIR-SHA256-v1` algorithm documented in [Core development](core-development.md).
Directory identity rejects symlinks. Both identities are rechecked after collection.
Use `include: [meta/info.json, meta/modality.json]` for an explicit subset: no glob,
traversal, absolute filename or duplicate is accepted. Its identity is
`XPL-SELECTED-FILES-v1`: SHA256 begins with ASCII `XPL-SELECTED-FILES-v1\n`; sorted
POSIX relative names are UTF-8 length-prefixed with 8 unsigned big-endian bytes,
followed by name bytes and the raw 32-byte file digest. A metadata subset never pins
all videos/parquet or authorizes an entire dataset.

A complete full-scope declaration can document the exact matching discovered
candidate of the corresponding kind. It does not document nested license scopes or
resource files automatically. Partial identity cannot clear a full-source candidate.
These scope rules prevent a parent's license or one matching hash from standing in
for unrelated assets.

## Recorded local findings

The repository scan found 67 adapters, 72 source/scope candidates, 78 local
license/notice/copyright files and 136 resource candidates: 37 model artifacts,
8 data assets and 91 media assets. These are inventory counts, not counts of license
violations. The repository remains BLOCKED for undocumented scopes/provenance/reviews.
The local G05 license alias is retained as a relative internal link, with the target
license hash; no source or licensing text was changed to make the scan pass.

The local G2 asset catalog under `runs/asset-inventory/` separately checks the existing
training registry file, trained checkpoint, recorded dataset metadata and Qwen3.5 base
files against identities from the earlier real-forward report. Their expected hashes
match; source/revision/asset-specific terms/review gaps remain BLOCKED. The checkpoint
run does not include checkpoint-specific license evidence in the supplied files;
the recorded dataset metadata does not declare data authorization. The local base
model card's `apache-2.0` label and license file are retained as local evidence.
They are not automatically inherited by the trained checkpoint or recorded dataset.

No hardware SDK or EVA resources have been admitted by this audit. A05 remains
incomplete until required scopes, original provenance and applicable permissions are
reviewed. The inventory makes those outstanding items explicit and reproducible.
