"""Download pinned upstream model assets, checking every file before accepting it.

Requires curl and, for composite GCS objects, google-crc32c. Partial downloads can
be resumed. Proxy settings are inherited from the caller.
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
G05_REVISION = "bb6cf7e45d0b2e8bd71c16c52961a93cc6125283"


def read_json(url):
    with urlopen(url, timeout=60) as response:
        return json.load(response)


def digest(path, algorithm):
    if algorithm == "crc32c":
        import google_crc32c
        value = google_crc32c.Checksum()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                value.update(chunk)
        return value.hexdigest().decode("ascii")
    value = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def download(entry):
    path = Path(entry["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    algorithm = entry.get("algorithm")
    expected = entry.get("digest")

    def verified(candidate):
        return (candidate.is_file() and candidate.stat().st_size == entry["size"]
                and (not expected or digest(candidate, algorithm) == expected))

    if verified(path):
        print(f"[verified] {path}", flush=True)
        return
    if path.exists():
        raise RuntimeError(f"Existing file failed integrity checks: {path}; inspect it before retrying")
    partial = path.with_name(path.name + ".partial")
    if partial.exists() and partial.stat().st_size > entry["size"]:
        raise RuntimeError(f"Oversized partial file: {partial}")
    if not verified(partial):
        print(f"[download] {path} ({entry['size'] / 1e9:.2f} GB)", flush=True)
        subprocess.run([
            "curl", "--fail", "--location", "--silent", "--show-error",
            "--retry", "5", "--retry-all-errors", "--connect-timeout", "30",
            "--continue-at", "-", "--output", str(partial), entry["url"],
        ], check=True)
    if not verified(partial):
        raise RuntimeError(f"Downloaded file failed size/checksum verification: {partial}")
    partial.replace(path)
    print(f"[verified] {path}", flush=True)


def download_multipart(entry, parts):
    """Resume disjoint HTTP byte ranges; accept the assembled file only by checksum."""
    path = Path(entry["path"])
    if path.exists():
        return download(entry)
    directory = path.with_name(path.name + ".parts")
    directory.mkdir(parents=True, exist_ok=True)
    layout_path = directory / "layout.json"
    layout = {"parts": parts, "size": entry["size"], "digest": entry["digest"]}
    if layout_path.exists() and json.loads(layout_path.read_text()) != layout:
        raise RuntimeError("Resume with the original --parts count and model revision")
    layout_path.write_text(json.dumps(layout) + "\n")
    width = (entry["size"] + parts - 1) // parts
    partial = path.with_name(path.name + ".partial")
    first = directory / "000"
    if partial.exists():
        if first.exists() or partial.stat().st_size > entry["size"]:
            raise RuntimeError(f"Inspect existing assembled partial file before retrying: {partial}")
        if partial.stat().st_size <= width:
            partial.replace(first)
        else:
            with partial.open("rb") as source:
                for index in range(parts):
                    remaining = min(width, partial.stat().st_size - source.tell())
                    if remaining <= 0:
                        break
                    with (directory / f"{index:03d}").open("wb") as destination:
                        while remaining:
                            chunk = source.read(min(8 * 1024 * 1024, remaining))
                            destination.write(chunk)
                            remaining -= len(chunk)
            partial.unlink()

    def fetch(index):
        start, end = index * width, min((index + 1) * width, entry["size"]) - 1
        target = directory / f"{index:03d}"
        if target.exists() and target.stat().st_size > end - start + 1:
            raise RuntimeError(f"Oversized download part: {target}")
        for attempt in range(6):
            offset = target.stat().st_size if target.exists() else 0
            if start + offset > end:
                return
            request = Request(entry["url"], headers={"Range": f"bytes={start + offset}-{end}"})
            try:
                with urlopen(request, timeout=60) as response:
                    expected = f"bytes {start + offset}-{end}/{entry['size']}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("The server did not honor the requested byte range")
                    with target.open("ab") as stream:
                        while chunk := response.read(min(8 * 1024 * 1024, end - start + 1 - stream.tell())):
                            stream.write(chunk)
                        if stream.tell() != end - start + 1:
                            raise EOFError("The server closed the byte range early")
                print(f"[part complete] {index + 1}/{parts}", flush=True)
                return
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(min(2 ** attempt, 10))

    with ThreadPoolExecutor(max_workers=parts) as executor:
        list(executor.map(fetch, range(parts)))
    with partial.open("wb") as destination:
        for index in range(parts):
            with (directory / f"{index:03d}").open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    if partial.stat().st_size != entry["size"] or digest(partial, entry["algorithm"]) != entry["digest"]:
        raise RuntimeError(f"Assembled file failed checksum verification: {partial}")
    partial.replace(path)
    shutil.rmtree(directory)
    print(f"[verified] {path}", flush=True)


def g05_entries(destination):
    repo = "OpenGalaxea/g05-robodojo"
    name = "g05_robodojo_fm_only_checkpoint.tar"
    entries = [{
        "path": str(destination / name), "size": 34425477120,
        "url": f"https://huggingface.co/{repo}/resolve/{G05_REVISION}/{name}",
        "algorithm": "sha256",
        "digest": "fd32a78840c78974750e3e33dab8bcb21d13510f3b31afc39dad0e0ee08ed510",
    }]
    return entries, {"repo": repo, "revision": G05_REVISION}


def g05_asset_entries(destination):
    repo = "OpenGalaxea/g05-robodojo"
    name = "g05_robodojo_train_base_assets.tar"
    return [{
        "path": str(destination / name), "size": 13699174400,
        "url": f"https://huggingface.co/{repo}/resolve/{G05_REVISION}/{name}",
        "algorithm": "sha256",
        "digest": "188fc8f092aa3ed2f67fcdb4a3e94ddbb44d4e0c41309e9f672a5a8594291218",
    }], {"repo": repo, "revision": G05_REVISION}


def pi05_entries(destination):
    prefix = "checkpoints/pi05_libero/"
    url = f"https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix={quote(prefix)}"
    objects = []
    while True:
        data = read_json(url)
        objects.extend(data.get("items", []))
        if "nextPageToken" not in data:
            break
        url = (f"https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix={quote(prefix)}"
               f"&pageToken={quote(data['nextPageToken'])}")
    if not objects:
        raise RuntimeError("The upstream pi05_libero checkpoint is empty")
    entries = []
    for file in objects:
        if file["name"].endswith("/"):
            continue
        entries.append({
            "path": str(destination / file["name"][len(prefix):]), "size": int(file["size"]),
            "url": (f"https://storage.googleapis.com/download/storage/v1/b/openpi-assets/o/"
                    f"{quote(file['name'], safe='')}?alt=media&generation={file['generation']}"),
            "algorithm": "md5" if "md5Hash" in file else "crc32c",
            "digest": base64.b64decode(file.get("md5Hash", file["crc32c"])).hex(),
            "generation": file["generation"],
        })
    return entries, {"source": "gs://openpi-assets/checkpoints/pi05_libero"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=["g05", "g05-assets", "pi05"])
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--parts", type=int, default=1, help="Parallel byte ranges for files larger than 1 GB")
    args = parser.parse_args()
    if args.workers < 1 or args.parts < 1 or args.parts > 16:
        parser.error("--workers must be positive; --parts must be between 1 and 16")
    defaults = {
        "g05": ROOT / "policy/G05/checkpoints/g05_robodojo_sim",
        "g05-assets": ROOT / "policy/G05/checkpoints/g05_robodojo_train_assets",
        "pi05": ROOT / "policy/Pi_05/checkpoints/pi05_libero",
    }
    factories = {"g05": g05_entries, "g05-assets": g05_asset_entries, "pi05": pi05_entries}
    destination = args.destination or defaults[args.model]
    entries, provenance = factories[args.model](destination.resolve())
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {**provenance, "files": entries}
    (destination / "download_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(
            lambda entry: download_multipart(entry, args.parts)
            if entry["size"] > 1024 ** 3 and args.parts > 1 else download(entry), entries,
        ))
    print(f"[complete] {args.model}: {len(entries)} verified files", flush=True)


if __name__ == "__main__":
    main()
