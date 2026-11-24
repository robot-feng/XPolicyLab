"""Actual HTTP resume/integrity checks without fetching model weights."""

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest


_SPEC = importlib.util.spec_from_file_location(
    "model_assets", Path(__file__).resolve().parents[1] / "scripts/download_pretrained_models.py"
)
assets = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(assets)
PAYLOAD = bytes(range(256)) * 1024


class RangeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        raw = self.headers.get("Range")
        if raw:
            start, end = map(int, raw.removeprefix("bytes=").split("-"))
            body = PAYLOAD[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(PAYLOAD)}")
        else:
            body = PAYLOAD
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class ModelAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
        cls.worker = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.worker.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.worker.join()

    def entry(self, path):
        return {"path": str(path), "size": len(PAYLOAD), "algorithm": "sha256",
                "digest": hashlib.sha256(PAYLOAD).hexdigest(),
                "url": f"http://127.0.0.1:{self.server.server_port}/weights"}

    def test_resume_ranges_and_verify_assembled_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights"
            path.with_name("weights.partial").write_bytes(PAYLOAD[:1234])
            assets.download_multipart(self.entry(path), 4)
            self.assertEqual(path.read_bytes(), PAYLOAD)
            self.assertFalse(path.with_name("weights.parts").exists())

    def test_corrupted_range_is_not_accepted_as_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights"
            path.with_name("weights.partial").write_bytes(b"bad prefix")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                assets.download_multipart(self.entry(path), 4)
            self.assertFalse(path.exists())

    def test_resume_prefix_spanning_multiple_ranges(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights"
            path.with_name("weights.partial").write_bytes(PAYLOAD[:150_123])
            assets.download_multipart(self.entry(path), 4)
            self.assertEqual(path.read_bytes(), PAYLOAD)

    def test_invalid_existing_checkpoint_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights"
            path.write_bytes(b"existing user file")
            with self.assertRaisesRegex(RuntimeError, "Existing file"):
                assets.download(self.entry(path))
            self.assertEqual(path.read_bytes(), b"existing user file")

    def test_changed_part_count_is_rejected_before_network_access(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "weights"
            parts = path.with_name("weights.parts")
            parts.mkdir()
            (parts / "layout.json").write_text('{"parts": 8, "size": 1, "digest": "old"}')
            with self.assertRaisesRegex(RuntimeError, "original --parts"):
                assets.download_multipart(self.entry(path), 4)


if __name__ == "__main__":
    unittest.main()
