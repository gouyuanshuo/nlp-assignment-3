import hashlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from mosei.download import download_file


@pytest.mark.parametrize("verify_hash", [True, False])
def test_download_ranges_reassembles_and_verifies_bytes(tmp_path, verify_hash):
    payload = b"synthetic-dataset-test-content" * 400

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            low, high = self.headers["Range"].removeprefix("bytes=").split("-")
            low, high = int(low), int(high)
            block = payload[low : high + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {low}-{high}/{len(payload)}")
            self.send_header("Content-Length", str(len(block)))
            self.end_headers()
            self.wfile.write(block)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        dest = tmp_path / "download.pkl"
        sha = hashlib.sha256(payload).hexdigest()
        download_file(
            f"http://127.0.0.1:{server.server_port}/fixture",
            dest,
            sha if verify_hash else None,
            workers=2,
            chunk_bytes=2048,
            expected_size=len(payload),
        )
        assert dest.read_bytes() == payload
        assert not dest.with_suffix(".pkl.parts").exists()
        assert download_file("http://unused.invalid", dest, sha) == dest
        with pytest.raises(ValueError, match="checksum|SHA-256"):
            download_file("http://unused.invalid", dest, "0" * 64)
        with pytest.raises(ValueError, match="size"):
            download_file("http://unused.invalid", dest, None, expected_size=len(payload) + 1)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_download_requires_size_when_no_published_checksum(tmp_path):
    with pytest.raises(ValueError, match="size|checksum"):
        download_file("http://unused.invalid", tmp_path / "x.csd", None)
