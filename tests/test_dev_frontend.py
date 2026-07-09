"""Smoke tests for Slice 4 — frontend serving.

The HTML pages themselves are too small to need DOM-level testing; the
interactive behaviour lives in ``app.js``, and the JSON endpoints are
covered in ``test_dev_server.py``. What this file guards is:

1. Every clean page URL (``/batches``, ``/batch``, etc.) returns a 200
   with HTML — so a user refreshing the page doesn't see a 404.
2. Shared assets (``/style.css``, ``/app.js``) are served with the right
   content type — a broken MIME on app.js silently kills every page.
3. The static dir contains only hand-authored files (no bundler output).
"""

from __future__ import annotations

import socket
import threading
import urllib.request
import urllib.error
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import pytest

from papayya.dev.server import STATIC_DIR, DevHandler
from papayya.durable.sqlite_store import SQLiteStore


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server(tmp_path: Path) -> Iterator[str]:
    db_path = tmp_path / "local.db"
    SQLiteStore(str(db_path)).close()  # fresh empty DB

    DevHandler.db_path = str(db_path)
    port = _free_port()
    srv = ThreadingHTTPServer(("127.0.0.1", port), DevHandler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()
        srv.server_close()


def _get(base: str, path: str) -> urllib.request.http.client.HTTPResponse:
    return urllib.request.urlopen(base + path, timeout=5)


class TestCleanUrlRouting:
    @pytest.mark.parametrize("path", [
        "/", "/agents", "/runs", "/run", "/items", "/item", "/record",
        "/search", "/cost", "/upgrade",
        # Legacy paths (one release): old bookmarks keep resolving.
        "/batches", "/batch",
    ])
    def test_page_returns_html(self, server: str, path: str) -> None:
        resp = _get(server, path)
        assert resp.status == 200
        assert resp.headers.get("Content-Type", "").startswith("text/html")
        body = resp.read().decode()
        assert "<body" in body
        # Every real page (everything except index.html's redirect) must
        # set a data-page attribute the dispatcher keys on.
        if path not in ("/",):
            assert "data-page=" in body

    def test_nav_speaks_new_nouns(self, server: str) -> None:
        """Plan 34 Unit 3: nav is Agents → Runs → Items; no page says
        'Batches' anywhere."""
        body = _get(server, "/runs").read().decode()
        for nav in ("Agents", "Runs", "Items"):
            assert f">{nav}</a>" in body
        assert "Batches" not in body
        assert "<title>Runs · Papayya Dev</title>" in body

    def test_legacy_batches_path_serves_runs_page(self, server: str) -> None:
        body = _get(server, "/batches").read().decode()
        assert 'data-page="runs"' in body


class TestStaticAssets:
    def test_css_served_with_correct_mime(self, server: str) -> None:
        resp = _get(server, "/style.css")
        assert resp.status == 200
        assert resp.headers.get("Content-Type", "").startswith("text/css")

    def test_js_served_with_correct_mime(self, server: str) -> None:
        resp = _get(server, "/app.js")
        assert resp.status == 200
        assert resp.headers.get("Content-Type", "").startswith("application/javascript")

    def test_unknown_static_falls_back_to_index(self, server: str) -> None:
        resp = _get(server, "/does-not-exist.html")
        assert resp.status == 200
        assert resp.headers.get("Content-Type", "").startswith("text/html")


class TestStaticDirHygiene:
    """Enforce the zero-build contract: only hand-authored files ship."""

    def test_no_bundler_artifacts(self) -> None:
        allowed = {".html", ".js", ".css", ".svg", ".ico", ".png"}
        for path in STATIC_DIR.iterdir():
            if path.is_file():
                assert path.suffix in allowed or path.name.startswith("_"), (
                    f"unexpected artifact {path.name}; "
                    "static dir must stay zero-build"
                )
            else:
                raise AssertionError(f"unexpected directory {path.name}")

    def test_no_package_json(self) -> None:
        assert not (STATIC_DIR / "package.json").exists()
        assert not (STATIC_DIR / "node_modules").exists()

    def test_bundle_size_reasonable(self) -> None:
        total = sum(f.stat().st_size for f in STATIC_DIR.iterdir() if f.is_file())
        # Arbitrary ceiling — today's total is well under this. The check
        # catches someone accidentally vendoring a library.
        assert total < 200_000, f"static dir is {total} bytes; keep it slim"
