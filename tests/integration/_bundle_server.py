"""Fake bundle download server for ADR-0003 slice 2 integration tests.

The hosted control-pane serves
``GET /v1/runtime/bundles?agent={slug}&version={N}`` (Go side at
``control-pane/internal/runtime/dispatcher/bundles.go``); the worker
fetches from there on first item for a new ``(agent, agent_version)``
tuple. Slice 2 tests run end-to-end without standing up a real
control-pane, so this stub serves a configured tarball with the same
response headers the worker reads.

Limited to what slice 2 needs:
  • happy-path: configured (slug, version) → 200 with tarball + headers
  • miss: unknown (slug, version) → 404 (worker maps to
    ``error_category="version_not_found"``)
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse


class FakeBundleServer:
    """Minimal HTTP server serving a single bundle.

    Each test instance binds to ``127.0.0.1:0`` (kernel-assigned port)
    so multiple parallel tests don't collide. ``url`` is the base used
    by the worker's ``--bundle-url-base`` flag — it includes the
    ``/v1/runtime/bundles`` path so the worker's
    ``f"{base}?agent=...&version=..."`` resolves cleanly.
    """

    def __init__(self) -> None:
        self._bundles: dict[tuple[str, int], dict[str, Any]] = {}
        self.calls: list[tuple[str, int]] = []
        # Cap at one connection at a time — slice 2 tests are sequential.
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="papayya-fake-bundles",
            daemon=True,
        )
        self._thread.start()

    @property
    def url(self) -> str:
        host, port = self._server.server_address
        return f"http://{host}:{port}/v1/runtime/bundles"

    def register(
        self,
        *,
        agent: str,
        version: int,
        tarball: bytes,
        entrypoint: str,
        artifact_hash: str,
        account_id: str = "00000000-0000-0000-0000-000000000acc",
        agent_id: str = "00000000-0000-0000-0000-000000000011",
        deployment_id: str = "00000000-0000-0000-0000-000000000099",
    ) -> None:
        self._bundles[(agent, version)] = {
            "tarball": tarball,
            "entrypoint": entrypoint,
            "artifact_hash": artifact_hash,
            "account_id": account_id,
            "agent_id": agent_id,
            "deployment_id": deployment_id,
        }

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)

    def _make_handler(self):
        bundles = self._bundles
        calls = self.calls

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any, **_kw: Any) -> None:
                # Silence default stderr logging — tests assert on
                # subprocess log files, not the in-process server.
                pass

            def do_GET(self) -> None:  # noqa: N802 — required by stdlib
                parsed = urlparse(self.path)
                if parsed.path != "/v1/runtime/bundles":
                    self.send_error(404, "Not Found")
                    return
                qs = parse_qs(parsed.query)
                agent = qs.get("agent", [""])[0]
                version_raw = qs.get("version", [""])[0].lstrip("v")
                try:
                    version = int(version_raw)
                except ValueError:
                    self.send_error(400, "bad version")
                    return
                calls.append((agent, version))
                entry = bundles.get((agent, version))
                if entry is None:
                    self.send_error(404, "bundle not found")
                    return
                tarball = entry["tarball"]
                self.send_response(200)
                self.send_header("Content-Type", "application/gzip")
                self.send_header("Content-Length", str(len(tarball)))
                self.send_header("ETag", f'"{entry["artifact_hash"]}"')
                self.send_header("X-Papayya-Entrypoint", entry["entrypoint"])
                self.send_header("X-Papayya-Account-Id", entry["account_id"])
                self.send_header("X-Papayya-Agent-Id", entry["agent_id"])
                self.send_header("X-Papayya-Deployment-Id", entry["deployment_id"])
                self.send_header("X-Papayya-Version", str(version))
                self.end_headers()
                self.wfile.write(tarball)

        return Handler
