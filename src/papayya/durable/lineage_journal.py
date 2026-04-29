"""Append-only JSONL sidecar for lineage writes that exhaust retries.

ADR-0002 #8. When a CloudStore POST exhausts its retry budget the SDK
appends the original request to this journal and returns successfully —
the customer's agent function does not see the transient outage. On the
next successful POST against the same control plane, the reconciler
drains the journal in FIFO order and reissues each entry, so eventually
every step row lands server-side.

The reconciler runs piggybacked on the next successful CloudStore POST.
A background daemon thread was rejected for two reasons: (1) the
SQLite-WAL-visibility regression from ADR-0002 #12 (a long-blocked
daemon thread inside the worker subprocess broke cross-process WAL
reads on macOS APFS); (2) "successful POST" is itself proof the server
is reachable, which is exactly the signal the reconciler needs.

Concurrent-process safety is best-effort. Two SDK processes sharing a
cwd may both attempt to drain the same journal; idempotency keys
(``(run_id, label)`` for save_task, ``run_id`` for set_status / create
once the server-side ``ON CONFLICT (run_id) DO NOTHING`` lands) make
duplicate drain a server-side no-op. The alternative — fcntl/flock —
is ergonomic friction for a corner case the keys already handle.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


log = logging.getLogger("papayya.durable.lineage_journal")


# Where the journal lives on disk.
_DEFAULT_JOURNAL_FILENAME = "lineage-journal.jsonl"
_DEFAULT_JOURNAL_DIR = ".papayya"
_ENV_JOURNAL_PATH = "PAPAYYA_LINEAGE_JOURNAL_PATH"
# When the worker starts, it sets PAPAYYA_LOCAL_DB_PATH to point at the
# shared SQLite store. If we're inside that worker (CloudStore wouldn't
# normally be used there, but a customer agent could opt in), placing
# the journal next to the DB keeps all SDK state co-located.
_ENV_DB_PATH = "PAPAYYA_LOCAL_DB_PATH"


@dataclass
class JournalEntry:
    """One persisted record. Each line of the JSONL file decodes to this.

    ``payload`` is the verbatim body of the original POST/PATCH — when
    the reconciler reissues, ``save_task`` payloads get
    ``delivery_attempts`` and ``journaled_at`` injected before the wire
    write so the server records the late-delivery audit on the row.
    """

    kind: str  # "create" | "save_task" | "set_status"
    method: str  # "POST" | "PATCH"
    url: str  # request path, e.g. "/v1/durable/runs/{id}/checkpoints"
    payload: dict[str, Any]
    idempotency_key: str
    first_attempt_at: str
    attempts: int
    journaled_at: str
    last_error: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json_line(cls, line: str) -> "JournalEntry":
        data = json.loads(line)
        # Tolerate older entries that lack ``extra`` (forward-compat with
        # any older format an upgraded SDK reads).
        data.setdefault("extra", {})
        return cls(**data)


def resolve_journal_path() -> Path:
    """Pick the journal file path, env override > sibling-of-DB > cwd default.

    Always returns a concrete path — does NOT create the file. The
    parent directory is created lazily on first append so a CloudStore
    that never journals leaves no footprint.
    """
    override = os.environ.get(_ENV_JOURNAL_PATH)
    if override:
        return Path(override)
    db_path = os.environ.get(_ENV_DB_PATH)
    if db_path:
        return Path(db_path).parent / _DEFAULT_JOURNAL_FILENAME
    return Path(_DEFAULT_JOURNAL_DIR) / _DEFAULT_JOURNAL_FILENAME


class LineageJournal:
    """Append-only JSONL store of CloudStore POSTs that exhausted retries.

    Operations:
    - ``append(entry)`` — atomic-ish append (POSIX guarantees small
      O_APPEND writes are atomic between processes). One JSON object
      per line, no trailing comma drama.
    - ``iter_entries()`` — read all entries in insertion order. Cheap
      enough at the journal sizes we expect (hundreds at most before a
      drain). Skips malformed lines with a WARNING — a hand-edit bug
      shouldn't poison the entire journal.
    - ``rewrite(remaining)`` — atomic replacement (tempfile + os.replace).
      Used by the reconciler after a drain attempt to keep only the
      entries that didn't deliver.
    - ``is_empty()`` — cheap stat-based check. Avoids opening + reading
      when there's no work to do.
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def is_empty(self) -> bool:
        try:
            return self._path.stat().st_size == 0
        except FileNotFoundError:
            return True

    def append(self, entry: JournalEntry) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = entry.to_json_line() + "\n"
        # O_APPEND keeps each write atomically appended at the end of
        # the file even with concurrent writers. We open + write +
        # close per call so an unhandled exception elsewhere doesn't
        # leave a dangling fd.
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())

    def iter_entries(self) -> Iterable[JournalEntry]:
        if not self._path.exists():
            return
        with open(self._path, "r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    yield JournalEntry.from_json_line(stripped)
                except (ValueError, TypeError) as exc:
                    log.warning(
                        "skipping malformed journal entry at %s:%d: %s",
                        self._path, lineno, exc,
                    )

    def rewrite(self, remaining: list[JournalEntry]) -> None:
        """Replace the journal file with exactly ``remaining`` entries.

        Atomic via tempfile-in-same-dir + ``os.replace`` (rename on the
        same filesystem is atomic on POSIX and on Windows since Python
        3.3). If ``remaining`` is empty, the file is removed entirely so
        the next ``is_empty`` check returns True without opening it.
        """
        if not remaining:
            try:
                self._path.unlink()
            except FileNotFoundError:
                pass
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # delete=False: we move the file into place ourselves; the
        # context manager just guarantees close() before rename.
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self._path.parent,
            prefix=self._path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as tf:
            for entry in remaining:
                tf.write(entry.to_json_line() + "\n")
            tf.flush()
            os.fsync(tf.fileno())
            tmp_name = tf.name
        os.replace(tmp_name, self._path)
