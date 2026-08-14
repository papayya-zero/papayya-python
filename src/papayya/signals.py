"""``papayya.signal()`` — tell Papayya the world said a record was wrong.

One line, wired into whatever already knows: a thumbs-down handler, a refund
webhook, a support ticket hook, the code path where a human edits your agent's
output.

    import papayya

    def on_thumbs_down(order_id):
        papayya.signal(order_id, agent="triage")

It is keyed on **your** id — the ``item_id`` you declared when you submitted the
work — because that is the only id the handler holds. Papayya resolves it to
whichever record answered for that order, including after a re-drive.

Plan 43 B2b, ADR 0009 D3/D8/D9.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock
from typing import Any

import httpx

from papayya.api import resolve_config

log = logging.getLogger("papayya.signal")

# The closed sets the server CHECKs. Mirrored here so a typo raises at the call
# site instead of being logged from a background thread where nobody reads it —
# same reasoning as the missing-API-key rule below, one class of bug lower.
VERDICTS = ("good", "bad", "corrected")
SOURCES = ("dashboard", "thumbs", "ticket", "edit", "refund", "rejection")

# ONE ATTEMPT, SUB-SECOND, AND DELIBERATELY NOT APIClient._request.
#
# This runs inside a user-facing handler — someone clicked thumbs-down and is
# waiting for a page. The shared client's retry policy was measured against a
# control plane that is not answering: a dead port blocks 1.52 s, a black-hole IP
# at timeout=2.0 blocks 11.5 s, and at the default timeout the policy is ~151 s
# of blocking. None of that is acceptable here, and none of it helps: a signal is
# evidence, not a transaction, and a lost one is better than a hung checkout.
#
# Measured with the shape below: dead port 0.001 s, black-hole IP 0.753 s,
# NXDOMAIN 0.016 s — and the CALLER returns in 0.0007 s, because the send is
# off-thread.
_TIMEOUT_SECONDS = 0.75

_lock = Lock()
_executor: ThreadPoolExecutor | None = None
_clients: dict[tuple[str, str], httpx.Client] = {}


def _dispatcher() -> ThreadPoolExecutor:
    """The send pool, created on first use.

    NON-DAEMON THREADS ON PURPOSE. concurrent.futures registers an atexit hook
    that joins them, so a script that signals and exits on the next line still
    delivers (verified: the request lands). Daemon threads would drop it, which
    is the one failure mode a customer would never suspect — their handler
    returned successfully.

    Two workers: enough that a slow send does not serialise a burst of
    thumbs-downs, small enough that a control plane which has stopped answering
    cannot grow a thread per click.
    """
    global _executor
    with _lock:
        if _executor is None:
            _executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="papayya-signal"
            )
        return _executor


def _client(base_url: str, api_key: str) -> httpx.Client:
    """One pooled client per credential, so repeated signals reuse a connection
    rather than paying a TLS handshake per thumbs-down."""
    key = (base_url, api_key)
    with _lock:
        client = _clients.get(key)
        if client is None:
            headers = {"Accept": "application/json"}
            if api_key.startswith("cpk_"):
                headers["X-Api-Key"] = api_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
            client = httpx.Client(
                base_url=base_url, timeout=_TIMEOUT_SECONDS, headers=headers
            )
            _clients[key] = client
        return client


def _send(client: httpx.Client, payload: dict[str, Any]) -> dict[str, Any] | None:
    """The whole background half: one request, no retry, nothing raised.

    What it logs is graded, because the two failures are not the same thing. A
    connection error is the network and may fix itself — WARNING. A 4xx is the
    CALLER'S BUG (a verdict the server does not accept, an agent slug that does
    not exist), it will fail identically forever, and off-thread there is nowhere
    else to say so — ERROR.
    """
    try:
        resp = client.post("/v1/durable/signals", json=payload)
    except Exception as exc:  # noqa: BLE001 — a signal never breaks the caller
        log.warning(
            "signal not delivered for item_id=%r: %s", payload.get("item_id"), exc
        )
        return None

    if resp.is_success:
        return resp.json()
    if 400 <= resp.status_code < 500:
        log.error(
            "signal REJECTED for item_id=%r: HTTP %s %s",
            payload.get("item_id"), resp.status_code, resp.text.strip(),
        )
    else:
        log.warning(
            "signal not delivered for item_id=%r: HTTP %s",
            payload.get("item_id"), resp.status_code,
        )
    return None


def signal(
    item_id: str,
    *,
    agent: str,
    verdict: str = "bad",
    source: str = "thumbs",
    reason: str | None = None,
    external_id: str | None = None,
    occurred_at: datetime | str | None = None,
    partition_key: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Future:
    """Record that the world judged one of your records.

    ``item_id`` is *your* id for the record — the one you declared on submit.
    ``agent`` is the agent that produced it. The defaults describe the common
    case, a thumbs-down, so the minimal call is one line::

        papayya.signal(order.id, agent="triage")

    Returns immediately; the send happens on a background thread. **Nothing about
    delivery raises** — a control plane that is down must not break a customer's
    checkout. Two things do raise, both on the calling thread, both because they
    are bugs in the call rather than in the network:

    * no API key (``PapayyaAPIError``) — otherwise a whole deployment signals
      into the void and looks fine doing it;
    * a ``verdict`` or ``source`` outside the closed set (``ValueError``).

    :param verdict: ``good`` / ``bad`` / ``corrected``. ``good`` is stored and
        currently unread by any surface — an accumulated suite needs the passes
        as well as the failures.
    :param source: where the judgement came from — ``thumbs``, ``ticket``,
        ``edit``, ``refund``, ``rejection``, ``dashboard``.
    :param external_id: your own id **for the signal** (a ticket id, a webhook
        event id) — not for the record. It is the idempotency key: send one and
        a retried webhook stores one row. Without it, two calls store two rows,
        deliberately — one record accumulates signals, and two people
        complaining about one order is two pieces of evidence, not a duplicate.
    :param occurred_at: when the world judged, if that is not now. It is what
        decides *which* record the signal names when an order was re-driven.
    :param partition_key: required if you submitted the work under one. A signal
        with no partition names only unpartitioned records — it will not guess
        across your tenants.
    :returns: a ``Future``. Ignore it in production; tests call ``.result()``.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")

    # ON THE CALLING THREAD, so a missing key raises where the customer can see
    # it. A configuration failure swallowed like a delivery failure is silent
    # forever — the difference between "we dropped one signal" and "we have
    # never stored any" (D9).
    config = resolve_config(api_key=api_key, base_url=base_url)

    if isinstance(occurred_at, datetime):
        stamp = occurred_at.astimezone(timezone.utc).isoformat()
    else:
        stamp = occurred_at  # already a string, or None for "now", server-side

    payload: dict[str, Any] = {
        "agent": agent,
        "item_id": item_id,
        "verdict": verdict,
        "source": source,
    }
    for key, value in (
        ("reason", reason),
        ("external_id", external_id),
        ("occurred_at", stamp),
        ("partition_key", partition_key),
    ):
        if value is not None:
            payload[key] = value

    client = _client(config.base_url, config.api_key)
    return _dispatcher().submit(_send, client, payload)
