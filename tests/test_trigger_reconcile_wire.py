"""Plan 53 — the bytes `papayya deploy` puts on the wire for @schedule / @trigger.

`@schedule` and `@trigger` have never deployed. Driven:

    $ papayya deploy
      Deployed hello → 64a81358-...
    Reconciling triggers for env 'dev'...
      + schedule 0 9 * * 1-5            create
    Error (managed_by preview): HTTP 400: {"message":"invalid request body"}

The server decodes with DisallowUnknownFields, and the SDK added two fields
its request structs do not have: `managed_by` on every item of both PUTs, and
`secret_env` on the webhook probe. Confirmed against the live endpoint — the
identical body minus those fields returns the diff envelope.

WHY THE SUITE WAS GREEN. test_deploy_reconcile.py patches APIClient wholesale
and asserts `put_schedules("agt1", [{...}], dry_run=True)` was CALLED. It
never sends a byte, so the shape the server actually receives was untested on
both sides of the boundary. These tests assert the payload the transport
builds, which is the layer where the two disagreed.
"""

from __future__ import annotations

from typing import Any

import pytest

from papayya.api import APIClient, APIConfig


class _CapturingClient(APIClient):
    """APIClient with the HTTP layer replaced, so the BODY is the assertion."""

    def __init__(self):
        super().__init__(APIConfig(api_key="cpk_test", base_url="https://x.test"))
        self.sent: list[tuple[str, str, Any]] = []

    def _request(self, method: str, path: str, **kw: Any) -> Any:
        self.sent.append((method, path, kw.get("json")))
        return {"managed_by": "code", "create": [], "update": [], "delete": []}


@pytest.fixture
def api():
    return _CapturingClient()


def test_put_schedules_does_not_send_managed_by(api):
    api.put_schedules("agt1", [{"cron_expression": "0 9 * * 1-5", "timezone": "UTC"}])

    _, _, body = api.sent[-1]
    assert body == {"items": [{"cron_expression": "0 9 * * 1-5", "timezone": "UTC"}]}
    assert "managed_by" not in body["items"][0], (
        "the server's replaceSchedulesItemRequest has no managed_by and decodes "
        "with DisallowUnknownFields — this 400s every deploy of a @schedule"
    )


def test_put_webhooks_does_not_send_managed_by(api):
    api.put_webhooks("agt1", [{"name": "hook"}])

    _, _, body = api.sent[-1]
    assert body == {"items": [{"name": "hook"}]}


def test_put_schedules_passes_the_caller_s_items_through_verbatim(api):
    """Whatever the reconciler decided IS the desired state. A transport that
    edits it is a second opinion nobody asked for — which is exactly how the
    added field got there."""
    items = [{"cron_expression": "*/5 * * * *", "budget_cents": 250}]
    api.put_schedules("agt1", items)

    assert api.sent[-1][2]["items"] == items


def test_dry_run_only_changes_the_query_string(api):
    api.put_schedules("agt1", [{"cron_expression": "0 9 * * *"}], dry_run=True)
    dry_method, dry_path, dry_body = api.sent[-1]

    api.put_schedules("agt1", [{"cron_expression": "0 9 * * *"}])
    live_method, live_path, live_body = api.sent[-1]

    assert dry_body == live_body, (
        "the preview probe's own docstring promises it is byte-faithful to "
        "what the next deploy would do"
    )
    assert dry_path.endswith("?dry_run=true")
    assert not live_path.endswith("?dry_run=true")


def test_the_probe_payload_matches_what_apply_sends(monkeypatch, tmp_path):
    """THE DIVERGENCE ITSELF. `_collect_managed_diff` built the webhook probe
    as {name, secret_env}; `apply_plan` has always sent {name} alone. The
    probe's extra field is a purely LOCAL declaration — which env var holds the
    signing secret — that the control plane has no concept of."""
    from papayya import cli as cli_module
    from papayya._config import ScheduleSpec, WebhookSpec

    class _Spec:
        schedules = [ScheduleSpec(cron="0 9 * * 1-5")]
        webhooks = [WebhookSpec(name="hook", secret_env="HOOK_SECRET")]

    class _Env:
        agents = {"hello": _Spec()}

    captured: dict = {}

    class _API:
        def put_schedules(self, agent_id, items, dry_run=False):
            captured["schedules"] = items
            return {}

        def put_webhooks(self, agent_id, items, dry_run=False):
            captured["webhooks"] = items
            return {}

    cli_module._collect_managed_diff(_Env(), {"hello": "agt1"}, _API())

    assert captured["webhooks"] == [{"name": "hook"}]
    assert "secret_env" not in captured["webhooks"][0]
