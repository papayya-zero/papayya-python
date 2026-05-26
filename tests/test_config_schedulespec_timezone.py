"""Tests for the ``timezone`` field on ScheduleSpec (Plan 11 decision D6).

The field is additive — existing papayya.yaml files that omit timezone
continue to parse, and pydantic round-trips the value verbatim.
"""

from __future__ import annotations

import yaml

from papayya._config import PapayyaYaml, ScheduleSpec


def test_schedulespec_defaults_to_utc():
    spec = ScheduleSpec(cron="0 * * * *")
    assert spec.timezone == "UTC"


def test_schedulespec_round_trips_explicit_timezone():
    spec = ScheduleSpec(cron="0 * * * *", timezone="America/Toronto")
    assert spec.timezone == "America/Toronto"
    # Pydantic dump preserves the value.
    dumped = spec.model_dump()
    assert dumped["timezone"] == "America/Toronto"


def test_legacy_yaml_without_timezone_loads():
    """An existing papayya.yaml that pre-dates Plan 11 (no timezone key
    on schedules) must continue to load — the timezone default makes
    the addition backwards-compatible."""
    yaml_text = """
version: 1
envs:
  prod:
    agents:
      ingest:
        schedules:
          - cron: "0 * * * *"
"""
    parsed = PapayyaYaml.model_validate(yaml.safe_load(yaml_text))
    schedule = parsed.envs["prod"].agents["ingest"].schedules[0]
    assert schedule.cron == "0 * * * *"
    assert schedule.timezone == "UTC"  # default


def test_yaml_with_explicit_timezone_loads():
    yaml_text = """
version: 1
envs:
  prod:
    agents:
      reports:
        schedules:
          - cron: "0 9 * * MON-FRI"
            timezone: "America/Toronto"
"""
    parsed = PapayyaYaml.model_validate(yaml.safe_load(yaml_text))
    schedule = parsed.envs["prod"].agents["reports"].schedules[0]
    assert schedule.timezone == "America/Toronto"
