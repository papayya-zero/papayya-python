"""Unit 2: single calling convention.

run.step has three legacy call shapes; only the canonical
``run.step('label', fn)`` should be silent. The other two — fn-only
(label-from-__name__) and decorator — emit DeprecationWarning and
will be removed in the next minor release.
"""

from __future__ import annotations

import pytest

from papayya.durable.run import PapayyaRun
from papayya.durable.types import DurableRunConfig


def _run() -> PapayyaRun:
    return PapayyaRun(DurableRunConfig(agent="test-agent"))


class TestCanonicalIsSilent:
    def test_run_step_label_fn_no_warning(self, recwarn) -> None:
        run = _run()

        def search() -> int:
            return 1

        run.step("search", search)()

        deprecations = [
            w
            for w in recwarn.list
            if issubclass(w.category, DeprecationWarning)
            and "run.step" in str(w.message)
        ]
        assert not deprecations, "canonical run.step should not warn"


class TestFnOnlyDeprecated:
    def test_run_step_fn_only_warns(self, recwarn) -> None:
        run = _run()

        def search() -> int:
            return 1

        run.step(search)()

        deprecations = [
            w for w in recwarn.list if issubclass(w.category, DeprecationWarning)
        ]
        assert deprecations
        msg = str(deprecations[0].message)
        assert "fn-only" in msg or "fn.__name__" in msg

    def test_lambda_still_raises(self) -> None:
        """Anonymous functions can't have a label derived; the existing
        ValueError stays — the deprecation doesn't change error semantics."""
        run = _run()
        with pytest.raises(ValueError, match="explicit label"):
            run.step(lambda: 1)


class TestDecoratorDeprecated:
    def test_decorator_warns(self, recwarn) -> None:
        run = _run()

        @run.step("search")
        def search() -> int:
            return 1

        search()

        deprecations = [
            w for w in recwarn.list if issubclass(w.category, DeprecationWarning)
        ]
        assert deprecations
        msg = str(deprecations[0].message)
        assert "decorator" in msg


class TestDedupePerLabel:
    def test_repeated_legacy_call_only_warns_once_per_label(
        self, recwarn
    ) -> None:
        run = _run()

        def search() -> int:
            return 1

        run.step(search)()
        run.step(search)()
        run.step(search)()

        # Filter to fn-only deprecations (kind='llm' could be a separate stream).
        fn_only = [
            w
            for w in recwarn.list
            if issubclass(w.category, DeprecationWarning)
            and "fn-only" in str(w.message)
        ]
        assert len(fn_only) == 1
