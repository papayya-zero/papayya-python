"""Structural outcome inspectors.

Pure functions that look at a task's return value (and optionally the LLM
usage extracted from it) and decide whether the outcome is 'ok' or
'degraded'. Used by durable/run.py to populate TaskEntry.outcome_status
without any customer code change.

Each inspector is independent. The orchestrator (inspect_result) runs them
in order and returns the first 'degraded' verdict; if all return 'ok',
the overall outcome is 'ok'.

Plan 40 Unit 1 adds the first inspector that looks at **two** values —
:func:`inspect_conservation` relates a step's output back to its input.
Emptiness is decidable from the artifact alone; "half-empty" is not, so
the truth has to be imported from somewhere else. Cheapest source is the
input the step was handed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

log = logging.getLogger("papayya.outcomes")


# Module-level switch. Reading the value at integration time in
# _post_call_success lets tests monkey-patch this to disable structural
# detection without needing a YAML/config schema. A future plan will
# replace this with proper configuration.
ENABLE_STRUCTURAL_DETECTION: bool = True


@dataclass(frozen=True)
class OutcomeVerdict:
    status: str           # 'ok' | 'degraded'
    reason: str | None    # short token; None when status == 'ok'


OK = OutcomeVerdict("ok", None)


def inspect_empty(result: Any) -> OutcomeVerdict:
    """Flag absent/empty results as degraded.

    Numeric ``0`` / ``0.0`` are legitimate outputs, not degraded.
    """
    if result is None:
        return OutcomeVerdict("degraded", "empty_none")
    if result is False:
        return OutcomeVerdict("degraded", "empty_false")
    if isinstance(result, (str, bytes)) and len(result) == 0:
        return OutcomeVerdict("degraded", "empty_string")
    if isinstance(result, (list, tuple)) and len(result) == 0:
        return OutcomeVerdict("degraded", "empty_sequence")
    if isinstance(result, dict) and len(result) == 0:
        return OutcomeVerdict("degraded", "empty_dict")
    return OK


def _all_zero_numeric_sequence(seq: Any) -> bool:
    """True iff ``seq`` is a non-empty sequence of numbers all equal to 0.

    Conservative: returns False for empty sequences (empty is flagged by
    inspect_empty, not here) and for any element that isn't a plain
    ``int`` / ``float``.
    """
    if not isinstance(seq, (list, tuple)):
        return False
    if len(seq) == 0:
        return False
    for x in seq:
        if isinstance(x, bool):
            return False
        if not isinstance(x, (int, float)):
            return False
        if x != 0:
            return False
    return True


def inspect_degenerate_embedding(result: Any) -> OutcomeVerdict:
    """Flag zero embeddings (all-zero vectors) as degraded.

    Looks at three shapes:
      - A bare list/tuple of numbers.
      - A dict with an ``"embedding"`` or ``"embeddings"`` key holding the
        same shape.
      - Anything else (including numpy arrays we can't cheaply introspect)
        falls through to OK.
    """
    if _all_zero_numeric_sequence(result):
        return OutcomeVerdict("degraded", "degenerate_embedding")
    if isinstance(result, dict):
        for key in ("embedding", "embeddings"):
            if key in result and _all_zero_numeric_sequence(result[key]):
                return OutcomeVerdict("degraded", "degenerate_embedding")
    return OK


_DEGENERATE_STOP_REASONS = frozenset({"length", "content_filter", "refusal", "error"})


def inspect_llm_stop_reason(usage: Any) -> OutcomeVerdict:
    """Flag degenerate LLM stop reasons as degraded.

    ``usage`` is expected to be an :class:`~papayya.llm_extract.LlmUsage`
    or ``None`` when the step wasn't LLM-shaped. The function duck-types
    on ``.stop_reason`` so non-LLM callers can pass ``None`` safely.
    """
    if usage is None:
        return OK
    stop_reason = getattr(usage, "stop_reason", None)
    if stop_reason in _DEGENERATE_STOP_REASONS:
        return OutcomeVerdict("degraded", f"llm_stop_reason:{stop_reason}")
    return OK


# --------------------------------------------------------------------- #
#  Conservation contracts (Plan 40 Unit 1)                              #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class ConservationContract:
    """What a step declares its output must conserve about its input.

    Built by ``run.step(..., expect_count=/expect_coverage=/expect_fields=)``;
    absent by default. Absence means no check — a step that legitimately
    drops rows is normal and must never false-positive (Plan 40 D4), so
    there is no global "outputs must equal inputs" inference anywhere.

    Fields:

    * ``count`` — declared output cardinality. An ``int`` (preferred: the
      number is usually already in a variable three lines up) or a callable
      invoked with the step's bound arguments as keywords.
    * ``coverage`` — a record-id key name. Every input record carrying that
      key must appear in the output under the same key.
    * ``fields`` — keys that must be present and non-blank on every output
      record.
    """

    count: int | Callable[..., int] | None = None
    coverage: str | None = None
    fields: tuple[str, ...] | None = None


def build_contract(
    *,
    expect_count: int | Callable[..., int] | None = None,
    expect_coverage: str | None = None,
    expect_fields: Sequence[str] | str | None = None,
) -> ConservationContract | None:
    """Normalize the ``expect_*`` step kwargs into a contract, or ``None``.

    Lenient by construction: a malformed declaration is logged and dropped
    rather than raised. Steps get wired up inside live runs, and the
    observer invariant (a check never fails the run) has to hold at
    declaration time too.
    """
    if expect_count is None and expect_coverage is None and expect_fields is None:
        return None

    count: int | Callable[..., int] | None = None
    if expect_count is not None:
        if callable(expect_count):
            count = expect_count
        else:
            as_int = _as_int(expect_count)
            if as_int is None:
                log.warning(
                    "papayya: expect_count=%r is neither an int nor a callable; "
                    "ignoring the cardinality contract",
                    expect_count,
                )
            count = as_int

    coverage: str | None = None
    if expect_coverage is not None:
        if isinstance(expect_coverage, str) and expect_coverage:
            coverage = expect_coverage
        else:
            log.warning(
                "papayya: expect_coverage=%r is not a record-id key name; "
                "ignoring the coverage contract",
                expect_coverage,
            )

    fields: tuple[str, ...] | None = None
    if expect_fields is not None:
        # A bare string is the obvious single-field shorthand, not an
        # iterable of one-character field names.
        names = (expect_fields,) if isinstance(expect_fields, str) else tuple(expect_fields)
        fields = tuple(n for n in names if isinstance(n, str) and n) or None
        if fields is None:
            log.warning(
                "papayya: expect_fields=%r holds no field names; "
                "ignoring the field-presence contract",
                expect_fields,
            )

    if count is None and coverage is None and fields is None:
        return None
    return ConservationContract(count=count, coverage=coverage, fields=fields)


def _as_int(value: Any) -> int | None:
    """Coerce via ``__index__`` so numpy/other integer types work; ``None``
    for anything that isn't an integer. ``bool`` is rejected — ``True`` as a
    declared cardinality is always a typo."""
    if isinstance(value, bool):
        return None
    try:
        return value.__index__()
    except (AttributeError, TypeError, ValueError):
        return None


def _is_blank(value: Any) -> bool:
    """Emptiness test for a *field* value, matching :func:`inspect_empty`'s
    definition: numeric ``0`` is a legitimate value, not a blank."""
    if value is None or value is False:
        return True
    if isinstance(value, (str, bytes, list, tuple, dict, set, frozenset)):
        return len(value) == 0
    return False


def _output_cardinality(result: Any) -> int | None:
    """How many records ``result`` holds, or ``None`` when undecidable.

    ``str``/``bytes`` are excluded: their length is characters, and a step
    declaring a cardinality never means "500 characters". Anything else
    that answers ``len()`` counts — that covers dicts keyed by record id
    plus DataFrames and arrays we can't otherwise introspect.
    """
    if isinstance(result, (str, bytes)):
        return None
    try:
        return len(result)
    except TypeError:
        return None


def _records_of(result: Any) -> list | None:
    """Normalize a step result to a list of records, or ``None``.

    A bare ``dict`` is ONE record (the common extraction shape), not a map
    of records — a step returning many records returns a sequence of them.
    """
    if isinstance(result, dict):
        return [result]
    if isinstance(result, (list, tuple)):
        return list(result)
    return None


def _field_of(record: Any, key: str) -> Any:
    """Read ``key`` off a dict record or an attribute off an object one."""
    if isinstance(record, dict):
        return record.get(key)
    return getattr(record, key, None)


def _ids_of(records: Sequence[Any], key: str) -> set | None:
    """Collect the ``key`` value from every record, or ``None`` when no
    record carries one (the caller then treats the relation as undeclared)."""
    ids = set()
    for record in records:
        value = _field_of(record, key)
        if value is None:
            continue
        try:
            ids.add(value)
        except TypeError:  # unhashable id — can't relate the two sides
            return None
    return ids or None


def _find_input_collection(inputs: dict[str, Any], key: str) -> list | None:
    """Find the bound argument that is a collection of records carrying ``key``.

    Answers Plan 40's open question 2: coverage does not need the customer
    to hand us the id list separately — naming the id key is enough to
    locate the input side, because the input the step was called with is
    already bound.
    """
    for value in inputs.values():
        if not isinstance(value, (list, tuple)) or not value:
            continue
        if any(_field_of(record, key) is not None for record in value):
            return list(value)
    return None


def _inspect_count(
    result: Any, expected: int | Callable[..., int], inputs: dict[str, Any]
) -> OutcomeVerdict:
    if callable(expected):
        # The callable form is the escape hatch; it receives the step's
        # bound arguments as keywords, so its parameter names must match
        # the step's (or it takes **kwargs).
        expected_value = _as_int(expected(**inputs))
    else:
        expected_value = expected
    if expected_value is None:
        log.warning("papayya: expect_count callable returned a non-integer; treating as a pass")
        return OK
    actual = _output_cardinality(result)
    if actual is None:
        log.warning(
            "papayya: expect_count declared but the result has no countable "
            "length (%s); treating as a pass",
            type(result).__name__,
        )
        return OK
    if actual < expected_value:
        return OutcomeVerdict("degraded", "conservation:count_short")
    if actual > expected_value:
        return OutcomeVerdict("degraded", "conservation:count_over")
    return OK


def _inspect_coverage(result: Any, key: str, inputs: dict[str, Any]) -> OutcomeVerdict:
    source = _find_input_collection(inputs, key)
    if source is None:
        log.warning(
            "papayya: expect_coverage=%r found no input collection carrying that "
            "key; treating as a pass",
            key,
        )
        return OK
    expected_ids = _ids_of(source, key)
    if not expected_ids:
        return OK
    produced = _records_of(result)
    if produced is None:
        log.warning(
            "papayya: expect_coverage declared but the result isn't a record or "
            "sequence of records (%s); treating as a pass",
            type(result).__name__,
        )
        return OK
    produced_ids = _ids_of(produced, key) or set()
    if expected_ids - produced_ids:
        return OutcomeVerdict("degraded", "conservation:coverage")
    return OK


def _inspect_fields(result: Any, fields: tuple[str, ...]) -> OutcomeVerdict:
    records = _records_of(result)
    if records is None:
        log.warning(
            "papayya: expect_fields declared but the result isn't a record or "
            "sequence of records (%s); treating as a pass",
            type(result).__name__,
        )
        return OK
    # Fields outer, records inner: the reported field is the first DECLARED
    # one that's missing anywhere, so the reason token doesn't depend on the
    # order records happen to arrive in.
    for field in fields:
        for record in records:
            if _is_blank(_field_of(record, field)):
                return OutcomeVerdict("degraded", f"conservation:field:{field}")
    return OK


def inspect_conservation(
    result: Any,
    contract: ConservationContract | None,
    inputs: dict[str, Any] | None = None,
) -> OutcomeVerdict:
    """Relate a step's output to the input it was handed.

    Evaluates the declared relations smallest-first — cardinality, then
    coverage, then the field-presence floor — and returns the first
    degraded verdict. Reasons are namespaced ``conservation:`` so the
    dashboard's reason histogram groups them (Plan 40 D2).

    Observer, like every other inspector (D3): it marks degraded, it never
    raises. A contract that blows up or can't be evaluated against this
    result is logged and treated as a pass. Escalation is the Plan 33
    fence's job.
    """
    if contract is None:
        return OK
    inputs = inputs or {}

    if contract.count is not None:
        try:
            verdict = _inspect_count(result, contract.count, inputs)
        except Exception:
            log.exception("papayya: expect_count raised; treating as a pass (observer)")
        else:
            if verdict.status != "ok":
                return verdict

    if contract.coverage is not None:
        try:
            verdict = _inspect_coverage(result, contract.coverage, inputs)
        except Exception:
            log.exception("papayya: expect_coverage raised; treating as a pass (observer)")
        else:
            if verdict.status != "ok":
                return verdict

    if contract.fields is not None:
        try:
            verdict = _inspect_fields(result, contract.fields)
        except Exception:
            log.exception("papayya: expect_fields raised; treating as a pass (observer)")
        else:
            if verdict.status != "ok":
                return verdict

    return OK


def inspect_result(
    result: Any,
    *,
    usage: Any = None,
    contract: ConservationContract | None = None,
    inputs: dict[str, Any] | None = None,
) -> OutcomeVerdict:
    """Run all inspectors and return the first degraded verdict.

    Order: stop-reason first (so LLM-shape signal wins over empty/zero
    checks on the response object), then conservation, then empty, then
    degenerate embedding. Returns :data:`OK` when all inspectors pass.

    Conservation outranks empty because a declared contract is the most
    specific signal available: for a step declaring ``expect_count=500``,
    ``conservation:count_short`` says strictly more than ``empty_sequence``
    does. Steps without a contract are unaffected — the order they see is
    unchanged.
    """
    verdict = inspect_llm_stop_reason(usage)
    if verdict.status != "ok":
        return verdict
    verdict = inspect_conservation(result, contract, inputs)
    if verdict.status != "ok":
        return verdict
    verdict = inspect_empty(result)
    if verdict.status != "ok":
        return verdict
    verdict = inspect_degenerate_embedding(result)
    if verdict.status != "ok":
        return verdict
    return OK


# ---------------------------------------------------------------------------
# D6 taint — provenance, not a verdict
# ---------------------------------------------------------------------------

# Reasons whose result value is not worth propagating. `empty_none` is here on
# the control pane's own existing judgement (`CountedDegradedSteps`, plan 41 R7):
# a step returning None is "overwhelmingly a void side effect (send_email,
# write_row)", which is why it is already excluded from re-execution. Tainting
# off a value the platform has declared meaningless would be backwards.
_NON_PROPAGATING_REASONS = frozenset({"empty_none"})


def _identity_is_meaningful(value: Any) -> bool:
    """True when ``x is value`` distinguishes this object from an unrelated one.

    CPython interns ``None``/``True``/``False``, the empty str/bytes/tuple and
    small ints, so identity against them fires on any argument that merely
    happens to hold the same singleton. ``[]`` and ``{}`` are freshly allocated
    per call and are safe — fortunate, because the empty *collection* is the
    shape of the incident D6 exists for (a context fetch returning no rows).

    Excluding a singleton source is a deliberate false negative: an unmarked
    step, rather than a mark on a step that never saw the value.
    """
    if value is None or value is True or value is False:
        return False
    if isinstance(value, (str, bytes)) and len(value) == 0:
        return False
    if isinstance(value, tuple) and len(value) == 0:
        return False
    if isinstance(value, int) and -5 <= value <= 256:
        return False
    return True


def _flatten_one_level(bound: dict[str, Any]) -> list[Any]:
    """Bound arguments, plus one level into containers.

    ``Signature.bind`` nests variadics one level down — ``def f(*chunks)`` binds
    to ``{"chunks": (value,)}`` — so comparing against ``bound.values()`` alone
    tests the tuple and never the element. The same is true of a value the
    caller wrapped on the way in (``answer(context={"docs": docs})``).

    One level, not recursive: a value buried deeper, or transformed by a pure
    function between steps, is a miss. The alternative is an unbounded walk of
    customer data on every step of every run.
    """
    out: list[Any] = []
    for arg in bound.values():
        out.append(arg)
        if isinstance(arg, (list, tuple)):
            out.extend(arg)
        elif isinstance(arg, dict):
            out.extend(arg.values())
    return out


def inspect_taint(
    bound: dict[str, Any] | None,
    prior: Sequence[Any],
) -> tuple[str | None, str | None]:
    """ADR 0009 D6: did this step consume a degraded step's output?

    Returns ``(tainted_by, tainted_reason)`` — the **root** degraded step's
    label and its reason — or ``(None, None)``. Every step in a taint chain
    names the same root, so a blast radius is one equality predicate rather
    than a walk.

    This is **not** a verdict on this step's output; nothing looked at the
    output. It is a statement about the provenance of its input, which is why
    it lands in its own columns and leaves ``outcome_status`` alone. Every
    existing reader of ``outcome_status`` (both degraded-streak fences, the
    drift rates learned against a floor, the failure-cluster keys, the reason
    histogram) therefore sees exactly the row it sees today.

    Matching is by identity, not equality: cheap, allocation-free, and free of
    false positives from two equal-but-unrelated values. It is transitive by
    construction — a tainted step's own result is cached carrying its taint, so
    its consumers taint in turn, which is D6's "blast radius is a graph
    traversal" with no graph to build.

    ``prior`` is the run's already-completed entries (duck-typed:
    ``.outcome_status`` / ``.outcome_reason`` / ``.result`` / ``.label``).
    """
    if not bound or not prior:
        return (None, None)
    sources: list[tuple[Any, str, str | None]] = []
    for e in prior:
        if not _identity_is_meaningful(getattr(e, "result", None)):
            continue
        if getattr(e, "outcome_status", "ok") != "ok":
            reason = getattr(e, "outcome_reason", None)
            if reason in _NON_PROPAGATING_REASONS:
                continue
            # A degraded step is a ROOT: it names itself.
            sources.append((e, e.label, reason))
        elif getattr(e, "tainted_by", None) is not None:
            # An already-tainted step is a CARRIER. It passes the root
            # through rather than naming itself — transitivity does not
            # come free once taint stops being a status value, and the
            # root is what an operator groups a blast radius by. Every
            # step in a chain therefore answers the same question the
            # same way: "which bad fetch touched this?"
            sources.append((e, e.tainted_by, getattr(e, "tainted_reason", None)))
    if not sources:
        return (None, None)
    candidates = _flatten_one_level(bound)
    for entry, root_label, root_reason in sources:
        for arg in candidates:
            if arg is entry.result:
                return (root_label, root_reason)
    return (None, None)
