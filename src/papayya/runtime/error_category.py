"""Classify a customer-code exception into the closed category set (plan 50).

Plan 48 R3 found ``runtime_completed.error_category`` empty on every crashed
record, and the cause was not a bug — the worker sets a category on exactly
four paths (timeout, version_not_found, recycle_pending, paused) and the
generic ``except Exception`` that catches the customer's own failures set none.
That is the common case, so in practice the field was never populated.

**The set is closed on purpose.** This is the field cross-record clustering
groups on; the message beside it is not, because a raw exception message is
customer data and would turn one tenant's payload into another tenant's group
label. Free text here would fragment every bucket it touched, so an
unrecognized value is not passed through — the server normalizes it to
``unknown``, which is a fact rather than a guess.

Classification is by the exception's defining MODULE, not its type name or its
message. A provider that renames ``APIError`` keeps its module; a customer who
defines their own ``APIError`` keeps theirs. Matching on the name would
conflate them.
"""

from __future__ import annotations

__all__ = ["classify_exception", "CUSTOMER_CODE", "PROVIDER_ERROR", "RUNTIME", "DEPS"]

CUSTOMER_CODE = "customer_code"
PROVIDER_ERROR = "provider_error"
RUNTIME = "runtime"
DEPS = "deps"

# Top-level modules whose exceptions mean "the provider failed", not "your code
# is wrong". The remedy differs — retry or check the provider status page,
# rather than read your own function — which is the only reason the split earns
# its keep.
_PROVIDER_ROOTS = frozenset(
    {
        "openai",
        "anthropic",
        "httpx",
        "httpcore",
        "google",  # google.genai / google.api_core
        "cohere",
        "mistralai",
        "boto3",
        "botocore",  # bedrock
    }
)

# papayya's own failures. Deliberately NOT the customer's: a bug in the SDK
# being reported as `customer_code` sends someone to debug a function that is
# fine.
_RUNTIME_ROOTS = frozenset({"papayya"})


def _root_module(exc: BaseException) -> str:
    """Top-level module that DEFINES this exception's type.

    ``type(exc).__module__`` is where the class was declared, which is what we
    want — an ``openai.APIError`` raised from inside customer code is still an
    openai error. Builtins report ``builtins``.
    """
    module = getattr(type(exc), "__module__", "") or ""
    return module.split(".", 1)[0]


def classify_exception(exc: BaseException) -> str:
    """Category for an exception that escaped the customer's ``@agent`` body.

    Defaults to :data:`CUSTOMER_CODE`. That default is the honest one: the
    exception escaped their function, and absent positive evidence of a
    provider or SDK fault, their code is where it came from. ``KeyError:
    'body'`` — the failure plan 48 was built around — lands here, correctly.
    """
    root = _root_module(exc)
    if root in _PROVIDER_ROOTS:
        return PROVIDER_ERROR
    if root in _RUNTIME_ROOTS:
        return RUNTIME
    # A missing import is the hosted pool's dependency limit showing up at run
    # time (ADR 0010), not the customer writing a bad import — the deploy-time
    # preflight is supposed to catch it first, and when it doesn't, saying
    # `deps` sends someone to the right place.
    if isinstance(exc, ModuleNotFoundError):
        return DEPS
    return CUSTOMER_CODE
