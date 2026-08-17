"""The deploy-time dependency preflight (ADR 0010).

The property under test is not "the parser works" — it is that the set the
preflight checks against is the set the worker image installs. Those are the
same object (papayya's own metadata: core deps + the `runtime` extra), and
these tests pin that, because the failure mode of drift is a deploy that passes
preflight and then dies on `import` at run time — which is plan 47 S2 back
again, wearing a green check.
"""

import pytest

from papayya.runtime.baked_deps import (
    baked_distributions,
    normalize,
    unsupported_requirements,
)


def test_baked_set_includes_the_runtime_extra_and_core_deps():
    baked = baked_distributions()
    assert baked is not None, "papayya metadata must be readable from an installed SDK"
    # From the `runtime` extra — what the worker image adds.
    assert "openai" in baked
    assert "anthropic" in baked
    assert "tiktoken" in baked
    # From core `dependencies` — installed with the SDK, so present in the
    # image too. A customer declaring these is not asking for anything new.
    assert "httpx" in baked
    assert "pydantic" in baked
    # The SDK itself.
    assert "papayya" in baked


def test_extras_that_are_not_in_the_image_stay_out():
    baked = baked_distributions()
    # `test` and `otel` are real extras on the same package. Including them
    # would make the preflight pass a deploy the pool cannot serve.
    assert "pytest" not in baked
    assert "opentelemetry-api" not in baked


@pytest.mark.parametrize(
    "line",
    [
        "openai",
        "openai>=1.0",
        "openai==1.2.3",
        "openai >= 1.0  # the provider",
        "OpenAI",
        "PyYAML",  # case-folds onto the core dep `pyyaml`
    ],
)
def test_available_requirement_shapes_pass(line):
    # Pins are parsed for the NAME only. We do not claim to satisfy a version:
    # asserting `openai>=99` is fine because the image has *an* openai would be
    # a worse lie than the one this check removes.
    assert unsupported_requirements(line) == []


def test_unavailable_requirement_is_reported_with_its_own_line():
    assert unsupported_requirements("pandas>=2.0") == ["pandas>=2.0"]


def test_mixed_manifest_reports_only_what_is_missing():
    manifest = "\n".join(
        [
            "# our deps",
            "openai>=1.0",
            "",
            "pandas>=2.0",
            "-r dev-requirements.txt",
            "--index-url https://example.invalid/simple",
            "scikit-learn",
        ]
    )
    assert unsupported_requirements(manifest) == ["pandas>=2.0", "scikit-learn"]


def test_extras_syntax_resolves_on_the_base_name():
    # `httpx[http2]` is httpx; the image has httpx. The extra may not be
    # installed, which is a pin-shaped claim we explicitly do not make.
    assert unsupported_requirements("httpx[http2]>=0.27") == []


def test_comment_and_blank_lines_are_not_requirements():
    assert unsupported_requirements("\n\n# nothing here\n   \n") == []


def test_unnamed_url_requirement_is_reported_rather_than_silently_passed():
    # The pool cannot fetch this, and a VCS URL with no leading name gives us
    # no distribution to check. Reporting it is the truthful answer.
    assert unsupported_requirements("git+https://example.invalid/x.git") == [
        "git+https://example.invalid/x.git"
    ]


def test_normalize_follows_pep503():
    assert normalize("Foo.Bar_baz") == "foo-bar-baz"
    assert normalize("  OpenAI  ") == "openai"
    # Separators collapse to '-', they are NOT stripped: `open_ai` is a
    # different distribution from `openai` and must not be treated as baked.
    assert normalize("open_ai") == "open-ai"
    assert unsupported_requirements("open_ai") == ["open_ai"]
