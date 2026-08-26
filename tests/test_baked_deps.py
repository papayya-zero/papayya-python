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


# --------------------------------------------------------------------------- #
#  Plan 69 P2 — the entrypoint's own directory is importable.
#
#  `python agent.py` puts the script's directory on sys.path[0]. The CLI's
#  loader used spec_from_file_location + exec_module and did not, so a project
#  of more than one file ran locally and could not be deployed — while the
#  WORKER resolved it fine through runtime/_bundle_loader's meta-path finder.
#  The pool could run bundles the CLI refused to upload.
# --------------------------------------------------------------------------- #
def _multifile_project(tmp_path):
    (tmp_path / "helpers.py").write_text(
        "def shout(x):\n    return {'v': str(x).upper()}\n")
    (tmp_path / "agent.py").write_text(
        "from papayya import agent\n"
        "from helpers import shout\n\n\n"
        "@agent(name='multifile', model='claude-sonnet-5')\n"
        "def multifile(run, x):\n"
        "    return run.step('s', shout, item_id=str(x))(x)\n")
    return tmp_path / "agent.py"


def test_discover_agents_resolves_a_sibling_module(tmp_path):
    from papayya.cli import _discover_agents

    agents = _discover_agents(str(_multifile_project(tmp_path)))
    assert [a.name for a in agents] == ["multifile"]


def test_discovery_restores_sys_path(tmp_path):
    import sys

    from papayya.cli import _discover_agents

    before = list(sys.path)
    _discover_agents(str(_multifile_project(tmp_path)))
    assert sys.path == before, "the project directory outlived the import"


def test_discovery_does_not_leave_a_stale_sibling_in_sys_modules(tmp_path):
    """Two projects, each with their own `helpers`, discovered in one process.

    Without the sys.modules cleanup the first `helpers` wins forever and the
    second project silently gets the first one's function — which is plan 47
    S1's shape, in the CLI instead of the worker.
    """
    import sys

    from papayya.cli import _discover_agents

    first = tmp_path / "first"
    first.mkdir()
    _discover_agents(str(_multifile_project(first)))
    assert "helpers" not in sys.modules

    second = tmp_path / "second"
    second.mkdir()
    (second / "helpers.py").write_text(
        "def shout(x):\n    return {'v': 'SECOND'}\n")
    (second / "agent.py").write_text(
        "from papayya import agent\n"
        "from helpers import shout\n\n\n"
        "@agent(name='second', model='claude-sonnet-5')\n"
        "def second(run, x):\n"
        "    return run.step('s', shout, item_id=str(x))(x)\n")
    _discover_agents(str(second / "agent.py"))

    import importlib.util
    spec = importlib.util.spec_from_file_location("_probe", second / "helpers.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    assert probe.shout("x") == {"v": "SECOND"}


def test_a_third_party_import_is_left_in_sys_modules(tmp_path):
    """Only the bundle's OWN modules are dropped.

    Evicting an unrelated package that happened to be imported for the first
    time during the exec would re-execute it on the next import for no reason.
    """
    import sys

    from papayya.cli import _discover_agents

    (tmp_path / "agent.py").write_text(
        "import json\n"
        "from papayya import agent\n\n\n"
        "@agent(name='usesjson', model='claude-sonnet-5')\n"
        "def usesjson(run, x):\n"
        "    return run.step('s', lambda v: {'v': json.dumps(v)}, "
        "item_id=str(x))(x)\n")
    _discover_agents(str(tmp_path / "agent.py"))
    assert "json" in sys.modules
