"""Shared error boundary for the CLI.

A single `SafeGroup` wraps every command invocation so unexpected exceptions
surface as a diagnosable one-liner instead of a cryptic `Error: <msg>` with
no type and no traceback. `PAPAYYA_DEBUG=1` prints the full traceback.
"""

from __future__ import annotations

import os
import sys
import traceback

import click

from papayya.api import PapayyaAPIError


def _debug_enabled() -> bool:
    return bool(os.environ.get("PAPAYYA_DEBUG"))


class TieredGroup(click.Group):
    """A group whose ``--help`` lists commands in named tiers.

    ``sections`` is a list of ``(title, [command names])`` pairs, set by
    the CLI module after all commands are registered. Rung-0 commands
    (the free local loop: init → example → dev → replay) come first so a
    fresh ``papayya --help`` reads as a quickstart, not an inventory;
    hosted/ops groups sit below. Commands not named in any section fall
    into a trailing "Other commands" bucket so nothing silently vanishes
    from help when a new command forgets to register itself in a tier.
    """

    sections: list[tuple[str, list[str]]] = []

    def format_commands(self, ctx: click.Context, formatter: click.HelpFormatter) -> None:
        visible: list[tuple[str, click.Command]] = []
        for name in self.list_commands(ctx):
            cmd = self.get_command(ctx, name)
            if cmd is None or cmd.hidden:
                continue
            visible.append((name, cmd))
        if not visible:
            return

        by_name = dict(visible)
        limit = formatter.width - 6 - max(len(n) for n, _ in visible)

        seen: set[str] = set()
        for title, names in self.sections:
            rows = [
                (n, by_name[n].get_short_help_str(limit))
                for n in names
                if n in by_name
            ]
            if rows:
                seen.update(n for n, _ in rows)
                with formatter.section(title):
                    formatter.write_dl(rows)

        rest = [
            (n, cmd.get_short_help_str(limit))
            for n, cmd in visible
            if n not in seen
        ]
        if rest:
            title = "Other commands" if self.sections else "Commands"
            with formatter.section(title):
                formatter.write_dl(rest)


class SafeGroup(TieredGroup):
    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        # `--env` is a root-group option but users naturally type it after the
        # subcommand (`papayya deploy --env staging`). Click rejects that
        # because subcommand parsing happens before group parsing. Hoist any
        # `--env <value>` or `--env=value` to the front so both positions work.
        # No subcommand declares its own `--env`, so this is unambiguous.
        hoisted: list[str] = []
        i = 0
        while i < len(args):
            tok = args[i]
            if tok == "--env" and i + 1 < len(args):
                hoisted.extend(args[i:i + 2])
                del args[i:i + 2]
                continue
            if tok.startswith("--env="):
                hoisted.append(tok)
                del args[i]
                continue
            i += 1
        if hoisted:
            args = hoisted + args
        return super().parse_args(ctx, args)

    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except (click.ClickException, click.Abort, click.exceptions.Exit, KeyboardInterrupt, SystemExit):
            raise
        except PapayyaAPIError as e:
            click.echo(f"Error: {e}", err=True)
            if _debug_enabled():
                traceback.print_exc()
            sys.exit(1)
        except Exception as e:  # noqa: BLE001
            click.echo(f"Error: {type(e).__name__}: {e}", err=True)
            click.echo("  Run with PAPAYYA_DEBUG=1 for a full traceback.", err=True)
            if _debug_enabled():
                traceback.print_exc()
            sys.exit(1)
