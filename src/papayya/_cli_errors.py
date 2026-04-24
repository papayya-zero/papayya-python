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


class SafeGroup(click.Group):
    def invoke(self, ctx: click.Context):
        try:
            return super().invoke(ctx)
        except (click.ClickException, click.Abort, KeyboardInterrupt, SystemExit):
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
