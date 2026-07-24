"""Central defaults for the Papayya SDK.

The default base URL points at the production control plane. For local
development against a docker-compose stack, set ``PAPAYYA_BASE_URL`` in
your environment or pass ``--base-url`` to the CLI.
"""

from __future__ import annotations

DEFAULT_BASE_URL = "https://api.getpapayya.com"

# The hosted dashboard — where you sign up, create a project, and mint the API
# key the CLI consumes. Account creation lives here, not in the CLI.
DEFAULT_DASHBOARD_URL = "https://app.getpapayya.com"
