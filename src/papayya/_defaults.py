"""Central defaults for the Papayya SDK.

When the production API domain is registered, update ``DEFAULT_BASE_URL``
here and all SDK entry points will pick up the new value.
"""

from __future__ import annotations

# TODO(launch): change to the production URL once the domain is registered.
DEFAULT_BASE_URL = "http://localhost:8090"
