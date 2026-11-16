"""Pure artifact vocabulary, configuration decoding and canonical identities.

No filesystem operations or imports from artifact validators: those boundaries
consume these definitions independently. Dynamic schema/JSON handling ends here.
"""

from __future__ import annotations

import json


def canonical_bytes(value: object) -> bytes:
    """Encode sorted strict JSON with a trailing newline; reject nonfinite data."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
