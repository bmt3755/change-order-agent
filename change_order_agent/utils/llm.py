from __future__ import annotations

from openai import OpenAI

# ---------------------------------------------------------------------------
# Single source of truth for how every agent talks to the OpenAI API.
#
# REQUEST_TIMEOUT: a hard cap on one API call. The SDK default is 600s (10 min)
# — far too long for an interactive triage tool. A stuck call must fail fast so
# the pipeline degrades to its fallback / human-review path instead of freezing
# the whole change order.
#
# MAX_RETRIES: the SDK retries transient errors (429 / 5xx / connection drops)
# with backoff. Worst-case wall time for a single call is therefore bounded at
# REQUEST_TIMEOUT * (1 + MAX_RETRIES) — about 90 seconds here, versus the
# unbounded hang the default allows.
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 30.0  # seconds
MAX_RETRIES = 2


def get_client() -> OpenAI:
    """Return an OpenAI client with a bounded timeout and retry budget.

    Reads OPENAI_API_KEY from the environment. Use this everywhere instead of
    constructing ``OpenAI()`` directly, so the timeout / retry policy lives in
    one place and every agent inherits the same fail-fast behavior.
    """
    return OpenAI(timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)
