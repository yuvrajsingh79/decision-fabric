"""Test-suite spend tripwire.

`pytest` must never cost money. Credentials now resolve on this machine, so any
code path that reaches `Executor(dry_run=None)` would silently go live — the
failure mode is a test run that bills real tokens and nobody notices.

This fixture is autouse: it replaces the Anthropic client constructor for the
whole suite so any attempted real call fails loudly with a clear message
instead of quietly succeeding. A test that genuinely needs the network must
request the `allow_network` fixture explicitly.
"""
from __future__ import annotations

import pytest


class NetworkCallInTests(RuntimeError):
    pass


@pytest.fixture(autouse=True)
def _no_live_api_calls(request, monkeypatch):
    if "allow_network" in request.fixturenames:
        return
    try:
        import anthropic
    except ImportError:  # SDK not installed — nothing to guard
        return

    def _blocked(*a, **kw):
        raise NetworkCallInTests(
            "A test attempted to construct a live Anthropic client. Tests must "
            "run in dry-run mode — pass dry_run=True explicitly, or request the "
            "`allow_network` fixture if the test genuinely needs the API."
        )

    monkeypatch.setattr(anthropic, "Anthropic", _blocked)
    monkeypatch.setattr(anthropic, "AsyncAnthropic", _blocked, raising=False)


@pytest.fixture
def allow_network():
    """Opt out of the tripwire. Use sparingly and never in the default suite."""
    return True
