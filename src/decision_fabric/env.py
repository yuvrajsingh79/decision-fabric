"""Load a local .env so `--live` works without exporting anything globally.

Deliberately tiny and deliberately non-destructive: a variable already present
in the real environment always wins, so a stray .env cannot silently redirect
billing to the wrong account.
"""
from __future__ import annotations

import os
from pathlib import Path


PLACEHOLDER_MARKERS = (
    "your-real-key", "your-api-key", "your_api_key", "replace", "changeme",
    "xxx", "<", "here", "todo",
)


def _looks_like_placeholder(value: str) -> bool:
    low = value.lower()
    if any(m in low for m in PLACEHOLDER_MARKERS):
        return True
    # Real console keys are ~100 chars; anything much shorter is a stub.
    return low.startswith("sk-ant-") and len(value) < 40


def load_dotenv(path: str | Path = ".env") -> list[str]:
    """Set any KEY=VALUE from `path` that is not already in os.environ.

    Returns the names of the variables it set (never their values).
    """
    p = Path(path)
    if not p.is_file():
        return []
    loaded: list[str] = []
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key or not value or key in os.environ:
            continue
        if _looks_like_placeholder(value):
            # A placeholder API key is worse than no key: it OUTRANKS an
            # `ant auth login` profile, so the SDK silently ignores real
            # credentials and every call fails auth for a reason that looks
            # nothing like "your .env has a dummy value in it".
            print(f"warning: ignoring placeholder value for {key} in {p}")
            continue
        os.environ[key] = value
        loaded.append(key)
    return loaded
