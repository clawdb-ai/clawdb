from __future__ import annotations

__all__ = ["app"]


def __getattr__(name: str):
    if name == "app":
        from .api import app

        return app
    raise AttributeError(f"module 'clawdb' has no attribute {name!r}")
