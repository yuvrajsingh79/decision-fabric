"""Decision Fabric — a knowledge graph that decides which Claude model and
which request config a query actually needs, instead of sending everything to
the flagship at default settings.

Entry point: `decision_fabric.router.Router`.
"""

__version__ = "0.1.0"

from .router import Router, RoutingDecision  # noqa: E402,F401
