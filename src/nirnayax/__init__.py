"""NirnayaX — enterprise L1 IT incident triage system.

Phase 1 scope: a clean, typed **domain + data foundation**. This package exposes
the incident taxonomy, Pydantic domain models, a deterministic synthetic dataset
generator, a runbook catalog, and lightweight validation/EDA utilities.

No ML, RAG, agents, ticketing integrations, or UI live here yet; the package is
structured so those layers can be added without reshaping the foundation.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
