"""Shared fixtures.

Two rules hold across the suite:

1. **Nothing writes to ``data/``.** Every stateful component here takes an
   explicit path (``ModelZooService(zoo_path=…)``,
   ``RLPolicyController(state_path=…)``), so tests construct isolated instances
   rather than patching module-level constants — those are read at import time
   and monkeypatching the env after import would not take effect.

2. **Tests import the dependency-light modules directly.** ``decision_engine``
   needs fastapi and the full backend stack, so it is deliberately not imported;
   logic that matters is tested against ``model_zoo`` / ``rl_controller`` /
   ``model_onboarding``, which are import-cycle-free by design.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

# The modules under test live at the repo root, not in a package.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="session")
def fixture_zoo_path() -> Path:
    """Path to the minimal, round-numbered zoo used for golden carbon values."""
    return FIXTURE_DIR / "model_zoo_min.json"


@pytest.fixture
def zoo(fixture_zoo_path: Path):
    """A ModelZooService over the fixture zoo — isolated from the shipped one."""
    from model_zoo import ModelZooService

    return ModelZooService(zoo_path=fixture_zoo_path)


@pytest.fixture(scope="session")
def shipped_zoo_path() -> Path:
    """The real ``config/model_zoo.json``. Used only for bounds assertions, never
    for golden values — retuning a real model must not break the carbon tests."""
    return REPO_ROOT / "config" / "model_zoo.json"

