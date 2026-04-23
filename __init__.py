"""
RegIntelEnv root package.
Exports the public API for use as a client library.
"""

import os
import sys

# Ensure this directory is on the path
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from models import (  # noqa: E402
    ActionType,
    ComplianceStatus,
    DifficultyLevel,
    RegAction,
    RegObservation,
    RegReward,
    RegState,
    StepResult,
)
from server.reg_intel_environment import RegIntelEnvironment  # noqa: E402

__all__ = [
    "RegIntelEnvironment",
    "RegAction",
    "RegObservation",
    "RegReward",
    "RegState",
    "StepResult",
    "ActionType",
    "ComplianceStatus",
    "DifficultyLevel",
]
