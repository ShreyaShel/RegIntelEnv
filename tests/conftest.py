"""
pytest conftest.py — adds the project root to sys.path so all tests
can import models, tasks, grader, and server/* without installing the package.
"""
import os
import sys

# Project root = the RegIntelEnv/ directory (one level up from tests/)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
