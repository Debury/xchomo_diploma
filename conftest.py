"""Root conftest.

Adds ``src/`` to ``sys.path`` so tests can ``import climate_embeddings`` etc.
without the ``src.`` prefix. Pytest discovers this before collecting tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
