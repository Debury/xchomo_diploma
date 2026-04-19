"""Root conftest.

Adds ``src/`` to ``sys.path`` so tests can ``import climate_embeddings`` etc.
without the ``src.`` prefix. Pytest discovers this before collecting tests.

Also sets ``AUTH_ALLOW_ANONYMOUS=1`` for the test process. The production auth
gate fails closed when no password is configured; the in-process test suite
(``httpx.AsyncClient`` + ``ASGITransport``) does not log in and relies on the
anonymous path. The corresponding production refusal only fires when
``APP_ENV=production``, so this is safe.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

os.environ.setdefault("AUTH_ALLOW_ANONYMOUS", "1")
