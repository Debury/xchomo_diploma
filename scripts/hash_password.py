"""Print a PBKDF2-SHA256 hash suitable for ``AUTH_PASSWORD_HASH``.

Usage:

    python scripts/hash_password.py
    # prompts for the password (hidden)

    python scripts/hash_password.py --password 'literal secret'
    # non-interactive; avoid in shell history

Paste the output into ``.env``:

    AUTH_PASSWORD_HASH=pbkdf2_sha256$600000$...$...

If both ``AUTH_PASSWORD_HASH`` and ``AUTH_PASSWORD`` are set, the hash wins.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from web_api.routes.auth import _hash_password  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--password", help="Password literal. Omit to be prompted.")
    ap.add_argument("--iterations", type=int, default=600_000, help="PBKDF2 iterations (default 600k).")
    args = ap.parse_args()

    password = args.password or getpass.getpass("Password: ")
    if not password:
        print("empty password refused", file=sys.stderr)
        return 2

    print(_hash_password(password, iterations=args.iterations))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
