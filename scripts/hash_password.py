"""Print an argon2id hash suitable for ``AUTH_PASSWORD_HASH``.

Usage:

    python scripts/hash_password.py
    # prompts for the password (hidden)

    python scripts/hash_password.py --password 'literal secret'
    # non-interactive; avoid in shell history

Paste the output into ``.env``:

    AUTH_PASSWORD_HASH=$argon2id$v=19$m=65536,t=3,p=4$...

If both ``AUTH_PASSWORD_HASH`` and ``AUTH_PASSWORD`` are set, the hash wins.

Backed by ``pwdlib`` — the same library the auth code uses to verify, so the
hash format stays in sync across upgrades.
"""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from web_api.routes.auth import hash_password  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--password", help="Password literal. Omit to be prompted.")
    args = ap.parse_args()

    password = args.password or getpass.getpass("Password: ")
    if not password:
        print("empty password refused", file=sys.stderr)
        return 2

    print(hash_password(password))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
