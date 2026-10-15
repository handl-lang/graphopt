"""``python -m benches`` entry point."""

from __future__ import annotations

import sys

from benches.cli import main

if __name__ == "__main__":
    sys.exit(main())
