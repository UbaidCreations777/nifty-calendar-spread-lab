"""CLI: python -m src.data.download [start] [end]"""
from __future__ import annotations

import sys
from datetime import date

from .. import config as C
from . import nse_fo


def main() -> int:
    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else C.BACKTEST_START
    end = date.fromisoformat(sys.argv[2]) if len(sys.argv) > 2 else C.BACKTEST_END

    print(f"NSE F&O bhavcopy: {start} -> {end}", flush=True)
    stats = nse_fo.download_range(start, end)

    print(f"\ndownloaded {stats['downloaded']} | cached {stats['cached']} | "
          f"holidays {stats['holiday']} | failed {stats['failed']}")
    if stats["failures"]:
        print("\nfailures:")
        for d, note in stats["failures"][:20]:
            print(f"  {d}: {note}")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
