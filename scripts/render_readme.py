"""Regenerate the results tables in README.md from artifacts/ (between marker comments).

Usage: uv run python scripts/render_readme.py [--check]

--check exits 1 when README.md is out of date instead of writing it.
"""

import argparse
import sys

from causetrace.evaluation.readme import ARTIFACTS, README, blocks, render


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    current = README.read_text(encoding="utf-8")
    updated = render(current, blocks(ARTIFACTS))
    if args.check:
        if updated != current:
            print("README.md results are out of date: run scripts/render_readme.py")
            return 1
        return 0
    README.write_text(updated, encoding="utf-8")
    print("README.md results updated" if updated != current else "README.md already up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main())
