"""Run the /v1/search load test headless at several user counts and write the summary.

Usage: uv run --group load python loadtests/run.py [--host URL] [--users 1 10 25 50]
       [--duration 60s]

Writes artifacts/load_results.{json,md} (RPS, p50, p95, p99, error rate per user count).
Needs the API running (`causetrace serve` or `docker compose --profile api up -d`).
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

from causetrace.evaluation.load_report import (
    DEFAULT_JSON,
    DEFAULT_MD,
    read_locust_stats,
    write_report,
)

LOCUSTFILE = Path(__file__).with_name("locustfile.py")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else None)
    parser.add_argument("--host", default="http://localhost:8000")
    parser.add_argument("--users", type=int, nargs="+", default=[1, 10, 25, 50])
    parser.add_argument("--duration", default="60s")
    parser.add_argument("--out-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_MD)
    args = parser.parse_args(argv)
    levels = []
    with tempfile.TemporaryDirectory() as tmp:
        for users in args.users:
            prefix = Path(tmp) / f"u{users}"
            print(f"--- {users} users for {args.duration}", flush=True)
            subprocess.run(
                [
                    sys.executable, "-m", "locust", "-f", str(LOCUSTFILE), "--headless",
                    "--host", args.host, "-u", str(users), "-r", str(users),
                    "-t", args.duration, "--csv", str(prefix), "--only-summary",
                ],
                check=False,  # locust exits 1 when any request failed; the summary says so
            )  # fmt: skip
            levels.append(read_locust_stats(Path(f"{prefix}_stats.csv"), users))
    write_report(levels, host=args.host, duration=args.duration,
                 out_json=args.out_json, out_md=args.out_md)  # fmt: skip
    print(args.out_md.read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
