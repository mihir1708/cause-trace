"""Build the small test fixture in data/fixtures/ from the mdb-export CSVs in data/interim/.

Selection: events with at least one Findings row where cm_inPc == 'T' and a
non-empty narratives.narr_accp. 200 events before 2024-01-01 (stratified
across ev_year) and 100 on/after. Seed 42. Rows are copied byte-for-byte from the export.

Usage: python scripts/make_fixture.py [interim_dir] [out_dir]
"""

import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42
N_HISTORICAL = 200
N_RECENT = 100
CUTOFF_YEAR = 2024  # ev_date before 2024-01-01 <=> ev_year < 2024
TABLES = ["events", "aircraft", "narratives", "Findings"]

csv.field_size_limit(sys.maxsize)


def read(path):
    """Return (header, rows, raw, header_raw); raw[i] is the exact source text of rows[i]."""
    buf = []

    def lines(f):
        for line in f:
            buf.append(line)
            yield line

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(lines(f))
        header = next(reader)
        header_raw = "".join(buf)
        buf.clear()
        rows, raw = [], []
        for row in reader:
            rows.append(row)
            raw.append("".join(buf))
            buf.clear()
        return header, rows, raw, header_raw


def stratified_sample(ids_by_year, n, rng):
    """Allocate n across years proportionally (largest remainder), then sample within each year."""
    total = sum(len(v) for v in ids_by_year.values())
    years = sorted(ids_by_year)
    quotas = {y: n * len(ids_by_year[y]) / total for y in years}
    alloc = {y: int(quotas[y]) for y in years}
    for y in sorted(years, key=lambda y: (alloc[y] - quotas[y], y))[: n - sum(alloc.values())]:
        alloc[y] += 1
    picked = []
    for y in years:
        picked += rng.sample(sorted(ids_by_year[y]), alloc[y])
    return picked


def main():
    interim = Path(sys.argv[1] if len(sys.argv) > 1 else "data/interim")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "data/fixtures")
    data = {t: read(interim / f"{t}.csv") for t in TABLES}

    ev_h, ev_rows, _, _ = data["events"]
    fi_h, fi_rows, _, _ = data["Findings"]
    na_h, na_rows, _, _ = data["narratives"]
    ev_id, year = ev_h.index("ev_id"), ev_h.index("ev_year")
    in_pc = {r[0] for r in fi_rows if r[fi_h.index("cm_inPc")] == "T"}
    with_narr = {r[0] for r in na_rows if r[na_h.index("narr_accp")].strip()}

    all_ids = {r[ev_id] for r in ev_rows}
    usable = all_ids & in_pc & with_narr
    print(f"events: {len(all_ids)}")
    print(f"events with cm_inPc='T' findings: {len(all_ids & in_pc)}")
    print(f"usable (also non-empty narr_accp): {len(usable)}")

    historical, recent = defaultdict(set), set()
    for r in ev_rows:
        if r[ev_id] in usable:
            y = int(r[year])
            (historical[y].add(r[ev_id]) if y < CUTOFF_YEAR else recent.add(r[ev_id]))
    print(f"usable historical: {sum(len(v) for v in historical.values())}, recent: {len(recent)}")

    rng = random.Random(SEED)
    chosen = set(stratified_sample(historical, N_HISTORICAL, rng))
    chosen |= set(rng.sample(sorted(recent), N_RECENT))

    out.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        _, rows, raw, header_raw = data[t]
        kept = [line for r, line in zip(rows, raw) if r[0] in chosen]
        with open(out / f"{t}.csv", "w", newline="", encoding="utf-8") as f:
            f.write(header_raw)
            f.writelines(kept)
        print(f"fixture {t}.csv: {len(kept)} rows")


if __name__ == "__main__":
    main()
