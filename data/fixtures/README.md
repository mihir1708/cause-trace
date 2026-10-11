# Fixture sample

A small, fixed subset of the NTSB aviation accident database for tests and development.

- **Source:** NTSB `avall.zip` (Access database `avall.mdb`), downloaded from
  https://data.ntsb.gov/avdata/FileDirectory/DownloadFile?fileID=C%3A%5Cavdata%5Cavall.zip.
  Hashes and download time are in `data_manifest.json` at the repo root.
- **Selection rule:** events with at least one `Findings` row where `cm_inPc = 'T'` and a
  non-empty `narratives.narr_accp`. 200 events with `ev_date` before 2024-01-01, stratified
  across `ev_year` proportionally, plus 100 events on/after 2024-01-01, sampled uniformly.
- **Seed:** 42 (`random.Random(42)`). Regenerate with `python scripts/make_fixture.py`
  after exporting the tables to `data/interim/` with `mdb-export`.
- **Files:** `events.csv`, `aircraft.csv`, `narratives.csv`, `Findings.csv`, containing only rows
  for the 300 selected `ev_id`s. Rows are byte-for-byte copies of the `mdb-export` output
  (original column names, quoting and date formats such as `01/10/08 00:00:00`).
- **License:** NTSB accident data is public information published by the U.S. government.
