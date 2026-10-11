# Data leakage policy

The task is to predict an event's NTSB causal findings from its facts. Anything the NTSB
wrote *as a result of* its analysis of that event is the answer, not an input. Every field
of the source tables belongs to exactly one class below. **Fields not listed are EXCLUDED**
from model input until they are classified here with a dated change-log note.

## Classes

### SAFE — may enter model input

Event and aircraft metadata known independently of the causal analysis:

| Concept | Source columns |
|---|---|
| Date | `events.ev_date`, `events.ev_time`, `events.ev_tmzn` |
| Location | `events.ev_city`, `events.ev_state`, `events.ev_country`, `events.dec_latitude`, `events.dec_longitude`, `events.ev_nr_apt_id`, `events.apt_name` |
| Aircraft category | `aircraft.acft_category` |
| Make, model | `aircraft.acft_make`, `aircraft.acft_model`, `aircraft.acft_series` |
| Phase of flight | `aircraft.phase_flt_spec` |
| Weather | `events.wx_cond_basic`, `events.light_cond`, `events.sky_cond_ceil`, `events.sky_ceil_ht`, `events.vis_sm`, `events.wx_temp`, `events.wx_dew_pt`, `events.wind_dir_deg`, `events.wind_vel_kts`, `events.gust_kts`, `events.altimeter` |
| Injuries | `events.ev_highest_injury`, `events.inj_tot_f`, `events.inj_tot_s`, `events.inj_tot_m`, `events.inj_tot_n` |
| Damage | `aircraft.damage` |

### CONDITIONAL — may enter model input only after the leakage gate

`narratives.narr_accp`, the factual narrative.

The factual narrative describes what happened, but in the NTSB data it often shares text
with the GOLD_ONLY narratives: for short reports `narr_accp` can be identical to
`narr_accf`, and analysis narratives routinely copy factual sentences. Raw `narr_accp`
therefore never enters model input. Only the gated version, `narratives.narr_accp_safe`,
does. The gate (`causetrace.leakage.sanitize_factual_narrative`, applied at ingest) is:

1. Split `narr_accp` into paragraphs (line breaks) and sentences.
2. Gold text for an event = every non-empty `narr_accf` and `narr_cause` of **all aircraft
   of that event**. Normalise text to lowercase alphanumeric word tokens.
3. Drop every sentence that takes part in a run of **12 consecutive words** that also
   occurs in the gold text, or that contains a whole gold sentence of 4 to 11 words.
   Gold texts shorter than 12 words are matched as whole phrases.
4. Repeat on the remaining sentences until nothing matches (joining sentences can create
   new runs).
5. If no sentence survives, the narrative is withheld (`narr_accp_safe` is NULL).

The ingest report (`artifacts/ingest_report.json`, `narrative_sanitization`) records how
many narratives were withheld and how many sentences were removed. Changing the rule
(including the 12-word and 4-word thresholds) is a definition change and needs a dated note below.

### GOLD_ONLY — never in model input for the target event

- `narratives.narr_accf` — analysis narrative.
- `narratives.narr_cause` — probable cause statement.
- The `Findings` table (`findings` in PostgreSQL) and everything derived from it: label
  codes, finding codes, finding descriptions, `cm_inPc`, `Cause_Factor`.

GOLD_ONLY data is read only through `causetrace.gold` (labels for scoring), and by ingest.

## Enforcement

- `causetrace.packets.get_incident_packet(ev_id)` is the only builder of model input for a
  target event. It selects an explicit allowlist of SAFE columns plus `narr_accp_safe`.
- `packets.py` never imports `causetrace.gold`, and never names a GOLD_ONLY column, the
  findings or labels tables, or raw `narr_accp`. A test checks its source for this.
- A test builds the packet for every fixture event and asserts it contains none of that
  event's `narr_cause` or `narr_accf` text (whole text, any sentence of 40+ characters, any
  12-word run) and none of its label or finding codes.
- `IncidentPacket` models forbid extra fields, so a new field needs a code change that is
  reviewed against this document.

## Retrieval context (earlier investigations)

Retrieval-augmented systems may show gold data (findings, probable cause, analysis) of
*earlier* investigations as context, because those are the precedent being retrieved.
Constraints:

- The retrieval corpus contains only historical-split events (`ev_date < 2024-01-01`, see
  `docs/EVAL_PROTOCOL.md`). Held-out events are never retrievable, so a held-out event can
  never retrieve itself or a sibling held-out event.
- The query for a target event is built from its packet only.

## Passages and search

`passages` (built by `make index`) holds, per event, the factual narrative (raw
`narr_accp`) and the analysis (`narr_accf`), both chunked when longer than 400 words, and
the probable cause (`narr_cause`). It therefore contains GOLD_ONLY text of every event it
covers. Rules:

- `causetrace.retrieval.search` / `search_events` filter `event_date < before_date` and
  `event_id <> exclude_event_id` in the SQL of every candidate query (dense and lexical),
  so fusion and reranking only ever see allowed passages. Tests check every mode.
- `before_date` is a required argument. With `history_only=True` (the default) it is
  capped at 2024-01-01 (`causetrace.split.corpus_cutoff`), so search returns history
  passages only whatever date is passed. Only an explicit `history_only=False` lifts the
  cap; evaluations and the agent never pass it. For a target event pass
  `before_date = corpus_cutoff(ev_date)` and `exclude_event_id = <target ev_id>`.
- The query for a target event is built from its packet only (SAFE fields and
  `narr_accp_safe`), never from its passages.
- Lexical term weights (`lexeme_stats`) are document frequencies over historical-split
  passages only, so held-out text never shapes how queries are weighted.

## Known limitations

- **Report publication dates.** The exported tables have no final-report date, so a
  historical event's report may have been finalised after a held-out event occurred.
  Splitting on `ev_date` is the documented proxy.
- **Phase of flight.** `aircraft.phase_flt_spec` is empty in the current export (modern
  records keep phase of flight in `Events_Sequence`, which is not ingested yet). The field
  is carried as SAFE and is NULL in practice.
- **`narratives.narr_inc`** (incident narrative) is stored but not classified, so it is
  EXCLUDED from model input.

## Change log

- 2026-10-11 — Initial version.
- 2026-10-11 — Added "Passages and search" for hybrid retrieval. No class changed.
- 2026-10-11 — Search caps `before_date` at 2024-01-01 by default (`history_only`), and
  analysis passages are chunked like factual ones. No class changed.
