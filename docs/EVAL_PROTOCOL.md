# Evaluation protocol

These definitions are fixed before any results exist. **They may not change after results
exist without a dated note in the change log at the end of this file**, stating what
changed, why, and which results were produced under the old definition.

## Split

- **Held-out:** events with `ev_date >= 2024-01-01`.
- **History:** events with `ev_date < 2024-01-01`.
- The split is by date only and deterministic (`causetrace.split`): lists are ordered by
  `(ev_date, ev_id)`. Nothing is tuned, selected or calibrated on held-out events.
- The retrieval corpus and every statistic a system uses (for example label frequencies)
  come from history only.

## Labels

- An event's gold label set `G` is the set of distinct codes
  `category_no-subcategory_no-section_no` (for example `02-04-15`) over its `Findings`
  rows with `cm_inPc = 'T'`, across all aircraft of the event (`causetrace.gold`).
- The label taxonomy (`labels` table) maps each code to a short name taken from the first
  three segments of `finding_description`.
- Events whose `G` is empty are not scored; their number is reported.

`J(A, B) = |A ∩ B| / |A ∪ B|` is the Jaccard similarity of two label sets (0 when both are
empty).

## Retrieval metrics

For a held-out query event with gold set `G`, a retriever returns a ranked list of history
events; `L_i` is the gold label set of the event at rank `i`. Only the top 10 are scored.

- **Label Recall@10** = `|G ∩ (L_1 ∪ … ∪ L_10)| / |G|`.
- **nDCG@10 with Jaccard gain**: gain `g_i = J(G, L_i)`;
  `DCG@10 = Σ_{i=1..10} g_i / log2(i + 1)`; `IDCG@10` is the DCG of the 10 history events
  with the highest `J(G, ·)` in descending order; nDCG@10 = DCG@10 / IDCG@10 (0 when
  IDCG@10 = 0).
- **Relevance** for the next two metrics: a retrieved event is relevant when
  `J(G, L_i) >= 0.5`.
- **MRR**: mean over queries of `1 / rank` of the first relevant event in the top 10
  (0 when there is none).
- **Hit@10**: fraction of queries with at least one relevant event in the top 10.
- **Latency**: wall-clock time per query, from packet in to ranked list out; report p50
  and p95.

All retrieval metrics are averaged over scored held-out events.

How the held-out retrieval evaluation runs (`scripts/eval_retrieval.py`):

- **Scored events:** held-out events with a non-empty `G` and a non-empty `narr_accp`.
  `--limit N --seed S` scores a seeded random sample of them instead of all.
- **Query:** the incident packet text (`causetrace.packets`): the gated factual narratives,
  or the packet's SAFE metadata when every narrative is withheld. Gold fields never enter
  the query.
- **Corpus:** every search uses `before_date = min(ev_date, 2024-01-01)`
  (`causetrace.split.corpus_cutoff`, the one helper for every evaluation and the agent) and
  excludes the event itself, so only history events can be returned. Search also enforces
  this itself: `history_only=True` (the default) caps any `before_date` at 2024-01-01.
- **Ranked list:** the top 10 *distinct* events (`search_events`); an event that appears
  more than once counts once, at its first position. The IDCG pool is every history event
  before `before_date`, label sets counted with multiplicity.
- **Percentiles:** nearest rank (the smallest value with at least p% of values at or
  below it).

## Known-item questions

A second retrieval check uses questions written for single history passages.

- **Sampling** (`sample_passages`): 220 eligible passages from events before 2024-01-01,
  split evenly across section types (factual, analysis, probable cause), proportionally to
  event year within a section, seed 42. Eligible means at least 20 words, text not shared
  with any other event (case and whitespace ignored), and none of the NTSB boilerplate
  markers in `BOILERPLATE` (foreign-investigation and FBI-led notices, pointers to separate
  reports), so every question has exactly one right event.
- **Drafting:** one "Which accident…?" question per passage naming two or three details
  that together single out the event, paraphrased, with no registration numbers, names,
  dates, airport identifiers or exact place names. The committed drafts
  (`evals/questions_draft.jsonl`) were written by Claude in a Claude Code session, recorded
  as `usage.model = "claude-code-session"`. `scripts/draft_questions.py` can draft with the
  LLM (`causetrace.llm`, `claude-haiku-5-5`) instead.
- **Review** (`evals/questions_reviewed.jsonl`): every draft is kept, edited or dropped.
  Only kept and edited questions are scored. `scripts/review_questions.py` records decisions
  made one at a time in the terminal. For the committed 220, Claude proposed the decisions
  (drop questions that cannot identify one event; edit questions stating facts not in the
  passage, after a check of every question against its passage) and the project owner
  approved the list as a whole; each record carries `reviewer` and, for drops and edits,
  `reason`.
- **Search:** each question is searched with `before_date = 2024-01-01` (history corpus)
  and nothing excluded, since the source event is the answer.
- **Recall@10:** share of questions whose source event is among the top 10 distinct events.
- **MRR:** mean of `1 / rank` of the source event in the top 10 (0 when absent).
- **p95:** nearest-rank 95th percentile of per-query wall-clock latency.

## Causal systems

Every system receives only the target event's packet (`causetrace.packets`) and, where
stated, retrieved history events.

1. **Frequency prior** — predicts the most frequent history labels (no event input).
2. **Nearest-neighbour vote** — labels voted by the top retrieved history events.
3. **Closed-book LLM** — LLM with the packet only, no retrieved context.
4. **Random-context LLM** — same prompt as the RAG agent, with history events drawn at
   random (fixed seed) instead of retrieved; controls for the effect of context itself.
5. **RAG agent** — LLM with the packet and retrieved history events.
6. **RAG agent + verifier** — the RAG agent's output checked by a verifier step that may
   drop labels or abstain.

System hyperparameters (k, thresholds, prompts) are chosen on history events only.

How the causal comparison runs (`scripts/eval_causal.py`):

- **Cases:** the scored held-out events (gold labels and a non-empty `narr_accp`; 514 on
  the full snapshot). `--cases N` samples N with a fixed seed; the list is frozen in
  `evals/causal_cases.json`. The frequency prior and the kNN vote always run on every
  scored event (they are free) and are also reported on the frozen cases.
- **Tuning set:** 2023 history events with gold labels and a non-empty `narr_accp`. Each is
  queried with its own history-only cutoff (`corpus_cutoff`, so its own date) and excludes
  itself.
- **Frequency prior:** the k labels found in the most history events; k in 1–5 maximizes
  micro F1 on the tuning set using counts from events before 2023 (ties: smaller k); the
  final list counts all history events.
- **kNN vote:** the top 10 distinct events from `search_events` (`rrf`, packet text,
  `corpus_cutoff`, target excluded). A label's score is the sum of 1/rank over the
  neighbours that have it, divided by the sum of 1/rank; labels scoring at least the
  cut-off are predicted (at most 5, highest first). The cut-off is chosen from 0.05–0.95
  (step 0.05) by micro F1 on the tuning set (ties: lower cut-off).
- **Closed-book LLM:** `prompts/closed_book.md` with the fixed system prompt and taxonomy;
  up to 5 labels; effort medium.
- **Random-context LLM:** the agent's analyze prompt with 10 history passages drawn by
  `md5(passage_id || seed || ev_id)` order from passages before the case's cutoff, target
  excluded; one call, no verify.
- **RAG agent:** `causetrace.agent.Agent(verify=False)`: the merged agent with the verify
  node skipped, so analyze's predictions are the answer. **RAG agent + verifier:** the full
  agent.
- LLM systems call `claude-haiku-5-5` through `AnthropicProvider` directly (no fallback);
  `--model` adds rows for another model on the same cases; `--repeat R` reports the mean
  and the range over R runs.

## Causal metrics

A system outputs, per held-out event, a predicted label set `P` (possibly empty), optional
citations (pairs of predicted label and cited history `ev_id`), or an abstention.

- **Precision, recall, F1** — micro-averaged over scored held-out events:
  `TP = Σ|P ∩ G|`, `FP = Σ|P \ G|`, `FN = Σ|G \ P|`. An abstention counts as `P = ∅`.
  Macro (per-event mean) values are reported alongside.
- **Citation validity** — fraction of citations whose `ev_id` is a history event that was
  in the context given to the system for that query.
- **Citation support** — fraction of citations whose cited event's gold label set
  contains the predicted label it is cited for.
- **Abstention rate** — fraction of scored held-out events on which the system abstained.

Citation metrics are reported only for systems that cite (not applicable to the frequency
prior and the closed-book LLM).

How these are computed (`causetrace.evaluation.causal_metrics`):

- A system that abstains or predicts nothing has `P = ∅` and counts as an abstention.
  Macro values give an empty prediction precision 0. **F1 answered** is micro F1 over the
  cases that were not abstentions. "Recovered X of Y" is `TP` of `Σ|G|`.
- Citations are passage ids; each maps to its history event through `passages`. A citation
  is valid when the passage was in the context given to the system for that case and its
  event predates the case's cutoff. Citation support is over all citations (invalid ones
  count as unsupported) and reads the cited event's gold labels through `causetrace.gold`,
  for scoring only.
- **Verifier support rate** (RAG agent + verifier): share of the (finding, citation) pairs
  the verifier judged in its final round that it judged SUPPORTED.
- Gold labels outside the taxonomy (codes never seen as causes before 2024) are misses for
  every system; their number is reported.
- **Human citation precision:** 50 citations sampled from RAG agent + verifier answers
  (`evals/citation_audit.csv`, seed 42, drawn once the run has answered every case) are
  graded 2 (supports), 1 (partly) or 0 (does not) with `scripts/grade_citations.py`; the
  report gives the share graded 2 and the share graded 1 or 2, next to the verifier support
  rate.
- Per-case cost: mean tokens, LLM calls, wall-clock seconds and dollars.

## Change log

- 2026-10-11 — Initial version.
- 2026-10-11 — Narrative gate measured on the full snapshot (`artifacts/ingest_report_full.json`;
  no definitions changed, no results existed yet). Among non-empty factual narratives, the gate
  withholds 5,934 of 23,043 history narratives and 0 of 1,357 held-out narratives, and removes
  88,913 of 780,984 history sentences and 2,316 of 46,760 held-out sentences. Many held-out
  narratives have no `narr_accf` or `narr_cause` to check against, but all of them belong to
  events with no `cm_inPc = 'T'` findings, which are never scored: all 514 scored held-out
  events (with findings and a non-empty `narr_accp`) have gold narratives, so the gate checked
  every scored packet. The gate applies only to target-event packets; history passages used for
  retrieval keep their full narratives.
- 2026-10-11 — Added how the held-out retrieval evaluation runs (scored events, query,
  corpus cutoff, distinct events, IDCG pool, percentile method) and the known-item
  question protocol. These spell out or add definitions before any retrieval results exist;
  no existing definition changed.
- 2026-10-11 — Known-item sampling skips passages that cannot identify one event (under 20
  words, text shared across events, NTSB boilerplate). A 24-passage pilot found about a third
  of the earlier sample unusable for known-item search. The 220 drafts were written in a
  Claude Code session instead of by `claude-haiku-5-5`: in the pilot, Haiku's questions were
  either too general to identify one event or, with a stricter prompt, copied rare numbers
  and names from the passage. No results existed yet; no other definition changed.
- 2026-10-11 — Review of the 220 drafts: 171 kept, 19 edited, 30 dropped (5 factual and 4
  analysis passages with nothing distinctive, 21 generic one-line probable causes). Decisions
  were proposed by Claude and approved as a list by the project owner rather than made one
  at a time; the Review entry above now says so. No results existed yet.
- 2026-10-11 — Spelled out how the causal comparison runs and how its metrics are computed
  (case freezing, tuning set, frequency-prior k and kNN cut-off tuning, random-context
  draw, empty predictions as abstentions, answered-only F1, citation validity and support
  per passage, verifier support rate, human citation audit). No causal results existed yet;
  no existing definition changed.
