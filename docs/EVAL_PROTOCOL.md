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
  excludes the event itself, so only history events can be returned.
- **Ranked list:** the top 10 *distinct* events (`search_events`); an event that appears
  more than once counts once, at its first position. The IDCG pool is every history event
  before `before_date`, label sets counted with multiplicity.
- **Percentiles:** nearest rank (the smallest value with at least p% of values at or
  below it).

## Known-item questions

A second retrieval check uses questions written for single history passages.

- **Sampling** (`scripts/draft_questions.py`): 220 passages from events before 2024-01-01,
  split evenly across section types (factual, analysis, probable cause), proportionally to
  event year within a section, seed 42.
- **Drafting:** the LLM (`causetrace.llm`, `claude-haiku-5-5`) writes one question per
  passage that an investigator might ask and that the passage answers, paraphrased.
- **Review** (`scripts/review_questions.py`): every draft is kept, edited or dropped by a
  person. Only kept and edited questions are scored.
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
