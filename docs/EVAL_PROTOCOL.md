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
