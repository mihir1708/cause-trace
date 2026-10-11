-- Agent runs served by the API (causetrace.api): the request, the full result with its node
-- trace, and the outcome. A run holds model output and retrieved history passage ids only;
-- the target event's gold labels are never stored here.

CREATE TABLE analysis_runs (
    run_id          uuid PRIMARY KEY,
    created_at      timestamptz NOT NULL DEFAULT now(),
    ev_id           text,                 -- NULL for a free-text packet
    status          text NOT NULL CHECK (status IN ('completed', 'failed')),
    model           text,
    abstained       boolean,
    grounding_score double precision,     -- share of findings kept by the citation guardrail
    request         jsonb NOT NULL,
    result          jsonb,                -- the agent result with its trace (completed runs)
    error           text                  -- why the run failed (failed runs)
);

CREATE INDEX analysis_runs_created_at_idx ON analysis_runs (created_at);
