-- CauseTrace initial schema. Column names follow the NTSB mdb-export headers (lowercased).
-- Leakage classes per column are defined in docs/DATA_LEAKAGE.md.

CREATE TABLE events (
    ev_id             text PRIMARY KEY,
    ntsb_no           text,
    ev_type           text,
    ev_date           date NOT NULL,
    ev_year           integer,
    ev_time           integer,          -- local time as HHMM
    ev_tmzn           text,
    ev_city           text,
    ev_state          text,
    ev_country        text,
    dec_latitude      double precision,
    dec_longitude     double precision,
    ev_nr_apt_id      text,
    apt_name          text,
    wx_cond_basic     text,
    light_cond        text,
    sky_cond_ceil     text,
    sky_ceil_ht       integer,
    vis_sm            double precision,
    wx_temp           integer,
    wx_dew_pt         integer,
    wind_dir_deg      integer,
    wind_vel_kts      integer,
    gust_kts          integer,
    altimeter         double precision,
    ev_highest_injury text,
    inj_tot_f         integer,
    inj_tot_s         integer,
    inj_tot_m         integer,
    inj_tot_n         integer,
    lchg_date         timestamp
);

CREATE INDEX events_ev_date_idx ON events (ev_date, ev_id);

CREATE TABLE aircraft (
    ev_id          text NOT NULL REFERENCES events (ev_id) ON DELETE CASCADE,
    aircraft_key   integer NOT NULL,
    regis_no       text,
    ntsb_no        text,
    far_part       text,
    damage         text,
    acft_category  text,
    acft_make      text,
    acft_model     text,
    acft_series    text,
    phase_flt_spec text,
    homebuilt      text,
    num_eng        integer,
    lchg_date      timestamp,
    PRIMARY KEY (ev_id, aircraft_key)
);

CREATE TABLE narratives (
    ev_id                  text NOT NULL,
    aircraft_key           integer NOT NULL,
    narr_accp              text,     -- CONDITIONAL: raw factual narrative, never model input
    narr_accf              text,     -- GOLD_ONLY
    narr_cause             text,     -- GOLD_ONLY
    narr_inc               text,     -- unclassified, excluded from model input
    narr_accp_safe         text,     -- narr_accp after the leakage gate; NULL = withheld
    accp_sentences_total   integer NOT NULL DEFAULT 0,
    accp_sentences_removed integer NOT NULL DEFAULT 0,
    lchg_date              timestamp,
    PRIMARY KEY (ev_id, aircraft_key),
    FOREIGN KEY (ev_id, aircraft_key) REFERENCES aircraft (ev_id, aircraft_key) ON DELETE CASCADE
);

-- Label taxonomy: code = category_no-subcategory_no-section_no.
CREATE TABLE labels (
    code       text PRIMARY KEY CHECK (code ~ '^[0-9]{2}-[0-9]{2}-[0-9]{2}$'),
    short_name text NOT NULL
);

-- GOLD_ONLY.
CREATE TABLE findings (
    ev_id               text NOT NULL,
    aircraft_key        integer NOT NULL,
    finding_no          integer NOT NULL,
    finding_code        text,
    finding_description text,
    category_no         text NOT NULL,
    subcategory_no      text NOT NULL,
    section_no          text NOT NULL,
    subsection_no       text,
    modifier_no         text,
    cause_factor        text,
    cm_inpc             boolean,
    label_code          text GENERATED ALWAYS AS
                            (category_no || '-' || subcategory_no || '-' || section_no) STORED
                            REFERENCES labels (code),
    lchg_date           timestamp,
    PRIMARY KEY (ev_id, aircraft_key, finding_no),
    FOREIGN KEY (ev_id, aircraft_key) REFERENCES aircraft (ev_id, aircraft_key) ON DELETE CASCADE
);

CREATE INDEX findings_ev_id_idx ON findings (ev_id);
