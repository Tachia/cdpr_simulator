-- cdpr — Supabase / PostgreSQL schema for persistent metadata.
--
-- Only metadata lives here. The scientific computation, the cable
-- constitutive laws, and the physics adapters all stay in the
-- cdpr.* Python core --- Supabase is the side-channel that lets
-- the deployed services remember experiments across cold-starts and
-- expose them to a small permissioned audience.
--
-- To apply this schema:
--   1. Create a new project at supabase.com
--   2. SQL editor → paste this file → Run
--   3. Set SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY in the backend env.

-- Experiments registered by the deployed framework.
create table if not exists experiments (
    id                uuid          primary key default gen_random_uuid(),
    name              text          not null,
    scenario_hash     text          not null,
    config_hash       text          not null,
    cable_mode        text          not null default 'none',
    backend           text          not null default 'cdpr',
    created_at        timestamptz   not null default now(),
    finished_at       timestamptz,
    seed              integer,
    cdpr_version      text,
    git_revision      text,
    notes             text,
    tags              jsonb         not null default '{}'::jsonb
);

create index if not exists experiments_scenario_hash_idx on experiments (scenario_hash);
create index if not exists experiments_cable_mode_idx    on experiments (cable_mode);
create index if not exists experiments_created_at_idx    on experiments (created_at desc);

-- Per-run summary metrics.
create table if not exists experiment_metrics (
    id                bigserial     primary key,
    experiment_id     uuid          not null references experiments(id) on delete cascade,
    metric            text          not null,
    value             double precision not null
);
create index if not exists experiment_metrics_exp_idx on experiment_metrics (experiment_id);

-- Uploaded experimental logs (raw CSV/XLSX/JSON, kept as bytes).
create table if not exists uploaded_logs (
    id                uuid          primary key default gen_random_uuid(),
    uploaded_at       timestamptz   not null default now(),
    filename          text          not null,
    content_type      text          not null,
    byte_size         bigint        not null,
    sha256            text          not null,
    storage_path      text          not null,         -- references a file in Supabase Storage
    n_rows_raw        integer,
    columns           jsonb         not null default '[]'::jsonb,
    tags              jsonb         not null default '{}'::jsonb
);
create unique index if not exists uploaded_logs_sha256_idx on uploaded_logs (sha256);

-- Generated report bundles (paths to artefacts in Supabase Storage).
create table if not exists report_bundles (
    id                uuid          primary key default gen_random_uuid(),
    experiment_id     uuid          not null references experiments(id) on delete cascade,
    created_at        timestamptz   not null default now(),
    storage_prefix    text          not null,         -- folder under the bundles/ bucket
    summary_md_path   text,
    n_figures         integer       not null default 0,
    n_tables          integer       not null default 0
);

-- Row-level security: deny all by default; the FastAPI backend uses
-- the service-role key, which bypasses RLS. If you later expose the
-- Supabase REST API directly to a browser, add explicit policies here.
alter table experiments         enable row level security;
alter table experiment_metrics  enable row level security;
alter table uploaded_logs       enable row level security;
alter table report_bundles      enable row level security;
