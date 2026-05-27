-- cdpr — initial Supabase schema.
--
-- Applied via:
--   supabase link --project-ref nohbtlhhisfiajjsbguy
--   supabase db push
--
-- Mirrors supabase/schema.sql but lives under migrations/ so the CLI
-- version-tracks it. Idempotent (everything uses IF NOT EXISTS).

create table if not exists experiments (
    id                uuid             primary key default gen_random_uuid(),
    name              text             not null,
    scenario_hash     text             not null,
    config_hash       text             not null,
    cable_mode        text             not null default 'none',
    backend           text             not null default 'cdpr',
    created_at        timestamptz      not null default now(),
    finished_at       timestamptz,
    seed              integer,
    cdpr_version      text,
    git_revision      text,
    notes             text,
    tags              jsonb            not null default '{}'::jsonb,
    local_path        text             -- absolute path to the local artefact bundle
);

create index if not exists experiments_scenario_hash_idx on experiments (scenario_hash);
create index if not exists experiments_cable_mode_idx    on experiments (cable_mode);
create index if not exists experiments_created_at_idx    on experiments (created_at desc);

create table if not exists experiment_metrics (
    id              bigserial         primary key,
    experiment_id   uuid              not null references experiments(id) on delete cascade,
    metric          text              not null,
    value           double precision  not null
);
create index if not exists experiment_metrics_exp_idx on experiment_metrics (experiment_id);

create table if not exists uploaded_logs (
    id              uuid              primary key default gen_random_uuid(),
    uploaded_at     timestamptz       not null default now(),
    filename        text              not null,
    content_type    text              not null,
    byte_size       bigint            not null,
    sha256          text              not null,
    storage_path    text              not null,
    n_rows_raw      integer,
    columns         jsonb             not null default '[]'::jsonb,
    tags            jsonb             not null default '{}'::jsonb
);
create unique index if not exists uploaded_logs_sha256_idx on uploaded_logs (sha256);

create table if not exists report_bundles (
    id              uuid              primary key default gen_random_uuid(),
    experiment_id   uuid              not null references experiments(id) on delete cascade,
    created_at      timestamptz       not null default now(),
    storage_prefix  text              not null,
    summary_md_path text,
    n_figures       integer           not null default 0,
    n_tables        integer           not null default 0
);

-- RLS: deny by default. The FastAPI backend uses the service-role key,
-- which bypasses RLS. Add policies later if you ever expose the REST
-- API directly to a browser.
alter table experiments         enable row level security;
alter table experiment_metrics  enable row level security;
alter table uploaded_logs       enable row level security;
alter table report_bundles      enable row level security;
