# Deployment status — May 28, 2026

Snapshot of the cdpr framework's live infrastructure. See
[deployment.md](deployment.md) for the platform-by-platform setup
guide; this file captures **the current connected state**.

## Connection map

```
┌──────────────────────────────────────────────────────────────────────┐
│  Local dev machine                                                    │
│  C:\Users\Jesus is Lord\Desktop\CDPR Simulator                        │
│  — editor + pytest (195 tests, ~52 s) + smoke_phase*.py + git push — │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ git push origin main
                               ▼
        ┌──────────────────────────────────────────────────────────┐
        │  GitHub:  Tachia/cdpr_simulator   (public, main branch)  │
        │  6 commits  |  277 KB  |  169 files                      │
        │  CI:  GitHub Actions / .github/workflows/ci.yml          │
        └──┬───────────────┬─────────────────┬──────────────┬──────┘
           │               │                 │              │
   webhook │       webhook │          webhook │             │ (CLI only:
           ▼               ▼                 ▼              │  no webhook)
   ┌───────────────┐  ┌────────────┐  ┌──────────────────┐  ▼
   │  Render       │  │ Streamlit  │  │  Cloudflare      │  Supabase
   │  Web Service  │  │ Community  │  │  Pages           │  Postgres
   │  cdpr-api     │  │ Cloud      │  │  cdpr-simulator  │  nohbtlhhisfiajjsbguy
   │  (Docker)     │  │ (Python)   │  │  (static)        │  (4 tables)
   └───────┬───────┘  └────────────┘  └──────────────────┘
           │
           │ optional dual-write
           │ (when SUPABASE_URL +
           │  SUPABASE_SERVICE_ROLE_KEY
           │  are set in Render env)
           └────────────────────────────────────────────► Supabase
```

## Live URLs

| Service | URL | Status | Notes |
| --- | --- | --- | --- |
| GitHub repo | <https://github.com/Tachia/cdpr_simulator> | live | public; `main` is the deployed branch |
| Render — FastAPI backend | <https://cdpr-api.onrender.com> | live | free tier sleeps after 15 min idle (~50 s cold start). `/health`, `/robots`, `/simulate`, `/workspace`, `/plot`, `/docs`, `/redoc` |
| Streamlit — research console | <https://cdprsimulator-a5u8bciz6tsnsxegg8zys2.streamlit.app> | live (owner-visible) | currently private; anonymous visitors get the sign-in screen. Set Sharing → Public if you want reviewers to access without an account. |
| Cloudflare Pages — docs | <https://cdpr-simulator.pages.dev> | live | static. `/` is the project landing, `/deployment` is the deployment guide |
| Supabase — Postgres | `db.nohbtlhhisfiajjsbguy.supabase.co` | schema applied | 4 tables: `experiments`, `experiment_metrics`, `uploaded_logs`, `report_bundles` |
| GitHub Actions CI | <https://github.com/Tachia/cdpr_simulator/actions> | failing on `pytest` step | install + lint succeed; pytest fails on both 3.11 and 3.12 jobs. Loose end — see "Remaining work" |

## Where each secret lives

| Secret | Set in | Used by | Rotated? |
| --- | --- | --- | --- |
| `SUPABASE_URL` | Render Environment tab | `cdpr.storage.supabase` when invoked from the Render-hosted backend | n/a (project URL is not secret) |
| `SUPABASE_SERVICE_ROLE_KEY` | Render Environment tab | same as above | **rotate after handoff** — was pasted in the deployment conversation |
| Supabase DB password | Supabase Studio (Settings → Database) | `supabase` CLI on local machine when running `db push` | **rotate after handoff** — was pasted in the deployment conversation |
| `CDPR_BACKEND_URL` | Streamlit Cloud secrets | reserved; not currently consumed by the GUI (which embeds cdpr locally) | not a secret |

The local checkout has **no `.env` file** — secrets are not on disk anywhere outside the cloud platforms.

## Development loop

```
edit src/cdpr/...                      # local
pytest tests/                          # local validation (~52 s)
git add ... && git commit
git push                               # → triggers GitHub Actions
                                       # → triggers Render auto-deploy (~5 min cold)
                                       # → triggers Streamlit auto-rebuild (~2 min)
                                       # → triggers Cloudflare Pages auto-deploy (~30 s)
```

For Supabase schema changes:

```
supabase migration new <name>          # creates new file under supabase/migrations/
# edit the new SQL file
supabase db push                       # applies to the linked project
git add supabase/migrations/<file>.sql
git commit -m "schema: <change>"
git push
```

The migrations directory is committed to git so the schema is
version-tracked alongside the code.

## Remaining work / loose ends

1. **GitHub Actions CI: failing on pytest** (both 3.11 and 3.12).
   Install + ruff steps succeed; pytest exits 1. Local CI subset
   passes (177/177 with the same `--ignore` flags). Linux-specific
   regression or a missing platform-specific extra is the most likely
   cause. Needs the actual log to diagnose — see the *next steps*
   section below.

2. **Streamlit visibility set to private.** The app works while you're
   signed in, but anonymous reviewers get the sign-in screen. If you
   want public access for the dissertation: Streamlit Cloud →
   **cdpr_simulator** app → ⋮ → **Settings → Sharing → Public**.

3. **Cloudflare Workers Build leftover.** The original mis-created
   Worker project was deleted, but the GitHub App that registered the
   "Workers Builds: cdpr-simulator" check is still installed. Harmless
   — the check passes on every push. Remove via GitHub → Settings →
   Integrations → Cloudflare Workers → Configure → uninstall.

4. **Rotate the Supabase password + service-role key** (both pasted in
   the deployment conversation; treat as compromised).

5. **Render free-tier cold-start.** Service sleeps after 15 min idle;
   first request takes ~50 s while it spins up. Upgrade to a paid plan
   to keep it warm, or accept the first-request penalty.

## Next steps to close the CI loop

When you have a moment:

1. Open <https://github.com/Tachia/cdpr_simulator/actions>.
2. Click the most recent failing run.
3. Click the **pytest + ruff (3.12)** job.
4. Expand the **Run pytest** step.
5. Copy the section from the first `FAILED` line through the end and
   paste it into the conversation. With the actual error text I can
   patch in minutes — without it I'm guessing.
