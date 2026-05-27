# Deployment guide

Connection map for the cdpr framework. The local checkout is the
development workspace; GitHub is the source of truth; each external
service redeploys from a single watched branch.

```
                    ┌─────────────────────────────────────────────────┐
                    │                  Local dev machine               │
                    │   ── editor + pytest + smoke_phase*.py + git ── │
                    └────────────────────────┬────────────────────────┘
                                             │ git push
                                             ▼
            ┌──────────────────────────────────────────────────────────┐
            │   GitHub:  Tachia/cdpr_simulator   (branch: main)        │
            │   CI:  .github/workflows/ci.yml   (lint + pytest)        │
            └──┬──────────────┬─────────────────┬─────────────────┬────┘
               │              │                 │                 │
        webhook │       webhook │          webhook │          webhook │
               ▼              ▼                 ▼                 ▼
   ┌──────────────────┐  ┌──────────────┐  ┌────────────────┐  ┌──────────────┐
   │  Render          │  │  Streamlit   │  │  Cloudflare    │  │  Supabase    │
   │  cdpr-api        │  │  Community   │  │  Pages         │  │  (optional)  │
   │  FastAPI         │  │  Cloud       │  │  docs/         │  │  metadata    │
   │  Dockerfile      │  │  streamlit_  │  │  index.html    │  │  + uploads   │
   │  render.yaml     │  │  app.py      │  │                │  │              │
   └──────────────────┘  └──────────────┘  └────────────────┘  └──────────────┘
            │                    │                                     ▲
            │ /simulate /workspace /plot /robots /health               │
            └────────────────────►◄──────────────────────────────────┘
                              (HTTP, JSON)
```

## What lives where

| Layer | Hosting | What's needed |
| --- | --- | --- |
| FastAPI backend | Render (Docker web service) | Render account; connect to the GitHub repo; first deploy uses `render.yaml` |
| Streamlit console | Streamlit Community Cloud | Streamlit account (free); point at `streamlit_app.py`, set `CDPR_BACKEND_URL` to the Render URL |
| Static docs | Cloudflare Pages | Cloudflare account; connect the repo; set the output directory to `docs/`; no build command |
| Storage | Supabase (optional) | Supabase project; copy URL + service-role key into Render env |
| Local physics | MuJoCo on the dev machine | `pip install mujoco`; runs identically to local |
| CI | GitHub Actions | Free, configured in `.github/workflows/ci.yml` |

## Render — backend service (FastAPI)

1. Sign in at **dashboard.render.com**.
2. **New +** → **Blueprint** → connect this repo.
3. Render reads `render.yaml` and proposes the `cdpr-api` web service.
4. Confirm the build (Docker) and the `/health` health-check.
5. Once green, copy the public URL (e.g. `https://cdpr-api.onrender.com`)
   and set it as `CDPR_BACKEND_URL` in the Streamlit deployment env.

Render's free plan sleeps after inactivity; the `/health` endpoint is
fast and cheap, which keeps cold-starts < 30 s.

## Streamlit Community Cloud — research console

1. Sign in at **share.streamlit.io** with the GitHub account that owns
   the repo.
2. **New app** → choose this repo / `main` branch / file
   `streamlit_app.py`.
3. Under **Advanced settings → Secrets**, add:

   ```toml
   CDPR_BACKEND_URL = "https://cdpr-api.onrender.com"
   ```

4. Deploy. The first build takes a few minutes (PyTorch and SciPy are
   chunky); subsequent pushes redeploy in seconds.

## Cloudflare Pages — documentation

1. Sign in at **dash.cloudflare.com** → **Workers & Pages**.
2. **Create application** → **Pages** → **Connect to Git** → select this
   repo.
3. **Build configuration**:
   - Framework preset: **None**
   - Build command: leave empty
   - Build output directory: `docs`
4. Save and deploy. The site lives at `https://<project>.pages.dev`.

## Supabase — persistent storage (optional)

1. Sign in at **supabase.com** → **New project**.
2. SQL editor → run `supabase/schema.sql` to create the tables.
3. Copy the project URL and the service-role key into Render's env
   under `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY`.
4. The FastAPI service reads them when present; with no Supabase
   credentials configured, all storage endpoints fall back to local
   in-memory operation and the rest of the API works unchanged.

## GitHub Actions — CI

`.github/workflows/ci.yml` runs ruff + pytest on every push and PR
against `main`. It uses the lean `[dev,viz,data,api]` install — heavy
extras (torch, sb3, mujoco) are exercised locally via the smoke scripts.

## Development loop

```
edit src/cdpr/...                  # local change
pytest tests/                      # local validation (~52 s)
git add ... && git commit
git push                           # triggers GitHub Actions + Render + Streamlit
```

Render and Streamlit Community Cloud both watch `main` by default;
Cloudflare Pages does the same with its automatic-deploy setting. A
single `git push` cascades to every connected service within a couple of
minutes.

## Required env vars

See `.env.example` at the repo root for the full annotated list. The
backend reads everything from the environment — there are no hardcoded
secrets in the source.
