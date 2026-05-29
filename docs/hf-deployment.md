# Hugging Face Space deployment — token-based (no CLI required)

This guide deploys the Gradio app to a Hugging Face Space using only
`git` and an HF access token. The `huggingface-cli` is **not**
required — it has been a source of friction (PATH issues, login
state) and is unnecessary for a one-app Space.

The local Gradio workflow (`python gradio_app.py` → `http://127.0.0.1:7860`)
is independent of this guide and never breaks when an HF push fails.

## Prerequisites

1. A Hugging Face account at <https://huggingface.co>.
2. A Space already created (one-time setup):
   * **Space name**: e.g. `cdpr-simulator`
   * **SDK**: Gradio
   * **Hardware**: CPU basic (free, 16 GB RAM)
   * **License**: MIT
3. A **write-scope access token** at
   <https://huggingface.co/settings/tokens> → *New token* → *Write*.
   Save the token (a string starting with `hf_…`).

## Why the previous deploy failed

The terminal log showed three errors stacked together:

* **Interactive rebase mid-flight, README.md conflict** — `git pull
  --rebase space main` started replaying every local commit on top of
  the empty Space, conflicting on the README the moment the first
  commit landed. With 19 commits in the queue this was not going to
  finish.
* **Non-fast-forward push rejected** — the Space's initial commit
  (the auto-generated README) was not in the local history, so a
  plain `git push space main` was refused.
* **CLI not on PATH** — irrelevant once the rebase blocked everything
  else.

The fixes are all on the git side; no CLI install is needed.

## Step 1 — make sure the repo is clean

If a rebase is mid-flight, abort it (your committed work is safe):

```powershell
git status                       # check
git rebase --abort               # only if "interactive rebase in progress"
git status                       # must say "working tree clean"
```

## Step 2 — add the Space as a remote (one-time)

```powershell
# Replace <user> and <space> with your username and Space name.
$user  = "your-hf-username"
$space = "cdpr-simulator"

git remote add space "https://huggingface.co/spaces/$user/$space"
git remote -v             # confirm 'space' appears
```

If the remote already exists from a previous attempt, the second `add`
errors — that's fine, the remote is already correct. To update its URL:

```powershell
git remote set-url space "https://huggingface.co/spaces/$user/$space"
```

## Step 3 — push with the token in the URL (the reliable path)

This is the part that avoids every CLI / credential issue:

```powershell
$user  = "your-hf-username"
$space = "cdpr-simulator"
# Paste the hf_… token here (or read it from a saved file). DO NOT
# commit the token anywhere.
$token = Read-Host "Paste your HF write token (hf_…)" -AsSecureString
$tokenPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($token))

# Force-push *the first time* to overwrite the Space's auto-README
# with this repo's README that carries the right YAML frontmatter.
# After this first push, subsequent pushes are plain ``git push``.
git push --force "https://${user}:${tokenPlain}@huggingface.co/spaces/$user/$space" main:main
```

Why force on the first push: the Space was created with one
auto-generated commit (the README HF made for you). Your local
history doesn't contain that commit, so a regular push is refused.
Force-pushing overwrites it with this repo's README — which is what
you want, because the local README has the correct YAML frontmatter:

```yaml
---
title: CDPR Simulator
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.15.2
app_file: app.py
pinned: false
---
```

That tells HF which file to run (`app.py` → which imports `demo` from
`gradio_app.py`), what SDK version to use, and the cosmetic theme.

## Step 4 — after the first force-push

From now on the remote and local share history, and ordinary pushes
work:

```powershell
git push space main
```

HF rebuilds the Space automatically (3-7 minutes for a clean build).
You can watch the build log on the Space's settings page.

## Step 5 — verify the Space

Open `https://huggingface.co/spaces/<user>/cdpr-simulator`. You
should see:

* the build banner from `gradio_app.py` (currently
  `gradio-2026-05-29-a`),
* three tabs (Built-in examples, Custom Phase-1 simulation, Upload
  CSV / Phase-2),
* clicking *Run example* on `circle` produces 14 figures + the
  feasibility JSON.

## What to do if the push still fails

| Symptom | Cause | Fix |
|---|---|---|
| `403 Forbidden` | Token is read-only, not write | Generate a new token with *Write* scope |
| `404` | Wrong username or Space name in the URL | Double-check `git remote -v` |
| `! [rejected] main -> main (non-fast-forward)` after step 4 | Someone (or you) committed directly on the Space | `git pull space main --rebase` then push again. If the only thing in the way is the README, repeat step 3. |
| `error: src refspec main does not match any` | Local branch is named `master`, not `main` | `git push space master:main --force` |

If anything else fails, the local Gradio workflow remains the source
of truth — it is independent of HF.

## Cross-references

* [docs/run-locally.md](run-locally.md) — PowerShell step-by-step for
  the local Gradio / Streamlit / Dash launchers
* [docs/frontend-architecture-revised.md](frontend-architecture-revised.md)
  — why Gradio is the chosen hosted demo
* [docs/multi-frontend.md](multi-frontend.md) — topology of the four
  coexisting frontends
