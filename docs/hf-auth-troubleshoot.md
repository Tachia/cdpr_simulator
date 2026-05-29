# Hugging Face deployment — fixing the `pre-receive hook declined` error

The exact error from the latest push attempt:

```
You are not authorized to push to this repo.
remote: Make sure that you are properly logged in.
remote: -------------------------------------------------------------------------
To https://huggingface.co/spaces/JoeTach/cdpr-simulator
 ! [remote rejected] main -> main (pre-receive hook declined)
error: failed to push some refs to 'https://huggingface.co/spaces/JoeTach/cdpr-simulator'
```

This is **authorization**, not authentication. HF received the
credentials but the credential's owner cannot write to
`JoeTach/cdpr-simulator`. There are three possible root causes —
diagnose them in order.

> **The agent cannot log in for you.** Each of the diagnostic
> commands below must be run in your PowerShell. After each step,
> note the output and continue.

## Diagnosis ladder

### Step A — Confirm which HF account is authenticated

```powershell
hf auth whoami
# (older versions)
huggingface-cli whoami
```

There are three possible answers:

| Output | What it means | Next step |
|---|---|---|
| `<some username>` (and *not* `JoeTach`) | You are logged in as a different account that has no write permission to `JoeTach/cdpr-simulator` | Step B: log out and back in as `JoeTach` |
| `JoeTach` | You ARE the owner. The issue is the token's scope or git credentials | Step C: re-issue a write-scope token and embed it in the URL |
| `Not authenticated` or command not found | The HF CLI isn't installed / not on PATH | Step D: skip the CLI entirely and use the URL-embedded token method |

### Step B — Log out and back in as the right account

```powershell
hf auth logout
hf auth login
# Paste your hf_… token when prompted. Token must come from the
# JoeTach account at https://huggingface.co/settings/tokens .
```

After login, verify:

```powershell
hf auth whoami      # must print JoeTach
```

Then retry the push:

```powershell
git push space main
```

### Step C — Issue a fresh write-scope token and bake it into the URL

The CLI sometimes stores a stale or read-only token. The most
reliable fix is to bypass CLI credential storage entirely:

1. Generate a NEW token at <https://huggingface.co/settings/tokens>
   * Click **New token**
   * Name: `cdpr-deploy`
   * Type: **Write**
   * Click **Generate**
   * Copy the token (`hf_…`)

2. Push with the token in the URL — single command, never stored:

```powershell
$user  = "JoeTach"
$space = "cdpr-simulator"
$token = Read-Host "Paste hf_… write token" -AsSecureString
$tokenPlain = [Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [Runtime.InteropServices.Marshal]::SecureStringToBSTR($token))

# First push: force-overwrite the Space's auto-README with this repo's
# README (which has the right YAML frontmatter for app_file: app.py).
git push --force "https://${user}:${tokenPlain}@huggingface.co/spaces/$user/$space" main:main

# After the first force push, subsequent pushes are plain:
git push space main
```

### Step D — CLI missing / PATH issues

You do not need the HF CLI to deploy. Skip directly to Step C above.
It uses only `git`, which you already have.

## Verifying after a successful push

```powershell
# Open the Space to watch the build
Start-Process "https://huggingface.co/spaces/JoeTach/cdpr-simulator"
```

The build log appears on the Space's *Logs* tab. A clean run shows
~3-6 minutes of:

1. `Cloning repo …`
2. `Building image …` (pip install -r requirements-gradio.txt)
3. `Booting server …` (`python app.py`)
4. `Application startup complete.`
5. The Gradio UI shows the build banner `gradio-2026-05-29-a`.

## If the build itself fails after the push succeeds

Different failure mode from the push — the push worked but HF can't
boot the app. Check the Logs tab for the actual traceback:

| Build error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: gradio` | `requirements-gradio.txt` not used | Confirm the Space SDK is set to **Gradio** (not Docker). Settings → SDK |
| `ImportError: cdpr` | `requirements-gradio.txt` doesn't include `.` | Confirm the last line of `requirements-gradio.txt` is just `.` (it should be in this repo) |
| `app_file: app.py` not found | README YAML wrong | Check `head -10 README.md` shows `app_file: app.py` |
| Build OOM | Should not happen on 16 GB free CPU-Basic | Check Settings → Hardware |

## What the agent cannot do

* It cannot create the token for you.
* It cannot log in on your behalf.
* It cannot transfer ownership of a Space to your account.
* It cannot accept the Space's terms of service for you.

Anything else — including the push command itself once you have the
token — is mechanical and can be done by copying the block in Step C
into your PowerShell.

## Cross-references

* [docs/hf-deployment.md](hf-deployment.md) — first-time Space setup
* [docs/run-locally.md](run-locally.md) — local Gradio workflow (works independently of HF)
