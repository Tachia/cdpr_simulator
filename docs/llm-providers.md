# LLM provider setup

The simulator's chat / explanation / conversational-simulation-builder
features go through `cdpr.llm`, a provider-agnostic layer that you
configure via environment variables. **No provider is required** —
the echo stub keeps everything functional when no API key is set, so
the LLM features degrade gracefully rather than crashing.

| Provider | Best for | Cost | Setup difficulty |
|---|---|---|---|
| **Gemini** | Default assistant — explanations, follow-ups | Free tier | easy |
| **OpenRouter (DeepSeek-R1)** | Mathematical reasoning, kinematics, RL design | Free tier | easy |
| **OpenRouter (DeepSeek-V3)** | Code generation, scripts, FastAPI endpoints | Free tier | easy |
| **Ollama** | Fully local, offline, private | $0 | medium (download install) |
| **LM Studio** | Local with a GUI manager | $0 | medium (download install) |
| **echo** | Stub for CI / when no provider is set | $0 | none |

## Step 1 — Decide which provider you want

Pick exactly one to start with. **Gemini is the recommended default**
because it is free, fast, and has the best long-context window for
explaining simulation outputs.

## Step 2 — Provider-specific setup

### Gemini (Google AI Studio — recommended default)

> The agent CANNOT do these steps for you. Each one needs your hands.

1. Create a Google account if you don't have one: <https://accounts.google.com/signup>
2. Open Google AI Studio: <https://aistudio.google.com/>
3. Accept Google's AI terms of service.
4. Click **Get API key** → **Create API key in new project**.
5. **Copy the key** and keep it private. Modern keys start with
   `AIzaSy…` (classic format) or `AQ.Ab…` (the v2 format introduced
   in 2025). Both authenticate the same way.
6. In the project root, copy `.env.example` to `.env` and add:
   ```
   GEMINI_API_KEY=<paste your key>
   ```
7. Verify with the smoke script:
   ```powershell
   pip install python-dotenv             # picks up .env automatically
   python scripts\test_llm.py --provider gemini
   ```

**Model choice.** The default is `gemini-2.0-flash`. Google retired
the `gemini-1.5-flash` and `gemini-1.5-pro` models in mid-2025; calls
to those endpoints now return HTTP 404 with a misleading "model not
found" message. If you see a 404 from this provider, your
`GEMINI_MODEL` env var is probably still pinned to the retired name
--- delete the override (or update to one of):

* `gemini-2.0-flash` --- general purpose, 1 M context, free tier (default)
* `gemini-2.0-flash-lite` --- faster, smaller, free tier
* `gemini-2.5-flash` --- newer, free tier where available

The key is sent via the `x-goog-api-key` HTTP header rather than the
URL query string, so a 404 / 401 message that quotes the request URL
will *not* leak your key into the chat box or the build log.

### OpenRouter (DeepSeek-R1 + DeepSeek-V3)

> The agent CANNOT do these steps for you.

1. Create an account at <https://openrouter.ai/>
2. **Settings → Keys** → **Create Key** → copy the key (`sk-or-…`).
3. Add to `.env`:
   ```
   OPENROUTER_API_KEY=sk-or-...your_key...
   # Optional model override (default is DeepSeek-R1 for math reasoning):
   # OPENROUTER_MODEL=deepseek/deepseek-r1
   # OPENROUTER_MODEL=deepseek/deepseek-chat   # DeepSeek-V3 for code
   ```
4. Verify:
   ```powershell
   python scripts\test_llm.py --provider openrouter
   ```

### Ollama (fully local)

> The agent CANNOT install software on your machine.

1. Download Ollama: <https://ollama.com/download>
2. Install and verify:
   ```powershell
   ollama --version
   ```
3. Pull a model the directive recommends:
   ```powershell
   ollama pull deepseek-r1        # math reasoning
   ollama pull llama3             # general assistant
   ollama pull mistral            # smaller, faster
   ```
4. Ollama auto-starts a daemon at `http://localhost:11434`.
5. Add to `.env`:
   ```
   OLLAMA_URL=http://localhost:11434
   OLLAMA_MODEL=deepseek-r1
   ```
6. Verify:
   ```powershell
   python scripts\test_llm.py --provider ollama
   ```

### LM Studio (local with GUI)

> The agent CANNOT install software on your machine.

1. Download LM Studio: <https://lmstudio.ai/>
2. Install and launch.
3. Use the in-app browser to download a model (e.g. `deepseek-r1`).
4. Open the **Local Server** tab and click **Start Server**. Default
   endpoint is `http://localhost:1234/v1`.
5. Add to `.env`:
   ```
   LMSTUDIO_URL=http://localhost:1234/v1
   LMSTUDIO_MODEL=local-model
   ```
6. Verify:
   ```powershell
   python scripts\test_llm.py --provider lmstudio
   ```

## Step 3 — Pin the active provider (optional)

The factory picks the first provider whose keys are present in this
order: Gemini → OpenRouter → Ollama → LM Studio → echo. To force a
specific one regardless of which keys exist:

```
CDPR_LLM_PROVIDER=openrouter
```

### Automatic fallback chain

The conversational simulation builder doesn't stop at one provider —
it walks a chain in order until one returns a parseable response.
Default chain (when `CDPR_LLM_FALLBACK_CHAIN` is unset): every
provider whose env var is configured, in the priority order above,
with the echo stub appended at the end so the call never raises.

Pin a custom chain via env var:

```
CDPR_LLM_FALLBACK_CHAIN=gemini,openrouter,ollama,echo
```

Why this matters: Gemini's free tier rate-limits aggressively
(HTTP 429). When the limit hits, the chain rolls over to OpenRouter,
parses the description there, and the user never sees the failure
in the form — only in the chat reply's *Notes* section, which lists
which provider answered and which providers it tried first.

## Step 4 — Deployment secrets

Never commit `.env`. The repo's `.gitignore` already excludes it.

For each hosting platform, add the key as a **secret / environment
variable** in the platform's UI:

| Platform | Where |
|---|---|
| Hugging Face Space | Space → **Settings** → **Repository secrets** |
| Render | Service → **Environment** → **Add Environment Variable** |
| GitHub Actions | Repo → **Settings** → **Secrets and variables → Actions** |
| Streamlit Cloud | App → **Settings** → **Secrets** |

## Using the LLM layer from Python

```python
from cdpr.llm import build_provider, LLMMessage

llm = build_provider()         # picks up the active config
resp = llm.complete([
    LLMMessage(role="system", content="You are a concise assistant."),
    LLMMessage(role="user",   content="Explain cable tension distribution in one sentence."),
])
print(resp.text)
```

Force a specific provider:

```python
llm = build_provider("openrouter")
```

## Conversational simulation builder

The directive's example — *"Simulate an 8-cable CDPR carrying 50 kg
following a figure-eight trajectory…"* — is handled by
`cdpr.llm.simulation_builder.describe_to_request`:

```python
from cdpr.llm.simulation_builder import describe_to_request

result = describe_to_request(
    "Simulate an 8-cable CDPR carrying 50 kg following a horizontal "
    "circle of radius 0.05 m for 10 seconds."
)
print(result.request.robot)               # 'dissertation_8cable'
print(result.request.trajectory.kind)     # 'circle'
print(result.confidence)                  # 'high' or 'low'
print(result.follow_up_questions)         # ['What payload mass?'] if ambiguous
```

The builder NEVER crashes:

* If the LLM is unreachable, returns a conservative default request
  and reports the failure in `result.notes`.
* If the LLM response isn't valid JSON, falls back to regex-based
  intent detection from the description.
* If required fields are ambiguous, the LLM is instructed to populate
  `follow_up_questions` — the calling UI surfaces these to the user.

## What can be automated and what can't

| Action | Who does it |
|---|---|
| Provider abstraction code | agent |
| Environment-variable loading | agent |
| Smoke / test scripts | agent |
| Integration into Gradio / Dash / Streamlit | agent (next sprint) |
| **Account creation on Google AI Studio / OpenRouter / HF** | **user** |
| **Accepting platform terms of service** | **user** |
| **API key generation** | **user** |
| **Login on `huggingface-cli`** | **user** |

The directive's stop-and-request rule applies whenever an action
requires user authentication.
