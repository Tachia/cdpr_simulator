# Dockerfile for the FastAPI backend (cdpr.interface.api)
#
# Two-stage build keeps the runtime image small; system deps (libgomp for
# scipy LAPACK, build-essential only at build time) are isolated to the
# builder stage. The MuJoCo + PyTorch extras are NOT installed here ---
# the production backend serves simulation / workspace / plot endpoints
# that the scientific core can fulfil with just NumPy/SciPy/Matplotlib.
# Add `pip install -e ".[learn,adapters-mujoco]"` to the install step if
# the deployment needs them.

# ---- builder -----------------------------------------------------------

FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# System build deps (removed in the runtime stage).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        && rm -rf /var/lib/apt/lists/*

# Copy only metadata first to maximise layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --upgrade pip wheel \
    && pip wheel --wheel-dir /wheels ".[api,viz,data]"

# ---- runtime -----------------------------------------------------------

FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Runtime-only system deps (libgomp1 for scipy / numpy LAPACK threads).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /wheels /wheels
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

RUN pip install --no-index --find-links=/wheels ".[api,viz,data]" \
    && rm -rf /wheels

# Render injects $PORT at runtime; default to 8000 for local docker run.
ENV PORT=8000
EXPOSE 8000

# Use shell form so $PORT is expanded by the shell at start time.
CMD uvicorn cdpr.interface.api:app --host 0.0.0.0 --port "${PORT}"
