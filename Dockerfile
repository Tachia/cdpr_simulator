# Dockerfile for the FastAPI backend (cdpr.interface.api).
#
# Single-stage build. The earlier two-stage variant pre-built wheels
# under /wheels then ran `pip install --no-index --find-links=/wheels .`
# at runtime --- but the project itself uses hatchling as its build
# backend, and `--no-index` prevented pip from fetching it. This version
# installs everything in one pass with `--prefer-binary` so pip takes
# pre-built wheels from PyPI for scipy / numpy / matplotlib / pillow /
# pydantic-core rather than trying to build them from source on Render's
# small free-tier builder.
#
# Python 3.11 chosen for the broadest wheel coverage on PyPI; 3.12 also
# works but a handful of transitive deps still lag.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Runtime system deps. libgomp1 = scipy/numpy LAPACK OpenMP threads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        && rm -rf /var/lib/apt/lists/*

# Package metadata + source. Order maximises layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/

# Install. --prefer-binary keeps pip from compiling scipy / matplotlib
# from source on a memory-constrained builder.
RUN pip install --upgrade pip wheel setuptools \
    && pip install --prefer-binary ".[api,viz,data]"

# Render injects $PORT at runtime; default to 8000 locally.
ENV PORT=8000
EXPOSE 8000

# Shell form so ${PORT} is expanded at start time.
CMD uvicorn cdpr.interface.api:app --host 0.0.0.0 --port "${PORT}"
