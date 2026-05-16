# pennsieve-ai-analyze — analyze-stage processor image.
#
# This container runs Claude Code under the Claude Agent SDK to do
# data-aware notebook generation + execution + iteration. The Claude
# Code CLI is a Node.js binary, so the image carries both Python (for
# the SDK + analysis stack) and Node (for the CLI itself).
#
# Auth contract: the platform injects ANTHROPIC_BASE_URL=http://localhost:8765
# into the user container's env (the in-task sidecar). The Claude Agent
# SDK respects ANTHROPIC_BASE_URL and routes all its HTTP traffic at the
# sidecar, which bridges to the governor via lambda:Invoke. We don't
# need any AWS SDK config inside this container — the sidecar handles it.

FROM python:3.12-slim

# Install Node.js (for the Claude Code CLI) and a small amount of build
# tooling needed by some Python wheels.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install the Claude Code CLI globally so `claude` resolves on PATH.
# The Claude Agent SDK shells out to this binary.
RUN npm install -g @anthropic-ai/claude-code

# Pre-create a writable HOME for UID 1000. Reason: at runtime the
# Claude Code agent routinely needs to `pip install <specialty-pkg>` to
# test a hypothesis (e.g., `lifelines`, `pynwb`, domain-specific deps
# we haven't pre-installed). With a non-root container, `pip install`
# without flags fails on the root-owned site-packages; `pip install
# --user` then falls back to $HOME/.local — but only if HOME is set
# and writable. This setup makes both root-mode (today) and
# UID-1000-mode (post-non-root migration) work transparently.
RUN useradd -u 1000 -m -s /bin/bash agent && \
    mkdir -p /home/agent/.local/bin && \
    chown -R 1000:1000 /home/agent
ENV HOME=/home/agent
ENV PATH=/home/agent/.local/bin:${PATH}

WORKDIR /app

# Source must be in place before `pip install .` because pyproject.toml's
# [tool.setuptools.packages.find] points at src/ and the build fails without it.
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "analyze.main"]
