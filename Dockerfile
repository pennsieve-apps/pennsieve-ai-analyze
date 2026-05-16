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

WORKDIR /app

# Source must be in place before `pip install .` because pyproject.toml's
# [tool.setuptools.packages.find] points at src/ and the build fails without it.
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

ENV PYTHONPATH=/app/src

ENTRYPOINT ["python", "-m", "analyze.main"]
