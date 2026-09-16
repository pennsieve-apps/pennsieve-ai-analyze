# pennsieve-ai-analyze — analyze-stage processor image.
#
# This container runs Claude Code under the Claude Agent SDK to do
# data-aware notebook generation + execution + iteration. The Claude
# Code CLI is a Node.js binary, so the image carries both Python (for
# the SDK + analysis stack) and Node (for the CLI itself).
#
# Runtime contract (handled entirely by the platform, no Dockerfile
# changes required of processor authors):
#   - The processor runs as uid:gid 1000:1000, forced by the ECS task
#     definition. No USER directive needed here.
#   - HOME=/tmp, ANTHROPIC_BASE_URL=http://localhost:8765, and the
#     sidecar that bridges to the LLM governor are all injected by
#     the provisioner. We don't configure them in the image.
#
# This Dockerfile is intentionally indistinguishable from what an
# external processor author would write — that's the contract.

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
RUN python -m ipykernel install --sys-prefix --name python3

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "analyze.main"]
