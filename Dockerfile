# pennsieve-ai-analyze — analyze-stage processor image.
#
# This container runs Claude Code under the Claude Agent SDK to do
# data-aware notebook generation + execution + iteration. The
# claude-agent-sdk package includes its own copy of the Claude Code
# program, so the image does not need Node.js. The SDK version is pinned
# in pyproject.toml, and that also fixes the Claude Code version.
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

# Local builds can use a local copy of pennsieve-ai-utils instead of the
# version on PyPI. To do that, build with:
#   docker build --build-context utils=../pennsieve-ai-utils ...
# Without that option, this stage is empty and the PyPI version is used.
FROM scratch AS utils

FROM python:3.12-slim

# curl lets the agent fetch public files from inside its Bash tool.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Source must be in place before `pip install .` because pyproject.toml's
# [tool.setuptools.packages.find] points at src/ and the build fails without it.
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .
COPY --from=utils / /tmp/pennsieve-ai-utils/
RUN if [ -f /tmp/pennsieve-ai-utils/pyproject.toml ]; then \
        pip install --no-cache-dir /tmp/pennsieve-ai-utils; \
    fi && rm -rf /tmp/pennsieve-ai-utils
RUN python -m ipykernel install --sys-prefix --name python3

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "-m", "analyze.main"]
