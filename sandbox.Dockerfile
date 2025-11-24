FROM ghcr.io/astral-sh/uv:python3.13-trixie-slim

COPY sandbox.toml /pyproject.toml
RUN uv venv && uv sync

ENTRYPOINT ["tail", "-f", "/dev/null"]
