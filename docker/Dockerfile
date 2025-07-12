FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Two layers approach so that code can be modified wihout having to rebuild everything

# Copy API dependency files first and install dependencies
COPY api/pyproject.toml api/uv.lock ./
RUN uv sync --frozen --no-cache --no-install-project

# Copy all source code and install projects
COPY . .
RUN uv sync --frozen --no-cache

# Set entrypoint
ENTRYPOINT ["uv", "run", "main.py"]
