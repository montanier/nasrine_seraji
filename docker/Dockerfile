FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Two layers approach so that code can be modified wihout having to rebuild everything

# Copy dependency files first and install dependencies only (skip project build since README.md not copied yet)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-cache --no-install-project

# Copy all source code (including README.md required by pyproject.toml) and install the project
COPY . .
RUN uv sync --frozen --no-cache

# Set entrypoint
ENTRYPOINT ["uv", "run", "api/main.py"]
