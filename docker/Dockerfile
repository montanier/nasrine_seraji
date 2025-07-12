FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Two layers approach so that code can be modified without having to rebuild everything

# Copy property_friends dependency files first and install dependencies only
COPY property_friends/pyproject.toml property_friends/uv.lock ./
RUN uv sync --frozen --no-cache --no-install-project

# Copy property_friends source code and install the project
COPY property_friends/ .
RUN uv sync --frozen --no-cache

# Set working directory to property_friends for commands
WORKDIR /app
