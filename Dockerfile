FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

RUN curl -sSL https://install.python-poetry.org | python - && \
    ln -s "${POETRY_HOME}/bin/poetry" /usr/local/bin/poetry

WORKDIR /app

COPY pyproject.toml poetry.lock* ./

RUN poetry install --no-root --only main

COPY . .