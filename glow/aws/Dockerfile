FROM python:3.12-slim

# install system dependencies for scientific computing
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /app

# copy project files
COPY pyproject.toml poetry.lock README.md ./
COPY glow/ ./glow/

# install poetry and project dependencies
RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install --only main --no-interaction --no-ansi

# set the entrypoint to consolidated worker
ENTRYPOINT ["python", "-m", "glow.aws.worker"]
