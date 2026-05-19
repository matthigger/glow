FROM python:3.12-slim

# unbuffer stdout/stderr so worker progress shows up in CloudWatch in
# real time rather than only on flush / process exit.
#
# Pin BLAS/threading to 1 thread per worker.  AWS Batch jobs run with
# vcpus=1 but the container sees the full host CPU count via
# /proc/cpuinfo, so without these caps OpenBLAS/OpenMP spawn N threads
# (where N = host vCPUs).  Multiple co-located workers then oversubscribe
# the host, causing ~9x slowdown on large einsums.
ENV PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1

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
