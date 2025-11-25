# Kosmos Execution Sandbox
# Secure, isolated environment for running generated experiment code
#
# Build: docker build -t kosmos-sandbox:latest .
# Run:   docker run --rm kosmos-sandbox:latest python3 -c "import pandas; print('OK')"

FROM python:3.11-slim

LABEL maintainer="kosmos-ai"
LABEL description="Sandboxed execution environment for Kosmos AI Scientist experiments"
LABEL version="2.0"

# Set working directory
WORKDIR /workspace

# Install system dependencies for scientific computing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    libhdf5-dev \
    libxml2-dev \
    libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Create non-root user for code execution (security)
RUN useradd -m -u 1000 -s /bin/bash sandbox && \
    chown -R sandbox:sandbox /workspace && \
    mkdir -p /home/sandbox/.local && \
    chown -R sandbox:sandbox /home/sandbox

# Switch to non-root user
USER sandbox

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    HOME=/home/sandbox \
    PATH="/home/sandbox/.local/bin:$PATH"

# Health check to verify scientific packages
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import pandas; import numpy; import scipy; print('OK')" || exit 1

# Default command (will be overridden by sandbox)
CMD ["python3", "--version"]
