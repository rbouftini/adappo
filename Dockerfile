# Builder
FROM python:3.10-slim AS builder
WORKDIR /adappo
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Runtime 
FROM python:3.10-slim AS runtime
WORKDIR /adappo
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1
COPY . .