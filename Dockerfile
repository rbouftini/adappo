FROM python:3.9-slim

WORKDIR /ppoh

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    swig \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .