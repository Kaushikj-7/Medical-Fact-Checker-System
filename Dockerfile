# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production runtime stage
FROM python:3.12-slim AS runtime

WORKDIR /app

RUN groupadd -r craguser && useradd -r -g craguser craguser \
    && mkdir -p /app/data /app/data/chroma_db \
    && chown -R craguser:craguser /app

COPY --from=builder /root/.local /home/craguser/.local
ENV PATH=/home/craguser/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1

COPY --chown=craguser:craguser . .

USER craguser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "medical_crag.api:app", "--host", "0.0.0.0", "--port", "8000"]
