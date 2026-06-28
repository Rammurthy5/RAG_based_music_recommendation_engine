FROM python:3.12-slim

WORKDIR /app

ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install a CPU-only torch wheel first so sentence-transformers does not pull
# the much larger CUDA-enabled bundle during dependency resolution.
RUN python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

RUN python -m pip install --no-cache-dir -r requirements.txt

# Pre-download the embedding model at build time to avoid cold-start latency
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

COPY . .

RUN adduser --disabled-password --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
