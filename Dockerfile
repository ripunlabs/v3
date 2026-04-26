# Hugging Face Spaces / OpenEnv: keep this Dockerfile simple so builds show steady pip progress
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    AACE_TASK=easy

WORKDIR /app

# Install dependencies first (large layer with visible log output on HF before app COPY)
COPY pyproject.toml requirements.txt README.md ./
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Application source (non-editable install below)
COPY . .

# Install package without re-resolving dependencies (already satisfied above)
RUN pip install --no-cache-dir --no-deps .

EXPOSE 7860

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"]
