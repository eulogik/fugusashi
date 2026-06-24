FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src/ ./src/
COPY config.yaml ./

RUN pip install --no-cache-dir -e .

EXPOSE 6060

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:6060/health')" || exit 1

CMD ["fugusashi", "serve", "--config", "config.yaml", "--host", "0.0.0.0", "--port", "6060"]
