FROM python:3.12-slim

WORKDIR /app

# System deps for psycopg2 and lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[dev]"

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "anomaly_detection.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
