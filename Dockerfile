FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libopus-dev \
    libvpx-dev \
    pkg-config \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml .
RUN pip install --no-cache-dir .

COPY app/ app/

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]