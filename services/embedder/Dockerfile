FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-build.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY build/ build/

RUN python -m build.prepare_model

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements-app.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY --from=builder /build/bge-m3/ ./bge-m3/

COPY app/ app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
