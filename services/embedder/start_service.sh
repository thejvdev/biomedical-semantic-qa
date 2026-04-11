#!/bin/sh

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

python -m uvicorn app.main:app --host 0.0.0.0 --port 8001

