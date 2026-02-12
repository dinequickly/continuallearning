FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md /app/
COPY continualcode /app/continualcode
COPY examples /app/examples

RUN python -m pip install --upgrade pip && \
    printf "torch==2.5.1+cpu\n" > /tmp/constraints.txt && \
    python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -c /tmp/constraints.txt .

EXPOSE 8765

CMD ["sh", "-lc", "continualcode run_mode=web web_host=0.0.0.0 web_port=${PORT:-8765} model_name=${MODEL_NAME:-moonshotai/Kimi-K2.5}"]
