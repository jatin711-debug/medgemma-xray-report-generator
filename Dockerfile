FROM python:3.12-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    HF_HOME=/home/app/.cache/huggingface
WORKDIR /app

COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir --only-binary=:all: --require-hashes -r requirements.txt \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /app/reports /home/app/.cache/huggingface \
    && chown -R app:app /app /home/app/.cache
COPY --chown=app:app main.py db.py aws_utils.py ./
COPY --chown=app:app src ./src
COPY --chown=app:app assets ./assets
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=3)" || exit 1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
