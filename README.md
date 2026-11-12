# MedGemma Chest X-ray Report Generator

[![Tests](https://github.com/jatin711-debug/medgemma-xray-report-generator/actions/workflows/tests.yml/badge.svg)](https://github.com/jatin711-debug/medgemma-xray-report-generator/actions/workflows/tests.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
![Status: experimental](https://img.shields.io/badge/status-experimental-orange)

A FastAPI service that sends a chest X-ray image to [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it) and renders the model's pneumothorax assessment as a PDF report.

> [!WARNING]
> **Research and educational prototype only.** This project is unvalidated and is not a medical device. Do not use it for diagnosis, clinical decisions, or patient care. Model outputs can be wrong or indeterminate, and confidence scores are not calibrated. Use synthetic or appropriately authorized, non-sensitive images only.

## Features

- **Single-image analysis** of JPEG, PNG, TIFF, BMP or WebP chest X-rays (DICOM is not supported).
- **Strict output parsing:** responses that don't match the requested finding format are marked *indeterminate* rather than guessed.
- **PDF reports** that are written to a temporary file and published atomically, so a failed build never leaves a partial report.
- **Role-based API keys** (reader / operator / admin) that fail closed when configuration is missing or weak.
- **Input safeguards:** 10 MiB / 25-megapixel image limits, a request-body cap, and one inference at a time per process.
- **Optional URL workflow** with SSRF protection, plus S3 upload and MySQL logging (disabled by default).
- **Reproducible setup:** hash-locked dependencies, a non-root read-only container, and CI.

## Quick start

**Requirements:** Python 3.12, a Hugging Face account with access to the gated MedGemma model, and ideally a CUDA GPU (CPU inference works but is slow and memory-hungry).

```shell
# 1. Install locked dependencies
python -m pip install --only-binary=:all: --require-hashes -r requirements.txt

# 2. Accept the model terms on Hugging Face, then authenticate
hf auth login

# 3. Configure: copy the sample env file and set API_KEY_ADMIN
cp samplenv .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # use the output as API_KEY_ADMIN

# 4. Run (startup downloads and loads the model, which takes a while)
python app.py            # or: uvicorn main:app --host 127.0.0.1 --port 8000
```

`app.py` accepts `--host`, `--port`, `--reload`, `--quantize` and `--no-quantize`.

Or with Docker, after setting `API_KEY_ADMIN` and `HF_TOKEN`:

```shell
docker compose up --build
```

## Usage

Every route except `/health/live` requires an `X-API-Key` header.

```powershell
curl.exe -X POST http://127.0.0.1:8000/analyze-and-report `
  -H "X-API-Key: $env:MEDGEMMA_API_KEY" `
  -F "file=@path/to/chest-xray.png"
```

The response includes an analysis ID and the path of the generated PDF on the server (there is no download endpoint).

| Method | Route | Role | Description |
| --- | --- | --- | --- |
| `GET` | `/health/live` | none | Process liveness |
| `GET` | `/` | reader | Service health summary |
| `GET` | `/health/ready` | reader | Model readiness |
| `GET` | `/reports` | reader | List completed local PDF reports |
| `GET` | `/model-status` | reader | Model load state |
| `POST` | `/analyze-and-report` | operator | Upload an image, analyze it, write a PDF to `reports/` |
| `POST` | `/load-model` | admin | Reload the model |
| `POST` | `/analyze-url` | operator | Analyze an image from a URL *(requires `ENABLE_URL_FETCH=true`)* |
| `POST` | `/analyze-url-and-report` | operator | URL analysis + S3 upload + MySQL record *(also requires AWS and MySQL)* |

Roles are shared service credentials, not user accounts: all keys see the same report namespace. Interactive API docs are off by default (`ENABLE_API_DOCS=true` enables them; use only on a trusted network).

## Configuration

All settings are read from the environment or `.env`; [`samplenv`](samplenv) documents each one.

| Variable | Default | Purpose |
| --- | --- | --- |
| `API_KEY_ADMIN` | **required** | Admin key (≥ 32 chars, unique). `API_KEY_OPERATOR` / `API_KEY_READER` are optional. Each can be read from a file via `API_KEY_<ROLE>_FILE`. |
| `MODEL_ID` | `google/medgemma-1.5-4b-it` | Hugging Face model to load |
| `MEDGEMMA_QUANTIZE` | `auto` | `auto` = 4-bit on CUDA and full precision on CPU; `1` / `0` force it on or off |
| `ENABLE_API_DOCS` | `false` | Expose `/docs` and `/openapi.json` |
| `ENABLE_URL_FETCH` | `false` | Enable the URL routes |
| `MYSQL_*`, `AWS_REGION`, `AWS_S3_BUCKET_NAME` | unset | Only needed for `/analyze-url-and-report`; see `db.py` for the `xray_reports` table schema |

## Testing

```shell
python -m unittest discover -s tests -v
```

The suite covers image and URL validation, auth, limits, API behaviour, concurrency, PDF output and failed-report cleanup. Inference is mocked, so it doesn't need a GPU or model download. Parser and report tests don't even import PyTorch:

```shell
python -m unittest discover -s tests -p "test_findings.py" -v
python -m unittest discover -s tests -p "test_reports.py" -v
```

Opt-in tests against local MySQL and S3-compatible storage (`RUN_STORAGE_INTEGRATION=1`) and a real model smoke test (`RUN_MODEL_SMOKE=1`) are described in [docs/deployment.md](docs/deployment.md).

## Project structure

```text
main.py                 ASGI entry point (uvicorn main:app)
app.py                  CLI launcher
src/api/                FastAPI app, schemas, API-key auth, request limits
src/core/               MedGemma analyzer and findings parser
src/services/           Image upload and URL validation
src/reports/            PDF rendering and atomic report storage
db.py, aws_utils.py     Optional MySQL and S3 integration
tests/                  Unit, API and opt-in integration tests
docs/deployment.md      Access control, containers, integration tests, dependency updates
```

## Limitations

- The model, prompt, parser and full application have **not been clinically evaluated**.
- Only pneumothorax is assessed; MedGemma's broader capabilities aren't used. Earlier attention-map visualizations were removed because they aren't validated localizations.
- The S3 and MySQL steps aren't transactional, so a database failure after an upload can leave an unreferenced S3 object.
- There's no distributed rate limiting, per-user isolation or audit trail. Keep the service on a trusted network behind a TLS proxy.

See [docs/deployment.md](docs/deployment.md) for deployment guidance. Contributions and independent evaluations are welcome.

## License

The code in this repository is released under the [MIT License](LICENSE).

This license doesn't cover third-party components. MedGemma is subject to Google's [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms), and the fonts in `assets/` remain under their original licenses.
