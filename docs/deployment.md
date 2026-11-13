# Deployment and testing

The application is an experimental research prototype, not a validated medical device or production-ready system. Use synthetic or appropriately authorized, non-sensitive data.

## Access control

All API routes except `/health/live` require `X-API-Key`. Generate a long random key for each role with Python's `secrets.token_urlsafe(32)`. Configure `API_KEY_ADMIN` before starting the server. Optionally set distinct `API_KEY_OPERATOR` and `API_KEY_READER`; keys must be unique, at least 32 non-whitespace ASCII characters, and generated with a cryptographically secure random generator. Startup fails closed if configuration is missing or invalid. Keys can instead be read from mounted files via `API_KEY_ADMIN_FILE`, `API_KEY_OPERATOR_FILE`, and `API_KEY_READER_FILE`; do not set both the value and file variables for the same role. Never commit keys or use the example test credentials outside isolated integration tests.

| Role | Access |
| --- | --- |
| Reader | Health summary, report list, model status |
| Operator | Reader access plus image analysis and report creation |
| Admin | Operator access plus model reload |

These are shared service credentials, not individual user accounts. All authorized users access the same report namespace; there is no per-user ownership, per-tenant isolation, revocation service, or audit trail. Rotate a key by changing its environment variable or mounted secret file; environment changes require a restart. Keep TLS termination and authentication logs at a trusted proxy, enforce rate limits and request restrictions there, and restrict network access. The app's one-at-a-time inference guard is per process, not a distributed rate limiter. Body size is capped at 11 MiB. By default, API documentation is disabled; set `ENABLE_API_DOCS=true` only on a trusted network. The unauthenticated `/health/live` endpoint reports process liveness. `/health/ready` requires at least reader access and checks model readiness, but not storage integrations. URL filtering also rejects common IPv6 translation and tunneling prefixes; use network-level egress restrictions for any enabled URL input.

## Container

`Dockerfile` installs hash-locked dependencies, uses a non-root account, and exposes port 8000. `compose.yaml` binds the port to loopback on the host, uses a read-only root filesystem and writable volumes for reports and the Hugging Face cache, and runs one worker. The default Compose configuration is CPU-oriented; GPU use requires compatible host drivers and a separately verified NVIDIA container setup. Do not expose the service directly to the internet. Provide TLS, authorization policy, resource limits, and egress filtering at your deployment boundary.

After setting `API_KEY_ADMIN` and `HF_TOKEN` in your environment or a private `.env` file, start with `docker compose up --build`. The model may require gated access and a large initial download. No secrets should be baked into images. Reports and model-cache volumes persist until deliberately removed. Container startup loads the model; the liveness probe does not provide a complete readiness or storage health check.

## Storage integration tests

The integration setup starts a dedicated local MySQL database and S3-compatible SeaweedFS instance with test-only credentials and loopback ports. It does not contact AWS. With Docker Compose running:

```powershell
docker compose -f compose.integration.yaml up -d
$env:RUN_STORAGE_INTEGRATION = '1'
python -m unittest discover -s tests -p test_storage_integration.py -v
Remove-Item Env:RUN_STORAGE_INTEGRATION
docker compose -f compose.integration.yaml down --volumes
```

The test is opt-in, permits only loopback endpoints and the dedicated test database, writes synthetic content under a unique key, and removes its own object and row. The Compose cleanup command removes the integration stack's data; run it only for this isolated stack. These tests do not exercise a live AWS account, real credentials, TLS, IAM, encryption configuration, or a model.

## Optional model smoke test

No model runs during the regular test suite. To verify a gated model download and real inference later, explicitly set `RUN_MODEL_SMOKE=1` and `MODEL_SMOKE_IMAGE` to a non-sensitive local sample image, authenticate to Hugging Face, then run `python -m unittest discover -s tests -p test_model_smoke.py -v`. This checks that inference returns a response, that the attribution map is produced or reported as unavailable, and that a PDF renders; it does not establish accuracy or clinical validity.

The attribution map adds one forward and backward pass through the language model, which takes extra time and memory beyond generation. If GPU memory is tight, set `MEDGEMMA_ATTRIBUTION=0`; failures are logged and the report notes the map is unavailable.

## Dependency updates

Python 3.12 is the baseline for `requirements.lock`. Install with `python -m pip install --only-binary=:all: --require-hashes -r requirements.txt`. Change direct versions in `requirements.in`, then regenerate with `uv pip compile requirements.in --universal --python-version 3.12 --generate-hashes --output-file requirements.lock` and rerun the checks. Standard PyPI PyTorch packages can be large; GPU-specific or alternate-platform wheels may need a separate, tested lock file. The lock improves repeatability and integrity but does not replace vulnerability review or testing on the target runtime.
