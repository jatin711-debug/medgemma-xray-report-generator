# MedGemma Chest X-ray Report Generator — Experimental Beta

> **Research and educational prototype only.** This project is experimental, unvalidated, and not intended for diagnosis, clinical decisions, patient care, or production use. Model outputs can be wrong or indeterminate. Do not upload real patient data without appropriate authorization and safeguards.

A FastAPI prototype that sends a single chest X-ray image to MedGemma and produces a PDF containing the model's pneumothorax assessment. It is a software experiment, not a validated medical device. The default model is [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it); see the [technical report](https://arxiv.org/html/2604.05081v2). The model's wider capabilities are not implemented by this application.

The response parser accepts only the requested exact finding format; other responses are marked indeterminate. Confidence scores are not calibrated. Earlier experimental attention-derived visualizations are omitted from reports because they are not validated probability or segmentation maps.

## Setup

1. Use a recent Python version (Python 3.10 or newer), install dependencies with `pip install -r requirements.txt`, and configure PyTorch for your hardware. A capable GPU is recommended; CPU inference may be slow and memory intensive.
2. Request/accept access to the model and its license on the [model page](https://huggingface.co/google/medgemma-1.5-4b-it), then authenticate with Hugging Face (for example, `hf auth login`). Model download requires network access and substantial storage.
3. Optionally copy `samplenv` to `.env`. `MODEL_ID` selects the model; most other sample settings are not currently wired into the application. The local upload workflow does not require S3 or MySQL.
4. Run `uvicorn main:app --host 127.0.0.1 --port 8000` and open `http://127.0.0.1:8000/docs`. Startup loads the model and can take time.

## API

- `POST /analyze-and-report`: multipart `file` image upload; creates a local PDF under `reports/` and returns its path and analysis ID.
- `GET /reports`: lists locally generated PDFs.
- `GET /model-status` and `POST /load-model`: inspect or reload the model.
- `POST /analyze-url` and `POST /analyze-url-and-report`: URL workflows. Disabled by default; `ENABLE_URL_FETCH=true` enables them. URL fetching is not protected against access to private/internal addresses, so enable it only in a trusted, isolated environment. The report URL workflow also requires configured AWS S3 and MySQL services and an existing `xray_reports` table; see `db.py` for its schema.

Example (PowerShell):

```powershell
curl.exe -X POST http://127.0.0.1:8000/analyze-and-report -F "file=@path/to/chest-xray.png"
```

The PDF path refers to the server filesystem; this prototype does not provide a report download endpoint. API endpoints do not implement authentication, access control, or rate limiting. Keep the server local and use non-sensitive sample data. The model, prompt, parser, output and full application have not been clinically evaluated. Results should not be interpreted as medical advice.

## Limitations and contributing

The application accepts ordinary image formats, not DICOM studies. Model access, library versions, hardware and GPU quantization can affect whether inference works. The URL-to-S3/MySQL integration is experimental and needs external services. There is no production deployment configuration or validated test dataset in this repository. Contributions and independent evaluation are welcome.

The model is distributed under its own license and access terms; this repository does not currently specify a code license.

## Code structure

`src/api/` contains the API and request/response schemas, `src/core/` contains the MedGemma analyzer, and `src/reports/` contains PDF generation. `main.py` remains the compatible server entry point. The older multi-condition and Grad-CAM work remains in the `dev` branch and is not part of this workflow.

Run the lightweight API and parser checks with `python -m unittest discover -s tests`. These checks use a mocked analyzer and do not validate model inference or clinical performance.
