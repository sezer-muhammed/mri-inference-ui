# MRI Inference UI

One-page demo for running 3D MRI model inference. Uses Niivue for in-browser NIfTI volume rendering and connects to the FastAPI backend at `mri-api/`.

## Start

```bash
cd mri-inference-ui
npm run dev          # → http://localhost:3000
```

The API runs on port 7860 by default. Click **API** in the header to change the base URL.

## API contract

| Endpoint | Method | What it does |
|---|---|---|
| `/inference` | POST | Accepts `file` (multipart), `model_name` (form), `label?` (form). Returns the inference result. |
| `/results` | GET | Returns `{ count, results[] }` ordered by newest first. |

## Viewer notes

- **.nii / .nii.gz** — fully supported by Niivue; previewed in-browser before inference
- **.nii.tar** — may not preview (tar extraction not supported by Niivue); inference will still work
- Switch between **4-panel**, **3D render**, **Axial**, **Coronal**, **Sagittal** views using the toolbar above the canvas

## Connect the API

1. Start `mri-api`: `uvicorn main:app --port 7860 --reload`
2. Open `http://localhost:3000`
3. Upload a .nii file, enter the model name, click **Run inference**
