# NeuroDiagnostix — AI Diabetes Risk App ✅

A user-friendly web app that predicts diabetes risk using two ML models (Logistic Regression and Decision Tree). The app provides a realtime single-patient assessment and batch CSV processing with interactive 3D visualizations to help interpret model outputs.

---

## ⚡ Features

- **Single-patient prediction** via a slick FastAPI + Jinja2 UI (slider & input controls) with immediate visualization. 🔍
- **Batch processing** from CSV upload with downloadable results. 📁
- Two models supported:
  - **Logistic Regression** (uses a BMI×Age interaction term)
  - **Decision Tree (optimized)**
- Pre-trained models shipped as `models/logistic_model.joblib` and `models/decision_tree_model.joblib`. 💾
- Built with **FastAPI**, **Scikit-learn**, **Pandas**, and **Plotly**.

---

## 🧭 Quickstart (Local)

Prerequisites: Python 3.12+, and optionally Node.js for building CSS.

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install Python requirements:

```bash
pip install -r requirements.txt
```

(or with Poetry: `poetry install` — `pyproject.toml` is included)

3. Start the server:

```bash
# development server
uvicorn main:app --reload --port 8000
```

4. Open the UI in your browser: http://127.0.0.1:8000/

> Tip: FastAPI automatically exposes interactive docs at `/docs` (Swagger) and `/redoc`.

---

## 🎨 Frontend / CSS

This project uses Tailwind CSS. A `package.json` script exists for a Tailwind dev watch process, but its input path may need to match this repo's root `input.css` file.

Example (if you use npm):

```bash
# watch + build Tailwind (adjust the input path if necessary)
npm run dev
# OR directly
npx @tailwindcss/cli -i ./input.css -o ./static/css/output.css --watch
```

---

## 🧪 API Reference

### POST /api/predict

Single prediction. JSON body expects the following fields:

```json
{
  "pregnancies": 0,
  "glucose": 120.0,
  "bp": 70.0,
  "bmi": 25.0,
  "dpf": 0.5,
  "age": 30,
  "model_choice": "logistic"  // or "dt"
}
```

Response:

```json
{
  "prediction": 0,           // 0 => Healthy, 1 => Diabetic
  "probability": 12.34,     // probability in percent
  "risk_level": "Low"      // High if > 50%
}
```

You can also test it with curl:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H 'Content-Type: application/json' \
  -d '{"pregnancies":0, "glucose":120, "bp":70, "bmi":25, "dpf":0.5, "age":30, "model_choice":"logistic"}'
```

---

### POST /api/batch-predict

Upload a CSV file with columns that match the `PatientInput` fields (one row per patient). Example headers:

```
pregnancies,glucose,bp,bmi,dpf,age
```

Curl example:

```bash
curl -X POST "http://127.0.0.1:8000/api/batch-predict" \
  -F "file=@/path/to/patients.csv" \
  -F "model_choice=dt"
```

Response payload: JSON with `results`, each row contains original fields plus `Prediction` and `Probability %`.

---

## 🔬 Data & Models

- Training data snapshot: `models/train_data.csv` (and `train_data-2.csv`).
- Pretrained artifacts: `models/logistic_model.joblib`, `models/decision_tree_model.joblib`.
- Reproducible training notebook: **Colab** — https://colab.research.google.com/drive/1rn1mGLtevedWMXJJjuWh2gbDQLeD8nvk?usp=sharing (contains model training and evaluation notebooks).

If you want to retrain models locally, follow the notebook and export updated `.joblib` files to `models/`.

---

## 📋 Schema (validation)

`PatientInput` fields (via `models/schemas.py`):

- `pregnancies` (int, >= 0)
- `glucose` (float, >= 0)
- `bp` (float, >= 0)
- `bmi` (float, >= 0)
- `dpf` (float, >= 0)
- `age` (int, >= 1)
- `model_choice` ("logistic" or "dt")

---

## ✅ Development Notes & Tips

- The FastAPI app mounts static files from `static/` and templates from `templates/`.
- Plotly is used client-side for interactive 3D visualization (`templates/index.html`).
- The Logistic Regression pipeline expects a `BMI_Age_Interaction` feature; the service computes it automatically for single and batch predictions.
- Use `/docs` to inspect the API schema and try requests interactively.

---

## ☁️ Deployment & Docker (suggested)

A simple Dockerfile (example):

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 🙌 Contributing

Contributions are welcome — open issues for bugs or features and use pull requests for fixes. Include tests, linting, and brief descriptions for major changes.

---

## 📝 License

This project includes a `LICENSE` file — see it for details.

---
