# AI PlantDocBot

Plant disease classification using the PlantVillage dataset. Includes CNN vision models (MobileNet, ResNet), a BERT symptom-text classifier, a Streamlit chatbot UI, a Flask REST API, and **Gemini AI chat** for free-form plant health Q&A.

**Highlights**
- Classify plant diseases from leaf images (ResNet50, ResNet18, MobileNetV2)
- Diagnose from symptom text via BERT or rule-based fallback
- Gemini-powered conversational chat with plant disease context
- Download diagnosis reports as JSON or TXT
- Flask REST API for programmatic access
- Docker support for production deployment

## Quick Start

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Gemini API key:
```bash
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=your_key_here
```

Run the Streamlit app:
```bash
streamlit run app.py
```

Run the Flask API:
```bash
python backend/server.py
```

## Docker

```bash
docker-compose up --build
```

- Streamlit: http://localhost:8501
- Flask API: http://localhost:5001

## Gemini Chat

Add `GEMINI_API_KEY` to your `.env` file to enable the **Gemini AI Chat** mode in the sidebar. This lets users ask free-form plant health questions with context from their last diagnosis automatically injected.

## Project Structure

```
├── app.py                  # Streamlit chatbot UI
├── gemini_chat.py          # Gemini API integration
├── backend/server.py       # Flask REST API
├── data/                   # Class names, treatments, label map, symptom CSVs
├── models/                 # Trained PyTorch weights (.pth)
├── notebooks/              # Colab training notebooks
├── scripts/                # Dataset utilities
├── material/               # Project report PDF
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Flask API Endpoints

- `GET /health` — available models and status
- `POST /predict/image` — form-data: `image`, optional `model`, `top_k`
- `POST /predict/text` — JSON: `{"text": "...", "model": "resnet50"}`

## Data Setup

1. Place PlantVillage images into `data/train` and `data/test`.
2. Optional label normalization:
   ```bash
   python scripts/clean_dataset.py
   ```
3. Optional stratified validation split:
   ```bash
   python scripts/make_val_split.py --ratio 0.2 --seed 42
   ```

## Text Model (BERT)

Export the fine-tuned BERT model so the app can use it:
```bash
python scripts/export_bert.py --checkpoint results/checkpoint-855 --output_dir models/bert_symptom_classifier
```
Without it, the app falls back to rule-based text diagnosis.

## Models

Pretrained checkpoints go in `models/`:
- `plant_disease_mobilenet_v1.pth`
- `plant_disease_resnet18_v1.pth`
- `plant_disease_resnet50_v1.pth`

## Notes

- Large datasets and model weights are excluded via `.gitignore`. Add them locally.
- Notebooks are written for Google Colab. For local use, update dataset paths.
