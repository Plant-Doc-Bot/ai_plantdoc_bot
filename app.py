import contextlib
import csv
import json
import os
from datetime import datetime
from pathlib import Path

# Load .env before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from gemini_chat import (
    chat_with_gemini,
    enhance_diagnosis,
    enhance_multi_diagnosis,
    get_followup_chips,
    get_gemini_client,
    streamlit_history_to_gemini,
    _are_same_plant,
)


ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
TEXT_MODEL_DIR = MODELS_DIR / "bert_symptom_classifier"
TEXT_LABELS_PATH = TEXT_MODEL_DIR / "labels.json"


VISION_MODEL_FILES = {
    "ResNet-50": "plant_disease_resnet50_v1.pth",
    "ResNet-18": "plant_disease_resnet18_v1.pth",
    "MobileNetV2": "plant_disease_mobilenet_v1.pth",
}


def _safe_read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def load_class_names(expected_count: int):
    json_path = DATA_DIR / "class_names.json"
    txt_path = DATA_DIR / "class_names.txt"
    train_dir = DATA_DIR / "train"
    data_cleaned_dir = DATA_DIR / "data_cleaned"

    if json_path.exists():
        data = _safe_read_json(json_path)
        if isinstance(data, list) and len(data) == expected_count:
            return data, "data/class_names.json"

    if txt_path.exists():
        names = [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(names) == expected_count:
            return names, "data/class_names.txt"

    if train_dir.exists():
        names = sorted([p.name for p in train_dir.iterdir() if p.is_dir()])
        if len(names) == expected_count:
            return names, "data/train/*"

    if data_cleaned_dir.exists():
        names = sorted([p.name for p in data_cleaned_dir.iterdir() if p.is_dir()])
        if len(names) == expected_count:
            return names, "data/data_cleaned/*"

    fallback = [f"Class_{i}" for i in range(expected_count)]
    return fallback, "generated fallback"


def infer_num_classes(state_dict: dict):
    if "classifier.1.weight" in state_dict:
        return state_dict["classifier.1.weight"].shape[0]
    if "fc.weight" in state_dict:
        return state_dict["fc.weight"].shape[0]
    return None


def build_model(arch: str, num_classes: int):
    if arch == "MobileNetV2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        return model
    if arch == "ResNet-18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model
    if arch == "ResNet-50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unknown architecture: {arch}")


@st.cache_resource
def load_vision_model(arch: str, model_path: str):
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    num_classes = infer_num_classes(state)
    if num_classes is None:
        raise ValueError("Could not infer number of classes from weights.")

    class_names, source = load_class_names(num_classes)
    model = build_model(arch, num_classes)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, class_names, source


IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _find_column(fieldnames, candidates):
    if not fieldnames:
        return None
    for name in fieldnames:
        if not name:
            continue
        if name.strip().lower() in candidates:
            return name
    return None


def _load_labels_from_csv(csv_path: Path):
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        label_col = _find_column(reader.fieldnames, {"label", "label (class)", "class"})
        if not label_col:
            return []
        labels = set()
        for row in reader:
            raw = row.get(label_col)
            if raw:
                labels.add(raw.strip())
        return sorted(labels)


def _labels_from_config(model):
    id2label = getattr(model.config, "id2label", None)
    if not isinstance(id2label, dict) or not id2label:
        return []
    try:
        pairs = sorted(((int(k), v) for k, v in id2label.items()), key=lambda x: x[0])
        return [label for _idx, label in pairs]
    except Exception:
        return []


def _labels_from_file(path: Path):
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, list):
        return [item for item in raw if isinstance(item, str) and item]
    if isinstance(raw, dict):
        labels = raw.get("labels")
        if isinstance(labels, list):
            return [item for item in labels if isinstance(item, str) and item]
    return []


@st.cache_resource
def load_text_model():
    if not TEXT_MODEL_DIR.exists():
        return None, None, [], "Rule-based (BERT model not found)"
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        return None, None, [], f"Rule-based (transformers unavailable: {exc})"

    try:
        model = AutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_DIR)
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_DIR)
    except Exception as exc:
        return None, None, [], f"Rule-based (BERT load failed: {exc})"

    model.eval()
    labels = _labels_from_config(model)
    if not labels:
        labels = _labels_from_file(TEXT_LABELS_PATH)
    if not labels:
        labels = _load_labels_from_csv(DATA_DIR / "symptom.csv")
    return model, tokenizer, labels, "BERT (models/bert_symptom_classifier)"


def predict_text_bert(text: str, model, tokenizer, labels):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probs = torch.softmax(logits, dim=1).squeeze(0)
    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())
    label = None
    if labels and pred_id < len(labels):
        label = labels[pred_id]
    if not label:
        id2label = getattr(model.config, "id2label", {})
        label = id2label.get(pred_id, "Unknown")
    return label, confidence


def predict_image(model, class_names, image: Image.Image, top_k: int = 3):
    image = image.convert("RGB")
    tensor = IMAGE_TRANSFORM(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_k = min(top_k, probs.shape[0])
    scores, indices = torch.topk(probs, k=top_k)
    results = []
    for score, idx in zip(scores.tolist(), indices.tolist()):
        results.append((class_names[idx], float(score)))
    return results


def _normalize_label(label: str):
    return label.replace("___", "_").replace("__", "_").replace("_", " ").strip()


def _clean_label_text(text: str):
    return text.replace("_", " ").strip()


def _split_label(label: str):
    if "___" in label:
        crop, disease = label.split("___", 1)
        return crop, disease
    if "_" in label:
        crop, disease = label.split("_", 1)
        return crop, disease
    return label, ""


def pretty_label(label: str):
    crop, disease = _split_label(label)
    crop = _clean_label_text(crop)
    disease = _clean_label_text(disease)
    if disease:
        return f"{crop} - {disease}"
    return crop


def confidence_band(score: float):
    if score >= 0.75:
        return "High"
    if score >= 0.5:
        return "Medium"
    if score >= 0.25:
        return "Low"
    return "Very low"


def confidence_bar(score: float, width: int = 10):
    score = max(0.0, min(1.0, score))
    filled = int(round(score * width))
    filled = min(width, max(0, filled))
    return "█" * filled + "░" * (width - filled)


@st.cache_resource
def load_label_map():
    path = DATA_DIR / "label_map.json"
    if path.exists():
        data = _safe_read_json(path)
        if isinstance(data, dict):
            labels = data.get("labels")
            if isinstance(labels, dict):
                return labels
    return {}


def build_summary(label: str, tips):
    label_l = label.lower()
    if "healthy" in label_l:
        return "No visible disease symptoms detected in the input."
    if isinstance(tips, dict):
        notes = tips.get("notes")
        if isinstance(notes, list) and notes:
            return notes[0]
    return "Symptoms appear consistent with the predicted disease."


def get_label_info(label: str, tips, label_map: dict):
    info = label_map.get(label, {}) if isinstance(label_map, dict) else {}
    crop_raw, disease_raw = _split_label(label)
    crop = _clean_label_text(crop_raw)
    disease = _clean_label_text(disease_raw)
    if not disease and "healthy" in label.lower():
        disease = "Healthy"
    display_name = info.get("display_name") or pretty_label(label)
    if display_name.lower().endswith(" - healthy"):
        display_name = display_name[:-9] + "Healthy"
    if disease.lower() == "healthy":
        disease = "Healthy"
    summary = info.get("summary") or build_summary(label, tips)
    return {
        "display_name": display_name,
        "crop": info.get("crop") or crop,
        "disease": info.get("disease") or disease or "Unknown",
        "summary": summary,
    }


def follow_up_questions(score: float, input_type: str):
    if score >= 0.5:
        return []
    if input_type == "image":
        return [
            "Can you upload a clearer, closer photo of the affected leaf?",
            "Which crop is this plant?",
            "Do you see spots, mold, or yellowing on the leaves?",
        ]
    return [
        "Which crop is affected?",
        "Are there spots, mold, or yellowing on the leaves?",
        "Can you upload a leaf photo for a more accurate result?",
    ]


def _detect_crop(text: str, class_names):
    crops = sorted({name.split("_")[0].lower() for name in class_names})
    for crop in crops:
        if crop and crop in text:
            return crop
    return None


def rule_based_text_diagnosis(text: str, class_names):
    text_l = text.lower()
    crop = _detect_crop(text_l, class_names)

    candidates = class_names
    if crop:
        candidates = [c for c in class_names if c.lower().startswith(crop)]

    if "healthy" in text_l or "no issue" in text_l or "no problem" in text_l:
        healthy = [c for c in candidates if "healthy" in c.lower()]
        if healthy:
            return healthy[0], 0.75

    keyword_map = [
        ("mosaic", "mosaic"),
        ("yellow", "yellow"),
        ("curl", "curl"),
        ("virus", "virus"),
        ("mildew", "mildew"),
        ("powder", "powdery"),
        ("blight", "blight"),
        ("scab", "scab"),
        ("rust", "rust"),
        ("mold", "mold"),
        ("spot", "spot"),
        ("lesion", "spot"),
        ("mite", "mite"),
        ("bacterial", "bacterial"),
    ]

    for key, token in keyword_map:
        if key in text_l:
            matches = [c for c in candidates if token in c.lower()]
            if matches:
                return matches[0], 0.65

    if candidates:
        return candidates[0], 0.35
    return "Unknown", 0.0


@st.cache_resource
def load_treatment_map():
    path = DATA_DIR / "treatments.json"
    if path.exists():
        data = _safe_read_json(path)
        if isinstance(data, dict):
            return data
    return None


def recommend_treatment(label: str):
    label_l = label.lower()
    treatment_map = load_treatment_map()

    if treatment_map:
        by_label = treatment_map.get("by_label", {})
        if label in by_label:
            return by_label[label]
        if "healthy" in label_l:
            return treatment_map.get("healthy", []) or []
        return treatment_map.get("default", []) or []

    if "healthy" in label_l:
        return [
            "No treatment needed. Continue regular watering and monitoring.",
            "Keep leaves dry when possible and remove debris to prevent future disease.",
        ]

    rule_map = {
        "blight": [
            "Remove infected leaves and discard away from the field.",
            "Avoid overhead irrigation and improve air circulation.",
        ],
        "rust": [
            "Remove heavily infected leaves and prune for airflow.",
            "Consider a labeled fungicide if the infection spreads.",
        ],
        "mildew": [
            "Reduce humidity and improve ventilation around plants.",
            "Apply sulfur or a labeled fungicide if needed.",
        ],
        "scab": [
            "Remove fallen leaves and fruit to reduce inoculum.",
            "Use resistant varieties where possible.",
        ],
        "mold": [
            "Improve airflow and avoid wet foliage.",
            "Remove infected tissue and sanitize tools.",
        ],
        "spot": [
            "Remove infected leaves and avoid splashing water.",
            "Apply a labeled fungicide or bactericide as appropriate.",
        ],
        "virus": [
            "Remove infected plants to prevent spread.",
            "Control insect vectors and disinfect tools.",
        ],
        "mite": [
            "Wash leaves with water or use insecticidal soap.",
            "Introduce or protect beneficial predators if available.",
        ],
        "bacterial": [
            "Avoid overhead watering and disinfect tools.",
            "Remove infected leaves to slow spread.",
        ],
        "rot": [
            "Remove infected plant parts and improve drainage.",
            "Avoid overwatering and sanitize tools.",
        ],
    }

    for key, tips in rule_map.items():
        if key in label_l:
            return tips

    return [
        "Remove infected tissue and monitor closely.",
        "Consider a labeled treatment specific to the crop and disease.",
    ]


@st.cache_resource
def load_gemini_client():
    return get_gemini_client()


def format_prediction(label: str, score: float):
    pretty = pretty_label(label)
    band = confidence_band(score)
    return f"{pretty} (confidence {band}, {score * 100:.1f}%)"


def format_treatment_lines(tips):
    if isinstance(tips, dict):
        lines = []
        sections = [
            ("immediate", "Immediate actions"),
            ("prevention", "Prevention"),
            ("notes", "Notes"),
        ]
        for key, title in sections:
            items = tips.get(key) if isinstance(tips, dict) else None
            if items:
                lines.append(f"{title}:")
                for item in items:
                    lines.append(f"- {item}")
        return lines

    if isinstance(tips, list):
        return ["Treatment tips:"] + [f"- {tip}" for tip in tips]

    return ["Treatment tips:", "- No treatment data available."]


def build_report_text(report: dict):
    lines = []
    lines.append(f"Timestamp: {report.get('timestamp', 'unknown')}")
    lines.append(f"Input type: {report.get('input_type', 'unknown')}")
    if report.get("input_type") == "text":
        lines.append(f"Input text: {report.get('input_text', '')}")
    if report.get("model"):
        lines.append(f"Model: {report.get('model')}")
    if report.get("class_source"):
        lines.append(f"Class source: {report.get('class_source')}")

    pred = report.get("prediction", {})
    if pred:
        lines.append("Prediction:")
        label = pred.get("label", "Unknown")
        score = pred.get("confidence", 0.0)
        lines.append(format_prediction(label, score))
        band = report.get("confidence_band")
        if band:
            lines.append(f"Confidence band: {band}")
        label_info = report.get("label_info") or {}
        if label_info.get("crop"):
            lines.append(f"Crop: {label_info.get('crop')}")
        if label_info.get("disease") and label_info.get("disease") != "Unknown":
            lines.append(f"Issue: {label_info.get('disease')}")
        summary = report.get("summary")
        if summary:
            lines.append(f"Summary: {summary}")

    top_k = report.get("top_k", [])
    if top_k:
        lines.append("Other candidates:")
        for item in top_k[1:]:
            lines.append(format_prediction(item.get("label", "Unknown"), item.get("confidence", 0.0)))

    tips = report.get("treatment")
    if tips is not None:
        lines.extend(format_treatment_lines(tips))

    return "\n".join(lines)


st.set_page_config(
    page_title="AI PlantDocBot",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Page background ── */
.stApp {
    background: linear-gradient(135deg, #0d1f0f 0%, #0a1a0c 50%, #071209 100%);
    color: #e8f5e9;
}

/* ── Hero header ── */
.hero {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 40%, #388e3c 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    border: 1px solid #4caf5033;
    box-shadow: 0 8px 32px rgba(76,175,80,0.15);
}
.hero h1 {
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: #c8e6c9;
    font-size: 1.05rem;
    margin: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d2b10 0%, #0a1f0c 100%) !important;
    border-right: 1px solid #2e7d3244;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #c8e6c9 !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #1b3a1e !important;
    border: 1px solid #4caf5044 !important;
    color: #e8f5e9 !important;
    border-radius: 8px !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 12px !important;
    margin-bottom: 0.6rem !important;
    border: 1px solid #2e7d3233 !important;
}
[data-testid="stChatMessage"][data-testid*="user"] {
    background: #1a3a1d !important;
}
[data-testid="stChatMessage"][data-testid*="assistant"] {
    background: #0f2611 !important;
}

/* ── Diagnosis result card ── */
.diag-card {
    background: linear-gradient(135deg, #1b3a1e 0%, #1a3320 100%);
    border: 1px solid #4caf5055;
    border-left: 4px solid #4caf50;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    box-shadow: 0 4px 16px rgba(76,175,80,0.1);
}
.diag-card.healthy {
    border-left-color: #66bb6a;
    background: linear-gradient(135deg, #1b3a1e 0%, #1e3d20 100%);
}
.diag-card.disease {
    border-left-color: #ff7043;
    background: linear-gradient(135deg, #3a1b1b 0%, #2d1a1a 100%);
}

/* ── Confidence badge ── */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}
.badge-high   { background: #1b5e20; color: #a5d6a7; border: 1px solid #4caf50; }
.badge-medium { background: #f57f1733; color: #ffcc80; border: 1px solid #ff9800; }
.badge-low    { background: #b71c1c33; color: #ef9a9a; border: 1px solid #f44336; }

/* ── Progress bar for confidence ── */
.conf-bar-wrap {
    background: #1a2e1c;
    border-radius: 6px;
    height: 8px;
    width: 100%;
    margin: 6px 0 10px 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 8px;
    border-radius: 6px;
    transition: width 0.4s ease;
}
.conf-bar-high   { background: linear-gradient(90deg, #2e7d32, #66bb6a); }
.conf-bar-medium { background: linear-gradient(90deg, #e65100, #ffa726); }
.conf-bar-low    { background: linear-gradient(90deg, #b71c1c, #ef5350); }

/* ── Section labels ── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #81c784;
    margin-bottom: 4px;
}

/* ── Treatment card ── */
.treatment-card {
    background: #0f2611;
    border: 1px solid #2e7d3244;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 0.5rem;
}
.treatment-item {
    display: flex;
    align-items: flex-start;
    gap: 8px;
    padding: 4px 0;
    color: #c8e6c9;
    font-size: 0.92rem;
}
.treatment-icon { color: #4caf50; flex-shrink: 0; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #0f2611 !important;
    border: 2px dashed #4caf5055 !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #4caf50 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #2e7d32, #388e3c) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388e3c, #43a047) !important;
    box-shadow: 0 4px 12px rgba(76,175,80,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ── Chat input ── */
[data-testid="stChatInput"] {
    background: #0f2611 !important;
    border: 1px solid #4caf5055 !important;
    border-radius: 12px !important;
}
[data-testid="stChatInput"] textarea {
    color: #e8f5e9 !important;
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
    background: #1b3a1e !important;
    border: 1px solid #4caf5055 !important;
    color: #a5d6a7 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
}

/* ── Success / warning / info boxes ── */
.stSuccess { background: #1b3a1e !important; border-color: #4caf50 !important; }
.stWarning { background: #3a2a0a !important; border-color: #ff9800 !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a1a0c; }
::-webkit-scrollbar-thumb { background: #2e7d32; border-radius: 3px; }

/* ── Follow-up chips ── */
.chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 4px 0; }
.chip {
    background: #1b3a1e;
    border: 1px solid #4caf5055;
    border-radius: 20px;
    padding: 5px 14px;
    font-size: 0.83rem;
    color: #a5d6a7;
    cursor: pointer;
    transition: all 0.15s ease;
    white-space: nowrap;
}
.chip:hover { background: #2e7d32; border-color: #4caf50; color: #fff; }

/* ── Low confidence warning ── */
.conf-warning {
    background: #3a2a0a;
    border: 1px solid #ff980055;
    border-left: 4px solid #ff9800;
    border-radius: 8px;
    padding: 8px 14px;
    margin: 6px 0;
    color: #ffcc80;
    font-size: 0.88rem;
}

/* ── Session history pill ── */
.hist-pill {
    display: inline-block;
    background: #0f2611;
    border: 1px solid #2e7d3233;
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: #81c784;
    margin: 2px;
}
</style>
""", unsafe_allow_html=True)


def render_diagnosis_card(label_info: dict, score: float, source: str = ""):
    """Render a rich HTML diagnosis result card."""
    display = label_info.get("display_name", "Unknown")
    crop = label_info.get("crop", "")
    disease = label_info.get("disease", "")
    summary = label_info.get("summary", "")
    band = confidence_band(score)
    pct = score * 100

    is_healthy = "healthy" in disease.lower()
    card_class = "healthy" if is_healthy else "disease"
    disease_icon = "✅" if is_healthy else "🔴"

    badge_class = {"High": "badge-high", "Medium": "badge-medium"}.get(band, "badge-low")
    bar_class = {"High": "conf-bar-high", "Medium": "conf-bar-medium"}.get(band, "conf-bar-low")

    source_html = f'<div class="section-label" style="margin-top:8px;opacity:0.5">via {source}</div>' if source else ""

    # Confidence warning for low scores
    conf_warning_html = ""
    if score < 0.5:
        conf_warning_html = (
            '<div class="conf-warning">⚠️ Low confidence — the model isn\'t very certain. '
            'Consider uploading a clearer photo or describing symptoms in the chat.</div>'
        )

    st.markdown(f"""
    <div class="diag-card {card_class}">
        <div style="display:flex; align-items:center; gap:10px; margin-bottom:8px;">
            <span style="font-size:1.5rem">{disease_icon}</span>
            <span style="font-size:1.15rem; font-weight:700; color:#e8f5e9">{display}</span>
            <span class="badge {badge_class}">{band} · {pct:.1f}%</span>
        </div>
        <div class="conf-bar-wrap">
            <div class="conf-bar-fill {bar_class}" style="width:{pct:.1f}%"></div>
        </div>
        {conf_warning_html}
        <div style="display:flex; gap:24px; flex-wrap:wrap; margin-bottom:6px;">
            <div><div class="section-label">Crop</div>
                 <div style="color:#c8e6c9; font-size:0.95rem">🌱 {crop}</div></div>
            <div><div class="section-label">Issue</div>
                 <div style="color:#c8e6c9; font-size:0.95rem">🔬 {disease}</div></div>
        </div>
        <div class="section-label">Summary</div>
        <div style="color:#a5d6a7; font-size:0.92rem; font-style:italic">{summary}</div>
        {source_html}
    </div>
    """, unsafe_allow_html=True)


def render_followup_chips(label: str, score: float, msg_idx: int):
    """Render clickable follow-up question chips after a diagnosis."""
    chips = get_followup_chips(label, score)
    st.markdown('<div class="section-label" style="margin-top:10px">💬 Quick questions</div>', unsafe_allow_html=True)
    cols = st.columns(len(chips))
    for i, (col, chip) in enumerate(zip(cols, chips)):
        # Use a unique key per chip; clicking sets chip_prompt and triggers rerun
        btn_key = f"chip_{msg_idx}_{i}"
        if col.button(chip, key=btn_key, use_container_width=True):
            st.session_state["chip_prompt"] = chip
            st.rerun()


def render_treatment_card(tips):
    """Render treatment tips as a styled card."""
    if isinstance(tips, dict):
        sections = [
            ("immediate", "🚨 Immediate Actions"),
            ("prevention", "🛡️ Prevention"),
            ("notes", "📝 Notes"),
        ]
        html = '<div class="treatment-card">'
        for key, title in sections:
            items = tips.get(key)
            if items:
                html += f'<div class="section-label" style="margin-top:8px">{title}</div>'
                for item in items:
                    html += f'<div class="treatment-item"><span class="treatment-icon">•</span><span>{item}</span></div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
    elif isinstance(tips, list):
        html = '<div class="treatment-card"><div class="section-label">💊 Treatment Tips</div>'
        for item in tips:
            html += f'<div class="treatment-item"><span class="treatment-icon">•</span><span>{item}</span></div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)


def render_other_candidates(preds: list):
    """Render other top-K predictions as small pills."""
    if len(preds) <= 1:
        return
    st.markdown('<div class="section-label" style="margin-top:12px">Other Candidates</div>', unsafe_allow_html=True)
    cols = st.columns(len(preds) - 1)
    for col, (lbl, sc) in zip(cols, preds[1:]):
        band = confidence_band(sc)
        badge_class = {"High": "badge-high", "Medium": "badge-medium"}.get(band, "badge-low")
        col.markdown(
            f'<div style="background:#1a2e1c;border:1px solid #2e7d3244;border-radius:8px;padding:8px 10px;">'
            f'<div style="font-size:0.82rem;color:#c8e6c9;font-weight:600">{pretty_label(lbl)}</div>'
            f'<span class="badge {badge_class}" style="margin-top:4px;display:inline-block">{sc*100:.1f}%</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ── Hero header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🌿 AI PlantDocBot</h1>
    <p>Upload a leaf photo or describe symptoms to get an instant plant disease diagnosis.</p>
</div>
""", unsafe_allow_html=True)


available_models = {
    name: MODELS_DIR / filename
    for name, filename in VISION_MODEL_FILES.items()
    if (MODELS_DIR / filename).exists()
}

# Text model options
TEXT_MODEL_OPTIONS = {}
if TEXT_MODEL_DIR.exists():
    TEXT_MODEL_OPTIONS["BERT (local)"] = "bert"
TEXT_MODEL_OPTIONS["Rule-based (no model needed)"] = "rule_based"

# Lazy-load heavy models only once, show spinner on first load
with st.spinner("Loading models...") if "models_loaded" not in st.session_state else contextlib.nullcontext():
    text_model, text_tokenizer, text_labels, text_status = load_text_model()
    gemini_client = load_gemini_client()
    st.session_state["models_loaded"] = True

gemini_available = gemini_client is not None

with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.markdown("---")

    # Vision model selector
    st.markdown('<div class="section-label">Vision Model</div>', unsafe_allow_html=True)
    if not available_models:
        st.warning("No vision models found in models/.")
        selected_arch = None
        model_path = None
    else:
        selected_arch = st.selectbox("Vision model", list(available_models.keys()), label_visibility="collapsed")
        model_path = available_models[selected_arch]

    # Text model selector
    st.markdown('<div class="section-label" style="margin-top:12px">Text Model</div>', unsafe_allow_html=True)
    selected_text_model_label = st.selectbox(
        "Text model",
        list(TEXT_MODEL_OPTIONS.keys()),
        label_visibility="collapsed",
        help="BERT gives better accuracy. Rule-based works without any model files.",
    )
    selected_text_model_key = TEXT_MODEL_OPTIONS[selected_text_model_label]

    st.markdown('<div class="section-label" style="margin-top:12px">Top-K Predictions</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K predictions", 1, 5, 3, label_visibility="collapsed")

    st.markdown("---")

    # Model status
    bert_available = text_model is not None and text_tokenizer is not None
    if selected_text_model_key == "bert" and bert_available:
        st.success("🧠 BERT active")
    elif selected_text_model_key == "bert" and not bert_available:
        st.warning("⚠️ BERT not found — using rule-based")
    else:
        st.info("📋 Rule-based mode")

    # Gemini status
    if gemini_available:
        st.success("✨ AI Chat: Active")
    else:
        st.warning("💬 AI Chat: Add GEMINI_API_KEY to .env")

    st.markdown("---")

    if st.session_state.get("last_report"):
        st.markdown("### 📥 Download Report")
        report = st.session_state.last_report
        report_json = json.dumps(report, indent=2)
        col1, col2 = st.columns(2)
        col1.download_button("JSON", data=report_json, file_name="plantdocbot_report.json", mime="application/json")
        col2.download_button("TXT", data=build_report_text(report), file_name="plantdocbot_report.txt", mime="text/plain")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.diagnosis_history = []
        st.session_state.pop("last_report", None)
        st.session_state.pop("chip_prompt", None)
        st.rerun()

    # Session history
    if st.session_state.get("diagnosis_history"):
        st.markdown("---")
        st.markdown("### 🕓 Recent Diagnoses")
        for d in reversed(st.session_state.diagnosis_history[-3:]):
            li = d.get("label_info", {})
            disease = li.get("disease", "Unknown")
            crop = li.get("crop", "")
            ts = d.get("timestamp", "")[:16].replace("T", " ")
            st.markdown(
                f'<span class="hist-pill">🌱 {crop} — {disease} · {ts}</span>',
                unsafe_allow_html=True,
            )


if "messages" not in st.session_state:
    st.session_state.messages = []
if "diagnosis_history" not in st.session_state:
    st.session_state.diagnosis_history = []  # last 3 diagnoses
if "chip_prompt" not in st.session_state:
    st.session_state.chip_prompt = None


# Render chat history
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"], caption="Uploaded leaf image", width=300)
        elif msg.get("type") == "multi_image":
            cols = st.columns(min(len(msg["content"]), 3))
            for col, img in zip(cols, msg["content"]):
                col.image(img, use_container_width=True)
        elif msg.get("rich"):
            rich = msg["rich"]
            if msg.get("content"):
                st.markdown(msg["content"])
            render_diagnosis_card(rich["label_info"], rich["score"], rich.get("source", ""))
            render_treatment_card(rich["tips"])
            render_other_candidates(rich.get("preds", []))
            # Follow-up chips only on the last assistant message
            if idx == len(st.session_state.messages) - 1:
                render_followup_chips(
                    rich["label_info"].get("disease", ""),
                    rich["score"],
                    idx,
                )
        elif msg.get("type") == "multi_rich":
            if msg.get("content"):
                st.markdown(msg["content"])
            for i, rich in enumerate(msg.get("rich_list", []), 1):
                st.markdown(f"**Image {i}**", unsafe_allow_html=False)
                render_diagnosis_card(rich["label_info"], rich["score"], rich.get("source", ""))
                render_treatment_card(rich["tips"])
            # Follow-up chips on last message
            if idx == len(st.session_state.messages) - 1:
                rich_list = msg.get("rich_list", [])
                if rich_list:
                    render_followup_chips(
                        rich_list[0]["label_info"].get("disease", ""),
                        rich_list[0]["score"],
                        idx,
                    )
        else:
            st.markdown(msg["content"])


# --- Multi-image diagnosis section ---
image_files = st.file_uploader(
    "📸 Upload leaf photo(s) — up to 3",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)
if image_files and selected_arch:
    # Cap at 3
    image_files = image_files[:3]
    if st.button("🔍 Diagnose Image(s)", use_container_width=False):
        images = [Image.open(f) for f in image_files]

        # Show images in user bubble
        st.session_state.messages.append({
            "role": "user",
            "type": "multi_image" if len(images) > 1 else "image",
            "content": images if len(images) > 1 else images[0],
        })

        try:
            vision_model, class_names, source = load_vision_model(selected_arch, str(model_path))
            label_map = load_label_map()
            reports = []

            for img in images:
                preds = predict_image(vision_model, class_names, img, top_k=top_k)
                primary_label, primary_score = preds[0]
                tips = recommend_treatment(primary_label)
                label_info = get_label_info(primary_label, tips, label_map)
                reports.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "input_type": "image",
                    "model": selected_arch,
                    "class_source": source,
                    "prediction": {"label": primary_label, "confidence": primary_score},
                    "top_k": [{"label": l, "confidence": s} for l, s in preds],
                    "treatment": tips,
                    "summary": label_info.get("summary"),
                    "confidence_band": confidence_band(primary_score),
                    "label_info": label_info,
                })

            # Detect same vs different plants
            same_plant = _are_same_plant(reports) if len(reports) > 1 else True

            # Save last report as the first/primary one
            st.session_state.last_report = reports[0]
            # Add to diagnosis history (keep last 3)
            st.session_state.diagnosis_history.append(reports[0])
            st.session_state.diagnosis_history = st.session_state.diagnosis_history[-3:]

            with st.spinner("🤖 Analyzing your plant(s)..."):
                if len(reports) == 1:
                    friendly = enhance_diagnosis(
                        reports[0], gemini_client,
                        history=st.session_state.diagnosis_history[:-1],
                    )
                else:
                    friendly = enhance_multi_diagnosis(reports, gemini_client, same_plant)
                    if not same_plant:
                        friendly = "🌿 These look like **different plants** — here's what I found for each:\n\n" + friendly

            rich_list = [
                {
                    "label_info": r["label_info"],
                    "score": r["prediction"]["confidence"],
                    "source": selected_arch,
                    "tips": r["treatment"],
                    "preds": [(item["label"], item["confidence"]) for item in r.get("top_k", [])],
                }
                for r in reports
            ]

            if len(reports) == 1:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": friendly,
                    "rich": rich_list[0],
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "multi_rich",
                    "content": friendly,
                    "rich_list": rich_list,
                })

        except Exception as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"❌ Diagnosis failed: {exc}"}
            )
        st.rerun()


# --- Chat input (also handles chip button clicks) ---
chip_prompt = st.session_state.get("chip_prompt")
typed_prompt = st.chat_input("💬 Ask me anything about your plant, describe symptoms, or follow up...")
prompt = typed_prompt or chip_prompt

# Clear chip after consuming it
if chip_prompt and prompt == chip_prompt:
    st.session_state["chip_prompt"] = None

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    last_report = st.session_state.get("last_report")
    diag_history = st.session_state.get("diagnosis_history", [])
    has_prior_diagnosis = last_report is not None

    # Chip questions are always follow-ups.
    # For typed messages: follow-up if there's a prior diagnosis AND
    # the message doesn't look like a new symptom description.
    is_chip = prompt == chip_prompt
    is_followup = has_prior_diagnosis and (
        is_chip or (
            len(prompt.split()) < 25 and not any(
                kw in prompt.lower() for kw in [
                    "my plant", "leaf", "symptom", "spot", "yellow", "brown",
                    "blight", "rust", "mold", "wilt", "upload", "image", "photo",
                ]
            )
        )
    )

    if is_followup and gemini_available:
        history = streamlit_history_to_gemini(st.session_state.messages[:-1])
        response_text = chat_with_gemini(
            user_message=prompt,
            history=history,
            last_diagnosis=last_report,
            diagnosis_history=diag_history,
            client=gemini_client,
        )
        st.session_state.messages.append({"role": "assistant", "content": response_text})

    else:
        class_names = []
        if selected_arch and model_path and model_path.exists():
            try:
                _, class_names, _source = load_vision_model(selected_arch, str(model_path))
            except Exception:
                class_names = []
        if not class_names:
            class_names = ["Healthy"]

        use_bert = (
            selected_text_model_key == "bert"
            and text_model is not None
            and text_tokenizer is not None
        )
        label, score = (
            predict_text_bert(prompt, text_model, text_tokenizer, text_labels)
            if use_bert else
            rule_based_text_diagnosis(prompt, class_names)
        )
        text_notes = "BERT" if use_bert else "Rule-based"

        tips = recommend_treatment(label)
        label_map = load_label_map()
        label_info = get_label_info(label, tips, label_map)

        report = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "input_type": "text",
            "input_text": prompt,
            "text_model": "bert" if use_bert else "rule_based",
            "prediction": {"label": label, "confidence": score},
            "treatment": tips,
            "summary": label_info.get("summary"),
            "confidence_band": confidence_band(score),
            "label_info": label_info,
            "notes": text_notes,
        }
        st.session_state.last_report = report
        st.session_state.diagnosis_history.append(report)
        st.session_state.diagnosis_history = st.session_state.diagnosis_history[-3:]

        with st.spinner("🌿 Thinking..."):
            friendly_response = enhance_diagnosis(
                report, gemini_client, history=diag_history,
            )

        st.session_state.messages.append({
            "role": "assistant",
            "content": friendly_response,
            "rich": {
                "label_info": label_info,
                "score": score,
                "source": text_notes,
                "tips": tips,
                "preds": [],
            },
        })

    st.rerun()

