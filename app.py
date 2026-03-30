import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


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
    data_cleaned_dir = DATA_DIR / "data_cleaned"

    if json_path.exists():
        data = _safe_read_json(json_path)
        if isinstance(data, list) and len(data) == expected_count:
            return data, "data/class_names.json"

    if txt_path.exists():
        names = [line.strip() for line in txt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(names) == expected_count:
            return names, "data/class_names.txt"

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
def load_vision_model(arch: str, model_path: Path):
    state = torch.load(model_path, map_location="cpu")
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


def format_prediction(label: str, score: float):
    pretty = _normalize_label(label)
    return f"{pretty} (confidence {score * 100:.1f}%)"


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

    top_k = report.get("top_k", [])
    if top_k:
        lines.append("Other candidates:")
        for item in top_k[1:]:
            lines.append(format_prediction(item.get("label", "Unknown"), item.get("confidence", 0.0)))

    tips = report.get("treatment")
    if tips is not None:
        lines.extend(format_treatment_lines(tips))

    return "\n".join(lines)


st.set_page_config(page_title="AI PlantDocBot", layout="wide")
st.title("AI PlantDocBot")
st.write(
    "Upload a leaf image or describe symptoms. The chatbot will route your input "
    "to the vision model or text heuristic and return a diagnosis with tips."
)


available_models = {
    name: MODELS_DIR / filename
    for name, filename in VISION_MODEL_FILES.items()
    if (MODELS_DIR / filename).exists()
}

with st.sidebar:
    st.header("Settings")
    if not available_models:
        st.warning("No vision models found in models/.")
        selected_arch = None
        model_path = None
    else:
        selected_arch = st.selectbox("Vision model", list(available_models.keys()))
        model_path = available_models[selected_arch]
    top_k = st.slider("Top-K predictions", 1, 5, 3)
    st.caption("Text diagnosis uses a rule-based fallback for now.")
    if st.session_state.get("last_report"):
        st.subheader("Download report")
        report = st.session_state.last_report
        report_json = json.dumps(report, indent=2)
        st.download_button(
            "Download JSON",
            data=report_json,
            file_name="plantdocbot_report.json",
            mime="application/json",
        )
        st.download_button(
            "Download TXT",
            data=build_report_text(report),
            file_name="plantdocbot_report.txt",
            mime="text/plain",
        )


if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("type") == "image":
            st.image(msg["content"], caption="Uploaded image", use_container_width=True)
        else:
            st.write(msg["content"])


image_file = st.file_uploader("Upload a leaf image", type=["png", "jpg", "jpeg"])
if image_file and selected_arch:
    if st.button("Diagnose Image"):
        image = Image.open(image_file)
        st.session_state.messages.append({"role": "user", "type": "image", "content": image})

        try:
            model, class_names, source = load_vision_model(selected_arch, model_path)
            preds = predict_image(model, class_names, image, top_k=top_k)
            primary_label, primary_score = preds[0]
            tips = recommend_treatment(primary_label)

            response_lines = [
                f"Model: {selected_arch} (classes from {source})",
                "Prediction:",
                format_prediction(primary_label, primary_score),
            ]
            if len(preds) > 1:
                response_lines.append("Other candidates:")
                for label, score in preds[1:]:
                    response_lines.append(f"- {format_prediction(label, score)}")

            response_lines.extend(format_treatment_lines(tips))

            report = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "input_type": "image",
                "model": selected_arch,
                "class_source": source,
                "prediction": {"label": primary_label, "confidence": primary_score},
                "top_k": [{"label": label, "confidence": score} for label, score in preds],
                "treatment": tips,
            }
            st.session_state.last_report = report

            st.session_state.messages.append({"role": "assistant", "content": "\n".join(response_lines)})
        except Exception as exc:
            st.session_state.messages.append(
                {"role": "assistant", "content": f"Image diagnosis failed: {exc}"}
            )


prompt = st.chat_input("Describe symptoms or ask for help...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})

    class_names = []
    if selected_arch and model_path and model_path.exists():
        try:
            state = torch.load(model_path, map_location="cpu")
            num_classes = infer_num_classes(state) or 0
            class_names, _source = load_class_names(num_classes)
        except Exception:
            class_names = []

    if not class_names:
        class_names = ["Healthy"]

    label, score = rule_based_text_diagnosis(prompt, class_names)
    tips = recommend_treatment(label)

    response = [
        "Text diagnosis (rule-based):",
        format_prediction(label, score),
    ]
    response.extend(format_treatment_lines(tips))

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "input_type": "text",
        "input_text": prompt,
        "prediction": {"label": label, "confidence": score},
        "treatment": tips,
        "notes": "Rule-based text model (BERT integration pending).",
    }
    st.session_state.last_report = report

    st.session_state.messages.append({"role": "assistant", "content": "\n".join(response)})
