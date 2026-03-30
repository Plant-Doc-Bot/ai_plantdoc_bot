import json
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


MODEL_FILES = {
    "resnet50": "plant_disease_resnet50_v1.pth",
    "resnet18": "plant_disease_resnet18_v1.pth",
    "mobilenetv2": "plant_disease_mobilenet_v1.pth",
}
DEFAULT_MODEL_KEY = "resnet50"


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
    if arch == "mobilenetv2":
        model = models.mobilenet_v2(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = torch.nn.Linear(in_features, num_classes)
        return model
    if arch == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model
    if arch == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_classes)
        return model
    raise ValueError(f"Unknown architecture: {arch}")


@lru_cache(maxsize=8)
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
        results.append({"label": class_names[idx], "confidence": float(score)})
    return results


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


@lru_cache(maxsize=1)
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

    return []


def resolve_model_key(value: str | None):
    if not value:
        return DEFAULT_MODEL_KEY
    key = value.strip().lower()
    if key in MODEL_FILES:
        return key
    for candidate, filename in MODEL_FILES.items():
        if key == filename.lower():
            return candidate
    if "resnet50" in key:
        return "resnet50"
    if "resnet18" in key:
        return "resnet18"
    if "mobilenet" in key:
        return "mobilenetv2"
    return None


def get_model_bundle(model_key: str):
    filename = MODEL_FILES.get(model_key)
    if not filename:
        return None
    path = MODELS_DIR / filename
    if not path.exists():
        return None
    model, class_names, source = load_vision_model(model_key, path)
    return model, class_names, source, filename


app = Flask(__name__)


@app.get("/health")
def health():
    available = [name for name, file in MODEL_FILES.items() if (MODELS_DIR / file).exists()]
    return jsonify({"status": "ok", "models": available})


@app.get("/models")
def models_list():
    return health()


@app.post("/predict/image")
def predict_image_endpoint():
    if "image" not in request.files:
        return jsonify({"error": "Missing image file. Use form-data key 'image'."}), 400

    image_file = request.files["image"]
    image = Image.open(BytesIO(image_file.read()))
    model_key = resolve_model_key(request.form.get("model") or request.args.get("model"))
    top_k_raw = request.form.get("top_k") or request.args.get("top_k") or "3"
    try:
        top_k = max(1, int(top_k_raw))
    except ValueError:
        top_k = 3

    if not model_key:
        return jsonify({"error": "Unknown model key."}), 400

    bundle = get_model_bundle(model_key)
    if not bundle:
        return jsonify({"error": "Requested model not found."}), 404

    model, class_names, source, filename = bundle
    preds = predict_image(model, class_names, image, top_k=top_k)
    primary = preds[0] if preds else {"label": "Unknown", "confidence": 0.0}
    treatment = recommend_treatment(primary["label"])

    return jsonify(
        {
            "model": model_key,
            "weights": filename,
            "class_source": source,
            "prediction": primary,
            "top_k": preds,
            "treatment": treatment,
        }
    )


@app.post("/predict/text")
def predict_text_endpoint():
    payload = request.get_json(silent=True) or {}
    text = payload.get("text") or request.form.get("text")
    if not text:
        return jsonify({"error": "Missing 'text' in JSON body or form data."}), 400

    model_key = resolve_model_key(payload.get("model") or request.form.get("model") or request.args.get("model"))
    class_names = ["Healthy"]
    source = "fallback"
    if model_key:
        bundle = get_model_bundle(model_key)
        if bundle:
            _model, class_names, source, _filename = bundle

    label, score = rule_based_text_diagnosis(text, class_names)
    treatment = recommend_treatment(label)

    return jsonify(
        {
            "model": model_key or "fallback",
            "class_source": source,
            "prediction": {"label": label, "confidence": score},
            "treatment": treatment,
            "notes": "Rule-based text model (BERT integration pending).",
        }
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
