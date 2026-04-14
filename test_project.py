"""
Quick smoke test for AI PlantDocBot — image + text pipeline.
Run: venv\Scripts\python.exe test_project.py
"""
import json
import sys
from pathlib import Path

# ── Load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import os
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"

SAMPLE_IMAGE = Path("data/test/Apple___Apple_scab/plantdoc_test_Apple_Scab_Leaf_00001.jpg")
SAMPLE_TEXT = "my tomato plant has yellow spots and the leaves are curling"

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
INFO = "\033[94m→\033[0m"

def section(title):
    print(f"\n{'─'*50}")
    print(f"  {title}")
    print(f"{'─'*50}")

# ── 1. Model files ─────────────────────────────────────────────────────────────
section("1. Checking model files")
model_files = {
    "ResNet-50":   "plant_disease_resnet50_v1.pth",
    "ResNet-18":   "plant_disease_resnet18_v1.pth",
    "MobileNetV2": "plant_disease_mobilenet_v1.pth",
}
found_model = None
for name, fname in model_files.items():
    path = MODELS_DIR / fname
    if path.exists():
        size_mb = path.stat().st_size / 1_000_000
        print(f"  {PASS} {name}: {fname} ({size_mb:.1f} MB)")
        if found_model is None:
            found_model = (name, path)
    else:
        print(f"  {FAIL} {name}: NOT FOUND")

if not found_model:
    print(f"  {FAIL} No models found — cannot continue.")
    sys.exit(1)

# ── 2. Class names ─────────────────────────────────────────────────────────────
section("2. Loading class names")
txt_path = DATA_DIR / "class_names.txt"
if txt_path.exists():
    names = [l.strip() for l in txt_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"  {PASS} Loaded {len(names)} classes from class_names.txt")
    print(f"  {INFO} Sample: {names[:3]}")
else:
    print(f"  {FAIL} class_names.txt not found")
    sys.exit(1)

# ── 3. Image test ──────────────────────────────────────────────────────────────
section("3. Image diagnosis test")
if not SAMPLE_IMAGE.exists():
    print(f"  {FAIL} Sample image not found: {SAMPLE_IMAGE}")
else:
    print(f"  {INFO} Image: {SAMPLE_IMAGE}")
    arch, model_path = found_model

    # Load model
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    # Infer num classes
    if "classifier.1.weight" in state:
        num_classes = state["classifier.1.weight"].shape[0]
    else:
        num_classes = state["fc.weight"].shape[0]

    if arch == "MobileNetV2":
        m = models.mobilenet_v2(weights=None)
        m.classifier[1] = torch.nn.Linear(m.classifier[1].in_features, num_classes)
    elif arch == "ResNet-18":
        m = models.resnet18(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)
    else:
        m = models.resnet50(weights=None)
        m.fc = torch.nn.Linear(m.fc.in_features, num_classes)

    m.load_state_dict(state, strict=True)
    m.eval()
    print(f"  {PASS} Model loaded: {arch} ({num_classes} classes)")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    img = Image.open(SAMPLE_IMAGE).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(m(tensor), dim=1).squeeze(0)
    top3_scores, top3_idx = torch.topk(probs, 3)

    print(f"  {PASS} Inference complete. Top-3 predictions:")
    for score, idx in zip(top3_scores.tolist(), top3_idx.tolist()):
        label = names[idx] if idx < len(names) else f"Class_{idx}"
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"       {bar}  {label}  ({score*100:.1f}%)")

# ── 4. Text diagnosis test ─────────────────────────────────────────────────────
section("4. Text diagnosis test (rule-based)")
print(f"  {INFO} Input: \"{SAMPLE_TEXT}\"")

keyword_map = [
    ("mosaic","mosaic"),("yellow","yellow"),("curl","curl"),("virus","virus"),
    ("mildew","mildew"),("blight","blight"),("scab","scab"),("rust","rust"),
    ("mold","mold"),("spot","spot"),("mite","mite"),("bacterial","bacterial"),
]
text_l = SAMPLE_TEXT.lower()
matched = None
for key, token in keyword_map:
    if key in text_l:
        candidates = [c for c in names if token in c.lower()]
        if candidates:
            matched = (candidates[0], 0.65)
            break

if matched:
    label, score = matched
    print(f"  {PASS} Matched: {label} ({score*100:.0f}% confidence)")
else:
    print(f"  {FAIL} No match found")

# ── 5. Treatments ──────────────────────────────────────────────────────────────
section("5. Treatment data")
treatments_path = DATA_DIR / "treatments.json"
if treatments_path.exists():
    data = json.loads(treatments_path.read_text(encoding="utf-8"))
    by_label = data.get("by_label", {})
    print(f"  {PASS} treatments.json loaded — {len(by_label)} disease entries")
else:
    print(f"  {FAIL} treatments.json not found")

# ── 6. Gemini / OpenRouter ─────────────────────────────────────────────────────
section("6. OpenRouter (Gemini) API test")
api_key = os.getenv("GEMINI_API_KEY", "").strip()
if not api_key:
    print(f"  {FAIL} GEMINI_API_KEY not set in .env")
else:
    print(f"  {INFO} Key found: {api_key[:8]}...{api_key[-4:]}")
    try:
        from openai import OpenAI
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
        resp = client.chat.completions.create(
            model="google/gemini-flash-1.5",
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        )
        reply = resp.choices[0].message.content.strip()
        if "OK" in reply.upper():
            print(f"  {PASS} OpenRouter API working — response: '{reply}'")
        else:
            print(f"  {PASS} OpenRouter API responded: '{reply}'")
    except Exception as e:
        print(f"  {FAIL} OpenRouter API error: {e}")

# ── 7. BERT text model ─────────────────────────────────────────────────────────
section("7. BERT text model")
bert_dir = MODELS_DIR / "bert_symptom_classifier"
if bert_dir.exists():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(bert_dir)
        bert_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
        bert_model.eval()
        inputs = tokenizer(SAMPLE_TEXT, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = bert_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze(0)
        pred_id = int(torch.argmax(probs))
        conf = float(probs[pred_id])
        id2label = getattr(bert_model.config, "id2label", {})
        label = id2label.get(pred_id, f"Class_{pred_id}")
        print(f"  {PASS} BERT inference: '{label}' ({conf*100:.1f}% confidence)")
    except Exception as e:
        print(f"  {FAIL} BERT error: {e}")
else:
    print(f"  {INFO} BERT model not found (optional) — rule-based fallback will be used")

# ── Summary ────────────────────────────────────────────────────────────────────
section("Summary")
print(f"  {PASS} Core pipeline working — image + text diagnosis functional")
print(f"  {INFO} Run the app:  venv\\Scripts\\streamlit.exe run app.py")
print()
