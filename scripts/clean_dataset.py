import shutil
from pathlib import Path
import re

# 🔧 Paths
SOURCE_DIR = Path("data")          # handles train/val/test automatically
DEST_DIR = Path("data_cleaned")

# 🛑 Safety check
if DEST_DIR.exists():
    raise Exception("❌ data_cleaned already exists. Delete it first.")

DEST_DIR.mkdir(parents=True, exist_ok=True)

# 🧠 Label mapping (merge duplicates)
LABEL_MAP = {
    # Tomato
    "tomato___bacterial_spot": "Tomato_Bacterial_Spot",
    "tomato_leaf_bacterial_spot": "Tomato_Bacterial_Spot",

    "tomato___late_blight": "Tomato_Late_Blight",
    "tomato_leaf_late_blight": "Tomato_Late_Blight",

    "tomato___early_blight": "Tomato_Early_Blight",
    "tomato_early_blight_leaf": "Tomato_Early_Blight",

    "tomato___leaf_mold": "Tomato_Leaf_Mold",
    "tomato_mold_leaf": "Tomato_Leaf_Mold",

    "tomato___septoria_leaf_spot": "Tomato_Septoria",
    "tomato_septoria_leaf_spot": "Tomato_Septoria",

    # Grape
    "grape___black_rot": "Grape_Black_Rot",
    "grape_leaf_black_rot": "Grape_Black_Rot",

    # Apple
    "apple___apple_scab": "Apple_Scab",
    "apple_scab_leaf": "Apple_Scab",

    # Potato
    "potato___early_blight": "Potato_Early_Blight",
    "potato_leaf_early_blight": "Potato_Early_Blight",

    "potato___late_blight": "Potato_Late_Blight",
    "potato_leaf_late_blight": "Potato_Late_Blight",
}

# 🔧 Normalize function
def normalize_label(label):
    label = label.lower()
    label = label.replace("___", "_")
    label = label.replace(",", "")
    label = re.sub(r"\s+", "_", label)
    label = re.sub(r"[^a-z0-9_]", "", label)

    return LABEL_MAP.get(label, label.title())


# 📦 Processing
total = 0

for img_path in SOURCE_DIR.rglob("*"):
    if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    # class name = parent folder
    class_name = img_path.parent.name
    clean_label = normalize_label(class_name)

    target_dir = DEST_DIR / clean_label
    target_dir.mkdir(parents=True, exist_ok=True)

    new_name = f"{img_path.stem}_{total}{img_path.suffix}"
    shutil.copy(img_path, target_dir / new_name)

    total += 1

# 📊 Done
print(f"\n✅ Done! {total} images processed.")

# 📈 Optional: class summary
class_counts = {}
for p in DEST_DIR.iterdir():
    if p.is_dir():
        class_counts[p.name] = len(list(p.glob("*")))

print(f"\n📁 Total classes: {len(class_counts)}")
for k, v in sorted(class_counts.items()):
    print(f"{k}: {v}")