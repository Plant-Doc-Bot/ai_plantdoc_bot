import pandas as pd
import random
from pathlib import Path

# Classes (your final list is correct)
classes = [
    'Apple_Scab', 'Apple_Black_Rot', 'Apple_Cedar_Apple_Rust',
    'Apple_Healthy',

    'Bell_Pepper_Leaf_Spot', 'Bell_Pepper_Healthy',

    'Blueberry_Healthy',

    'Cherry_Healthy', 'Cherry_Powdery_Mildew',

    'Corn_Gray_Leaf_Spot', 'Corn_Leaf_Blight',
    'Corn_Cercospora_Leaf_Spot', 'Corn_Common_Rust',
    'Corn_Healthy', 'Corn_Northern_Leaf_Blight',

    'Grape_Black_Rot', 'Grape_Esca', 'Grape_Healthy',
    'Grape_Leaf_Blight',

    'Orange_Citrus_Greening',

    'Peach_Bacterial_Spot', 'Peach_Healthy',

    'Potato_Early_Blight', 'Potato_Healthy', 'Potato_Late_Blight',

    'Raspberry_Healthy',

    'Soybean_Healthy',

    'Squash_Powdery_Mildew',

    'Strawberry_Healthy', 'Strawberry_Leaf_Scorch',

    'Tomato_Bacterial_Spot', 'Tomato_Early_Blight', 'Tomato_Healthy',
    'Tomato_Late_Blight', 'Tomato_Leaf_Mold',
    'Tomato_Mosaic_Virus', 'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_Septoria', 'Tomato_Spider_Mites', 'Tomato_Target_Spot'
]

# Better symptom diversity
def get_symptom(cls):
    name = cls.lower()

    mapping = {
        "scab": ["rough patches", "scabby spots", "dark crust"],
        "rust": ["orange dust", "rusty spots", "reddish marks"],
        "blight": ["burnt leaves", "dry brown areas", "leaf dying"],
        "virus": ["leaf curling", "yellowing", "mosaic pattern"],
        "mildew": ["white powder", "fungal dust", "powder layer"],
        "spot": ["black spots", "tiny lesions", "dark marks"],
        "mite": ["tiny webs", "leaf speckles", "insect damage"],
        "healthy": ["looks fine", "no issue", "green healthy leaves"],
    }

    for key in mapping:
        if key in name:
            return mapping[key]

    return ["weird patches", "color change", "leaf damage"]


# Real-world noisy phrases
prefix_noise = [
    "i think", "not sure", "maybe", "looks like", "anyone help",
    "pls tell", "can someone check", "",
]

suffix_noise = [
    "pls help", "what to do", "any solution", "urgent",
    "need advice", "", "",
]

fillers = [
    "on my plant", "on leaf", "in farm", "in garden",
    "recently noticed", "",
]

# More diverse templates
templates = [
    "{prefix} {symptom} {filler}, is this {name}",
    "seeing {symptom} {filler}, maybe {name}",
    "{symptom} observed {filler}, could be {name}",
    "{name}? because {symptom}",
    "plant showing {symptom}, looks like {name}",
    "{symptom} + damage, is it {name}",
    "{prefix} {name} or something else, {symptom}",
    "{symptom} happening {filler}, not sure if {name}",
]

# Typo + spacing noise
def add_noise(text):
    # random typo
    if text and random.random() < 0.3:
        idx = random.randrange(len(text))
        text = text[:idx] + text[idx] + text[idx:]

    # random lowercase/uppercase mix
    if random.random() < 0.2:
        text = text.upper()

    # random extra spaces
    if random.random() < 0.2:
        text = text.replace(" ", "  ")

    return text


data = []

# Slight imbalance (real-world effect)
for cls in classes:
    name = cls.replace("_", " ").lower()
    symptoms = get_symptom(cls)

    samples_per_class = random.randint(60, 90)

    for _ in range(samples_per_class):
        text = random.choice(templates).format(
            prefix=random.choice(prefix_noise),
            name=name,
            symptom=random.choice(symptoms),
            filler=random.choice(fillers),
        )

        text += " " + random.choice(suffix_noise)
        text = add_noise(text)

        data.append([text.strip(), cls])

# Create DataFrame
df = pd.DataFrame(data, columns=["text", "class_name"])

# Shuffle (important)
df = df.sample(frac=1).reset_index(drop=True)

# Save
output_path = Path("final_clean_dataset.csv")
df.to_csv(output_path, index=False, encoding="utf-8")

print("FINAL realistic dataset created!")
