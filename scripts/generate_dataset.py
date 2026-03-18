import pandas as pd
import random

df = pd.read_csv("symptoms_aligned.csv")

print("Columns in dataset:", df.columns)

templates = [
    "The plant shows {} on the leaves.",
    "Symptoms observed include {}.",
    "Farmers report {} affecting the crop.",
    "The leaves develop {} during infection.",
    "Visible signs include {} on plant foliage.",
    "The plant exhibits {} across the leaves.",
    "Infected plants display {} symptoms.",
    "Crop leaves show {} due to disease."
]

symptom_variations = [
    "dark brown lesions",
    "yellow mosaic patterns",
    "black fungal spots",
    "irregular necrotic patches",
    "yellow chlorotic rings",
    "water soaked lesions",
    "leaf curling and deformation",
    "circular black spots",
    "brown necrotic areas",
    "powdery fungal growth",
    "leaf blight symptoms",
    "yellowing between veins"
]

rows = []

for _, row in df.iterrows():

    disease = row["class_name"]
    base_text = row["symptom_text_clean"]

    for i in range(90):

        template = random.choice(templates)
        symptom = random.choice(symptom_variations)

        sentence = template.format(symptom) + " " + base_text

        rows.append({
            "class_name": disease,
            "text": sentence
        })

aug_df = pd.DataFrame(rows)

print("Generated rows:", len(aug_df))

aug_df.to_csv("plant_disease_5000_dataset.csv", index=False)

print("Dataset saved as plant_disease_5000_dataset.csv")