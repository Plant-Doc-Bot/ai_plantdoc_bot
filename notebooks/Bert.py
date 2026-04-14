# -*- coding: utf-8 -*-
"""
BERT text classifier for plant disease symptoms.
Trains a model, evaluates accuracy, and reports confidence metrics.
"""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments


class PlantDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = list(labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def load_data(csv_path="data/symptom.csv", split=True):
    df = pd.read_csv(csv_path)
    # Normalize column names (strip spaces, handle common variants)
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    if "text" not in df.columns:
        for c in df.columns:
            if c.strip().lower() == "text":
                col_map[c] = "text"
                break
    if "label" not in df.columns:
        for c in df.columns:
            if c.strip().lower() in {"label", "label (class)", "class"}:
                col_map[c] = "label"
                break
    if col_map:
        df = df.rename(columns=col_map)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected columns 'text' and 'label', got: {df.columns.tolist()}")

    le = LabelEncoder()
    df["label"] = le.fit_transform(df["label"])

    if split:
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df["text"],
            df["label"],
            test_size=0.2,
            stratify=df["label"],
            random_state=42
        )
        return train_texts, test_texts, train_labels, test_labels, le

    return df["text"], None, df["label"], None, le


def encode_texts(tokenizer, texts, max_length=128):
    return tokenizer(
        list(texts),
        truncation=True,
        padding=True,
        max_length=max_length
    )


def softmax(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    probs = softmax(logits)
    avg_conf = float(np.mean(np.max(probs, axis=1)))
    return {
        "accuracy": acc,
        "avg_confidence": avg_conf
    }


def build_trainer(train_dataset, test_dataset, num_labels, label_list=None):
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=num_labels
    )
    if label_list:
        id2label = {i: label for i, label in enumerate(label_list)}
        label2id = {label: i for i, label in enumerate(label_list)}
        model.config.id2label = id2label
        model.config.label2id = label2id

    # Handle transformers version differences gracefully
    init_vars = TrainingArguments.__init__.__code__.co_varnames
    args = {
        "output_dir": "./results",
        "num_train_epochs": 5,
        "logging_dir": "./logs",
    }
    if "per_device_train_batch_size" in init_vars:
        args["per_device_train_batch_size"] = 8
    elif "train_batch_size" in init_vars:
        args["train_batch_size"] = 8
    if "per_device_eval_batch_size" in init_vars:
        args["per_device_eval_batch_size"] = 8
    elif "eval_batch_size" in init_vars:
        args["eval_batch_size"] = 8
    if "save_strategy" in init_vars:
        args["save_strategy"] = "epoch"
    if "evaluation_strategy" in init_vars:
        args["evaluation_strategy"] = "epoch"
    elif "eval_strategy" in init_vars:
        args["eval_strategy"] = "epoch"
    elif "evaluate_during_training" in init_vars:
        args["evaluate_during_training"] = True
    if "do_eval" in init_vars:
        args["do_eval"] = True
    if "report_to" in init_vars:
        args["report_to"] = []

    training_args = TrainingArguments(**args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )
    return trainer


def test_accuracy_and_confidence(trainer, test_dataset):
    metrics = trainer.evaluate(test_dataset)
    print("Accuracy:", metrics.get("eval_accuracy"))
    print("Avg Confidence:", metrics.get("eval_avg_confidence"))
    return metrics


def predict_with_confidence(model, tokenizer, le, text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.detach().cpu().numpy()

    probs = softmax(logits)
    pred_id = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs, axis=1)[0])
    label = le.inverse_transform([pred_id])[0]
    return label, confidence


def chatbot(model, tokenizer, le):
    while True:
        text = input("Enter symptoms (or 'exit'): ")
        if text == "exit":
            break
        label, conf = predict_with_confidence(model, tokenizer, le, text)
        print(f"Predicted Disease: {label} (confidence: {conf:.4f})")


def _normalize_columns(df):
    df.columns = [c.strip() for c in df.columns]
    col_map = {}
    if "text" not in df.columns:
        for c in df.columns:
            if c.strip().lower() == "text":
                col_map[c] = "text"
                break
    if "label" not in df.columns:
        for c in df.columns:
            if c.strip().lower() in {"label", "label (class)", "class"}:
                col_map[c] = "label"
                break
    if col_map:
        df = df.rename(columns=col_map)
    return df


def load_labeled_dataset(csv_path, tokenizer, le):
    df = pd.read_csv(csv_path)
    df = _normalize_columns(df)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Expected columns 'text' and 'label', got: {df.columns.tolist()}")

    try:
        labels = le.transform(df["label"])
    except ValueError as e:
        raise ValueError(
            "Test CSV contains labels not seen in training. "
            "Make sure label names match the training labels exactly."
        ) from e

    encodings = encode_texts(tokenizer, df["text"])
    dataset = PlantDataset(encodings, labels)
    return dataset


def evaluate_external_csv(trainer, tokenizer, le, csv_path):
    dataset = load_labeled_dataset(csv_path, tokenizer, le)
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    labels = predictions.label_ids

    preds = np.argmax(logits, axis=1)
    acc = accuracy_score(labels, preds)
    probs = softmax(logits)
    avg_conf = float(np.mean(np.max(probs, axis=1)))

    top3 = np.argsort(logits, axis=1)[:, -3:]
    top3_acc = float(np.mean([labels[i] in top3[i] for i in range(len(labels))]))

    print(f"External Test Accuracy: {acc}")
    print(f"External Test Avg Confidence: {avg_conf}")
    print(f"External Test Top-3 Accuracy: {top3_acc}")
    return {"accuracy": acc, "avg_confidence": avg_conf, "top3_accuracy": top3_acc}


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate BERT for plant disease symptoms.")
    parser.add_argument("--train_csv", default="data/symptom.csv", help="Path to training CSV.")
    parser.add_argument("--test_csv", default="", help="Optional labeled test CSV (unseen data).")
    parser.add_argument(
        "--save_dir",
        default="",
        help="Optional output directory to save the trained model + tokenizer + labels.",
    )
    args = parser.parse_args()

    use_external_test = bool(args.test_csv)
    train_texts, test_texts, train_labels, test_labels, le = load_data(
        args.train_csv,
        split=not use_external_test
    )

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_encodings = encode_texts(tokenizer, train_texts)

    train_dataset = PlantDataset(train_encodings, train_labels)
    if use_external_test:
        test_dataset = load_labeled_dataset(args.test_csv, tokenizer, le)
    else:
        test_encodings = encode_texts(tokenizer, test_texts)
        test_dataset = PlantDataset(test_encodings, test_labels)

    label_list = list(le.classes_)
    trainer = build_trainer(
        train_dataset,
        test_dataset,
        num_labels=len(label_list),
        label_list=label_list
    )
    trainer.train()

    # Test for accuracy and confidence on the test set
    test_accuracy_and_confidence(trainer, test_dataset)

    if args.test_csv:
        evaluate_external_csv(trainer, tokenizer, le, args.test_csv)

    # Example predictions with confidence
    label, conf = predict_with_confidence(trainer.model, tokenizer, le, "yellow spots on leaves")
    print(f"Example: {label} (confidence: {conf:.4f})")
    label, conf = predict_with_confidence(trainer.model, tokenizer, le, "white powder on leaves")
    print(f"Example: {label} (confidence: {conf:.4f})")

    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(save_dir)
        tokenizer.save_pretrained(save_dir)
        labels_path = save_dir / "labels.json"
        labels_payload = {
            "labels": label_list,
            "id2label": {str(i): label for i, label in enumerate(label_list)},
            "label2id": {label: i for i, label in enumerate(label_list)},
        }
        labels_path.write_text(json.dumps(labels_payload, indent=2), encoding="utf-8")
        print(f"Saved model + labels to {save_dir}")

    chatbot(trainer.model, tokenizer, le)


if __name__ == "__main__":
    main()
