from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import sys
from contextlib import nullcontext
from collections import Counter
from pathlib import Path

try:
    import mlflow
except ImportError:
    mlflow = None


COMMON_TEXT_COLUMNS = ("text", "message", "sms", "v2")
COMMON_LABEL_COLUMNS = ("label", "target", "class", "category", "v1")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "spam-classifier")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import NaiveBayesSpamModel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a spam classifier from a CSV dataset using pure Python."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/spam.csv"),
        help="Path to the CSV dataset.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path("models/spam_model.pkl"),
        help="Path to save the trained model artifact.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("models/metrics.json"),
        help="Path to save evaluation metrics as JSON.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of the dataset reserved for evaluation.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for deterministic splits.",
    )
    return parser


def load_dataset(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    if csv_path.stat().st_size == 0:
        raise ValueError(f"Dataset is empty: {csv_path}")

    last_error: Exception | None = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with csv_path.open("r", encoding=encoding, newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                if not reader.fieldnames:
                    raise ValueError("CSV file is missing a header row.")
                return list(reader)
        except Exception as exc:
            last_error = exc

    raise ValueError(f"Unable to read dataset: {csv_path}") from last_error


def detect_columns(rows: list[dict[str, str]]) -> tuple[str, str]:
    first_row = rows[0]
    normalized = {column.lower().strip(): column for column in first_row.keys()}

    label_column = next(
        (normalized[name] for name in COMMON_LABEL_COLUMNS if name in normalized),
        None,
    )
    text_column = next(
        (normalized[name] for name in COMMON_TEXT_COLUMNS if name in normalized),
        None,
    )

    if label_column and text_column:
        return label_column, text_column

    columns = list(first_row.keys())
    if len(columns) >= 2:
        return columns[0], columns[1]

    raise ValueError(
        "Could not detect label and text columns. "
        "Expected columns like label/text, class/message, or v1/v2."
    )


def normalize_label(value: str) -> str:
    normalized = str(value).strip().lower()
    mapping = {
        "spam": "spam",
        "1": "spam",
        "true": "spam",
        "yes": "spam",
        "ham": "ham",
        "0": "ham",
        "false": "ham",
        "no": "ham",
    }

    if normalized not in mapping:
        raise ValueError(f"Unsupported label value found in dataset: {value}")

    return mapping[normalized]


def prepare_dataset(rows: list[dict[str, str]]) -> tuple[list[str], list[str]]:
    if not rows:
        raise ValueError("Dataset contains no rows.")

    label_column, text_column = detect_columns(rows)
    cleaned_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for row in rows:
        raw_label = row.get(label_column, "")
        raw_text = row.get(text_column, "")

        if raw_label is None or raw_text is None:
            continue

        text = str(raw_text).strip()
        if not text:
            continue

        label = normalize_label(str(raw_label))
        pair = (label, text)
        if pair in seen_pairs:
            continue

        seen_pairs.add(pair)
        cleaned_pairs.append(pair)

    if not cleaned_pairs:
        raise ValueError("Dataset has no usable rows after cleaning.")

    labels = [label for label, _ in cleaned_pairs]
    if len(set(labels)) < 2:
        raise ValueError("Training requires at least two label classes.")

    class_counts = Counter(labels)
    if min(class_counts.values()) < 2:
        raise ValueError(
            "Each class must contain at least 2 rows to create a train/test split."
        )

    texts = [text for _, text in cleaned_pairs]
    return texts, labels


def split_dataset(
    texts: list[str],
    labels: list[str],
    test_size: float,
    random_state: int,
) -> tuple[list[str], list[str], list[str], list[str]]:
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    paired_by_label: dict[str, list[tuple[str, str]]] = {"spam": [], "ham": []}
    for text, label in zip(texts, labels):
        paired_by_label[label].append((text, label))

    rng = random.Random(random_state)
    train_pairs: list[tuple[str, str]] = []
    test_pairs: list[tuple[str, str]] = []

    for label_pairs in paired_by_label.values():
        rng.shuffle(label_pairs)
        test_count = max(1, round(len(label_pairs) * test_size))
        test_count = min(test_count, len(label_pairs) - 1)
        test_pairs.extend(label_pairs[:test_count])
        train_pairs.extend(label_pairs[test_count:])

    rng.shuffle(train_pairs)
    rng.shuffle(test_pairs)

    x_train = [text for text, _ in train_pairs]
    y_train = [label for _, label in train_pairs]
    x_test = [text for text, _ in test_pairs]
    y_test = [label for _, label in test_pairs]
    return x_train, x_test, y_train, y_test


def evaluate_model(
    model: NaiveBayesSpamModel,
    x_test: list[str],
    y_test: list[str],
) -> dict[str, float]:
    predictions = [model.predict(text) for text in x_test]

    true_positive = sum(
        predicted == "spam" and actual == "spam"
        for predicted, actual in zip(predictions, y_test)
    )
    true_negative = sum(
        predicted == "ham" and actual == "ham"
        for predicted, actual in zip(predictions, y_test)
    )
    false_positive = sum(
        predicted == "spam" and actual == "ham"
        for predicted, actual in zip(predictions, y_test)
    )
    false_negative = sum(
        predicted == "ham" and actual == "spam"
        for predicted, actual in zip(predictions, y_test)
    )

    total = len(y_test)
    accuracy = (true_positive + true_negative) / total if total else 0.0
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1_score = 2 * precision * recall / f1_denominator if f1_denominator else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1_score, 4),
    }


def save_outputs(
    model: NaiveBayesSpamModel,
    model_out: Path,
    metrics: dict[str, float],
    metrics_out: Path,
) -> None:
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    with model_out.open("wb") as model_file:
        pickle.dump(model, model_file)

    with metrics_out.open("w", encoding="utf-8") as metrics_file:
        json.dump(metrics, metrics_file, indent=2)


def configure_mlflow() -> bool:
    if mlflow is None:
        return False

    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    return True


def main() -> None:
    args = build_parser().parse_args()

    rows = load_dataset(args.data)
    texts, labels = prepare_dataset(rows)
    x_train, x_test, y_train, y_test = split_dataset(
        texts,
        labels,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    mlflow_enabled = configure_mlflow()
    run_context = mlflow.start_run() if mlflow_enabled and mlflow is not None else nullcontext()

    with run_context:
        model = NaiveBayesSpamModel()
        model.fit(x_train, y_train)

        metrics = evaluate_model(model, x_test, y_test)
        save_outputs(model, args.model_out, metrics, args.metrics_out)

        if mlflow_enabled and mlflow is not None:
            mlflow.log_params(
                {
                    "model_type": "naive_bayes",
                    "data_path": str(args.data),
                    "model_output_path": str(args.model_out),
                    "metrics_output_path": str(args.metrics_out),
                    "test_size": args.test_size,
                    "random_state": args.random_state,
                    "samples": len(texts),
                    "train_samples": len(x_train),
                    "test_samples": len(x_test),
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(args.model_out))
            mlflow.log_artifact(str(args.metrics_out))

    print("Training complete.")
    print(f"Samples: {len(texts)}")
    print(f"Model saved to: {args.model_out}")
    print(f"Metrics saved to: {args.metrics_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
