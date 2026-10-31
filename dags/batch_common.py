import hashlib
import json
import math
import os
import re
import tempfile
import logging
from datetime import date
from pathlib import Path

import numpy as np

FEATURES = ["age", "tenure_months", "monthly_spend", "support_tickets_30d"]
LIMITS = [(18, 100), (0, 240), (0, 10000), (0, 100)]
ARTIFACTS = Path(os.environ.get("BATCH_ARTIFACTS", "/opt/airflow/include/artifacts"))
logger = logging.getLogger(__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(value):
    content = canonical(value)
    sha = digest(content)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ARTIFACTS, delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(ARTIFACTS / f"{sha}.json")
    return {"sha256": sha}


def read(ref):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid artifact reference")
    content = (ARTIFACTS / f"{sha}.json").read_bytes()
    if digest(content) != sha:
        raise ValueError("Artifact checksum mismatch")
    return json.loads(content)


def business_date(explicit, interval_start):
    if explicit:
        return date.fromisoformat(explicit).isoformat()
    if interval_start is None:
        raise ValueError("business_date is required without a data interval")
    return interval_start.in_timezone("Europe/Moscow").to_date_string()


def validate_rows(rows, day):
    if not rows:
        raise ValueError("Empty list is prohibited")

    seen_ids = set()

    for idx, row in enumerate(rows):
        if not isinstance(row["customer_id"], str):
            raise ValueError("customer_id must be string")
        if len(row["customer_id"]) == 0:
            raise ValueError("customer_id must be a non-empty string")
        if row["customer_id"] in seen_ids:
            raise ValueError("Duplicate")
        if row["business_date"] != day:
            raise ValueError("Wrong day")
        seen_ids.add(row["customer_id"])
        for feature_id in range(len(FEATURES)):
            feature = FEATURES[feature_id]
            lim_min, lim_max = LIMITS[feature_id]
            value = row[feature]
            if not isinstance(value, (int, float)):
                raise ValueError("Wrong type")
            if isinstance(value, bool):
                raise ValueError("Wrong type")
            if not math.isfinite(value):
                raise ValueError("Wrong value")
            if not (lim_min <= value <= lim_max):
                raise ValueError("Out of range")
        
    return rows


def validate_model(model):
    if model["feature_names"] != FEATURES:
        raise ValueError("Feature order mismatch")
    for field in ("mean", "scale", "weights"):
        if len(model[field]) != len(FEATURES):
            raise ValueError("Model dimensions mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in model[field]):
            raise ValueError("Invalid model coefficient")
    if any(v <= 0 for v in model["scale"]) or type(model["bias"]) not in (int, float) or not math.isfinite(model["bias"]):
        raise ValueError("Invalid model scale or bias")


def predict(rows, model):
    validate_model(model)
    matrix = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float64)
    normalized = (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = normalized @ np.asarray(model["weights"]) + model["bias"]
    scores = np.round(1 / (1 + np.exp(-np.clip(logits, -700, 700))), 8)
    return [{"customer_id": row["customer_id"], "score": float(score), "label": int(score >= 0.5)}
            for row, score in zip(rows, scores)]


def infer_batches(rows, model, batch_size):
    if isinstance(batch_size, bool):
        raise ValueError("batch_size must be integer")
    if not isinstance(batch_size, int):
        raise ValueError("batch_size must be integer")
    if not (1 <= batch_size <= 1000):
        raise ValueError("batch_size out of range")

    predictions = []
    batch_no = 0

    for start in range(0, len(rows), batch_size):
        batch_no += 1
        batch = rows[start:start + batch_size]
        logger.info(f"Batch {batch_no}: size {len(batch)}")
        predictions.extend(predict(batch, model))

    return predictions


def validate_predictions(predictions, expected_ids):
    if len(predictions) != len(expected_ids):
        raise ValueError("Count mismatch")

    expected_set = set(expected_ids)
    seen_ids = set()

    for pred in predictions:
        customer_id = pred["customer_id"]
        if customer_id in seen_ids:
            raise ValueError("Duplicate")
        seen_ids.add(customer_id)

        score = pred["score"]
        if isinstance(score, bool):
            raise ValueError("Wrong score type")
        if not isinstance(score, (int, float)):
            raise ValueError("Wrong score type")
        if not math.isfinite(score):
            raise ValueError("Wrong score")
        if not (0 <= score <= 1):
            raise ValueError("Score out of range")

        label = pred["label"]
        if isinstance(label, bool):
            raise ValueError("Wrong label type")
        if not isinstance(label, int):
            raise ValueError("Wrong label type")
        if label not in (0, 1):
            raise ValueError("Wrong label value")
        if label != int(score >= 0.5):
            raise ValueError("Label mismatch")

    if seen_ids != expected_set:
        raise ValueError("ID set mismatch")


def replace_day(conn, day, model_version, predictions):
    if not predictions:
        raise ValueError("Empty predictions")

    lock_key = f"{day}:{model_version}"

    with conn.transaction():
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (lock_key,),
            )
            cur.execute(
                "DELETE FROM predictions "
                "WHERE business_date = %s AND model_version = %s",
                (day, model_version),
            )
            cur.executemany(
                "INSERT INTO predictions "
                "(business_date, customer_id, model_version, score, label) "
                "VALUES (%s, %s, %s, %s, %s)",
                [
                    (day, p["customer_id"], model_version, p["score"], p["label"])
                    for p in predictions
                ],
            )
