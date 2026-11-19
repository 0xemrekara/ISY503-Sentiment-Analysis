"""Predict sentiment using the trained Electronics review classifier."""

from functools import lru_cache
from pathlib import Path

import tensorflow as tf

MODEL_FILE = Path(__file__).resolve().parent / "sentiment_model.keras"


@lru_cache(maxsize=1)
def load_model():
    return tf.keras.models.load_model(MODEL_FILE, compile=False)


def predict_sentiment(text: str) -> str:
    """Return the sentiment label for one product review."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a product review.")

    probability = float(
        load_model().predict(tf.constant([text]), verbose=0)[0][0]
    )
    return "Positive review" if probability >= 0.5 else "Negative review"
