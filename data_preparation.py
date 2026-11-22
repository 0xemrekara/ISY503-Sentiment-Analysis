#!/usr/bin/env python3
"""
ISY503 Assessment 3 - Role A: Data Preparation (Electronics only)
Johns Hopkins Multi-Domain Sentiment Dataset - Electronics category

Loads positive and negative electronics reviews, cleans carefully
(preserving sentiment words), removes duplicates / short / invalid reviews,
and creates an 80/10/10 train/val/test split with fixed seed.

Output: train.csv, val.csv, test.csv
Columns: text, label (0 = negative, 1 = positive)
"""

import os
import re
import random
import csv
from pathlib import Path
from collections import Counter

try:
    from bs4 import BeautifulSoup
except ImportError:
    raise SystemExit("Please install beautifulsoup4 and lxml: pip install beautifulsoup4 lxml")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = Path("/tmp/amazon_reviews/sorted_data_acl")
DOMAIN = "electronics"
OUTPUT_DIR = Path("/workspace/artifacts")
SEED = 42
MIN_WORDS = 5          # extremely short threshold (kept low to preserve short but valid reviews)
TRAIN_RATIO = 0.80
VAL_RATIO = 0.10
TEST_RATIO = 0.10

# ---------------------------------------------------------------------------
# Cleaning (preserve sentiment-bearing words)
# ---------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """
    Light cleaning that preserves meaning and negation words.
    - Normalise whitespace
    - Decode common HTML entities already handled by BeautifulSoup
    - Keep letters, numbers, apostrophes, and basic punctuation that can affect sentiment
    - Do NOT remove words such as 'not', 'never', 'bad', 'no', etc.
    """
    if not text:
        return ""
    # Collapse newlines and multiple spaces
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    # Keep letters, numbers, spaces, apostrophes, and a few useful punctuation marks
    # (exclamation, question, period, comma) — remove other special characters
    text = re.sub(r"[^a-zA-Z0-9\s'.,!?]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Lowercase for consistency (modelling can re-case if needed)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def word_count(text: str) -> int:
    return len([w for w in text.split() if w])


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def parse_review_file(filepath: Path, label: int) -> list:
    """Parse a .review file and return list of dicts with text, label, rating."""
    reviews = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return reviews

    soup = BeautifulSoup(content, "lxml")
    for rev in soup.find_all("review"):
        rating_tag = rev.find("rating")
        text_tag = rev.find("review_text")
        title_tag = rev.find("title")

        rating_str = rating_tag.get_text(strip=True) if rating_tag else ""
        raw_text = text_tag.get_text(separator=" ", strip=True) if text_tag else ""
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Combine title + body so short titles still contribute useful text
        full_raw = (title + " " + raw_text).strip() if title else raw_text
        cleaned = clean_text(full_raw)

        if not cleaned:
            continue

        try:
            rating = float(rating_str)
        except (ValueError, TypeError):
            rating = None

        reviews.append({
            "text": cleaned,
            "label": label,
            "rating": rating,
        })
    return reviews


def load_electronics_reviews() -> list:
    domain_path = DATA_ROOT / DOMAIN
    if not domain_path.exists():
        raise FileNotFoundError(f"Electronics data not found at {domain_path}")

    all_reviews = []
    pos_file = domain_path / "positive.review"
    neg_file = domain_path / "negative.review"

    if pos_file.exists():
        pos = parse_review_file(pos_file, label=1)
        print(f"Loaded {len(pos)} positive electronics reviews")
        all_reviews.extend(pos)
    if neg_file.exists():
        neg = parse_review_file(neg_file, label=0)
        print(f"Loaded {len(neg)} negative electronics reviews")
        all_reviews.extend(neg)

    return all_reviews


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------
def filter_reviews(reviews: list) -> tuple:
    """
    Remove empty, duplicate, extremely short, and clearly invalid / mismatched reviews.
    Returns (filtered_list, stats_dict)
    """
    stats = {
        "raw": len(reviews),
        "duplicates": 0,
        "too_short": 0,
        "empty_after_clean": 0,
        "label_mismatch": 0,
    }

    # Exact-duplicate removal (by cleaned text)
    seen = set()
    unique = []
    for r in reviews:
        if not r["text"]:
            stats["empty_after_clean"] += 1
            continue
        if r["text"] in seen:
            stats["duplicates"] += 1
            continue
        seen.add(r["text"])
        unique.append(r)

    # Length filter
    length_ok = []
    for r in unique:
        if word_count(r["text"]) < MIN_WORDS:
            stats["too_short"] += 1
            continue
        length_ok.append(r)

    # Clear label / rating mismatch filter
    # Positive file should not contain rating <= 2; negative should not contain >= 4
    consistent = []
    for r in length_ok:
        rating = r["rating"]
        label = r["label"]
        if rating is None:
            consistent.append(r)
            continue
        if (label == 1 and rating <= 2.0) or (label == 0 and rating >= 4.0):
            stats["label_mismatch"] += 1
            continue
        consistent.append(r)

    stats["retained"] = len(consistent)
    return consistent, stats


# ---------------------------------------------------------------------------
# Stratified-style split that preserves proportions and avoids cross-set duplicates
# ---------------------------------------------------------------------------
def stratified_split(reviews: list):
    """
    Shuffle with fixed seed, then split 80/10/10 while keeping
    approximate positive/negative balance in each set.
    Because we already removed exact duplicates, no review can appear in more than one set.
    """
    random.seed(SEED)
    random.shuffle(reviews)

    pos = [r for r in reviews if r["label"] == 1]
    neg = [r for r in reviews if r["label"] == 0]

    def split_class(data):
        n = len(data)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        # remainder goes to test
        train = data[:n_train]
        val = data[n_train:n_train + n_val]
        test = data[n_train + n_val:]
        return train, val, test

    pos_train, pos_val, pos_test = split_class(pos)
    neg_train, neg_val, neg_test = split_class(neg)

    train = pos_train + neg_train
    val = pos_val + neg_val
    test = pos_test + neg_test

    # Final shuffle within each split so the order is mixed
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    return train, val, test


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def save_csv(data: list, filepath: Path):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        for r in data:
            writer.writerow({"text": r["text"], "label": r["label"]})
    print(f"Saved {len(data):>5} rows → {filepath.name}")


def write_summary(stats: dict, train, val, test, path: Path):
    def dist(data):
        c = Counter(r["label"] for r in data)
        total = len(data)
        return f"pos={c.get(1,0)} ({c.get(1,0)/total*100:.1f}%), neg={c.get(0,0)} ({c.get(0,0)/total*100:.1f}%)"

    lines = [
        "ISY503 Assessment 3 – Data Preparation Summary",
        "Electronics category only (Johns Hopkins Multi-Domain Sentiment Dataset)",
        "=" * 60,
        "",
        f"Raw reviews loaded:          {stats['raw']}",
        f"Removed – duplicates:        {stats['duplicates']}",
        f"Removed – empty after clean: {stats['empty_after_clean']}",
        f"Removed – too short (<{MIN_WORDS} words): {stats['too_short']}",
        f"Removed – label/rating mismatch: {stats['label_mismatch']}",
        f"Retained after filtering:    {stats['retained']}",
        "",
        f"Split (seed={SEED}, 80/10/10, stratified by label):",
        f"  Train: {len(train):>5}  |  {dist(train)}",
        f"  Val:   {len(val):>5}  |  {dist(val)}",
        f"  Test:  {len(test):>5}  |  {dist(test)}",
        "",
        "Cleaning notes:",
        "- Light normalisation only (whitespace, lowercasing, removal of non-essential special characters).",
        "- Sentiment-bearing words (not, never, bad, no, etc.) were deliberately preserved.",
        "- Title text was concatenated with the review body.",
        "- Exact duplicate texts were removed so no review appears in more than one split.",
        "",
        "Label quality notes:",
        f"- Clear mismatches (positive file with rating ≤2 or negative with rating ≥4) removed: {stats['label_mismatch']}",
        "- Original positive/negative files in this dataset are generally consistent with star ratings (4–5 vs 1–2).",
        "",
        "Ethical / bias considerations for the report:",
        "- Dataset is from ~2007; language, product types, and reviewer demographics reflect that period.",
        "- Electronics reviews may over-represent certain product categories (e.g. cameras, audio, computers).",
        "- Star-rating → binary conversion discards nuance (3-star reviews are absent from the labelled files).",
        "- Possible selection bias in which reviews Amazon surfaced or which the original authors sampled.",
        "",
        "Output files (columns: text, label):",
        "- train.csv, val.csv, test.csv",
        "  label 0 = Negative, 1 = Positive",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSummary written to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=== ISY503 Data Preparation: Electronics only ===\n")

    if not (DATA_ROOT / DOMAIN).exists():
        print(f"ERROR: Electronics data not found at {DATA_ROOT / DOMAIN}")
        print("Download and extract domain_sentiment_data.tar.gz first.")
        return

    print("Loading electronics reviews...")
    raw = load_electronics_reviews()

    print("\nFiltering...")
    filtered, stats = filter_reviews(raw)

    print("\nSplitting (80/10/10, seed=42, stratified)...")
    train, val, test = stratified_split(filtered)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_csv(train, OUTPUT_DIR / "train.csv")
    save_csv(val, OUTPUT_DIR / "val.csv")
    save_csv(test, OUTPUT_DIR / "test.csv")

    write_summary(stats, train, val, test, OUTPUT_DIR / "data_preparation_summary.txt")

    print("\nDone. Datasets are ready for neural-network development.")


if __name__ == "__main__":
    main()
