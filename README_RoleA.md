# ISY503 Assessment 3 – Role A: Data Preparation
## Electronics category only (Johns Hopkins Multi-Domain Sentiment Dataset)

### Dataset
- Source: http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html
- File used: `domain_sentiment_data.tar.gz` → `sorted_data_acl/electronics/`
- Only the labelled files: `positive.review` and `negative.review` (1 000 each)

### What the script does
1. Parses the pseudo-XML review files with BeautifulSoup.
2. Concatenates title + review body.
3. Light cleaning only:
   - whitespace normalisation
   - lowercasing
   - removal of non-essential special characters
   - **preserves** sentiment words (`not`, `never`, `bad`, `no`, etc.)
4. Removes exact duplicate texts, empty reviews, and reviews shorter than 5 words.
5. Removes clear label/rating mismatches (none were found).
6. Stratified 80 / 10 / 10 split with fixed seed **42** so positive/negative proportions are maintained and no review appears in more than one set.

### Output files (exactly two columns: `text`, `label`)
| File       | Rows | Positive | Negative |
|------------|------|----------|----------|
| train.csv  | 1572 | 789 (50.2 %) | 783 (49.8 %) |
| val.csv    | 195  | 98  (50.3 %) | 97  (49.7 %) |
| test.csv   | 199  | 100 (50.3 %) | 99  (49.7 %) |

- `label` = 0 → Negative  
- `label` = 1 → Positive  

### How to reproduce
```bash
# 1. Download & extract the dataset (only needed once)
mkdir -p /tmp/amazon_reviews
cd /tmp/amazon_reviews
curl -L -o domain_sentiment_data.tar.gz \
  "http://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz"
tar -xzf domain_sentiment_data.tar.gz

# 2. Run the script (edit DATA_ROOT inside the script if you extracted elsewhere)
python data_preparation.py
```

Dependencies: `beautifulsoup4`, `lxml`  
(`pip install beautifulsoup4 lxml`)

### Quality / labelling notes
- 34 exact duplicates removed; 0 short or empty reviews; 0 clear label/rating mismatches.
- Original positive/negative files are consistent with star ratings (mostly 4–5 vs 1–2).
- No 3-star reviews exist in the labelled files (they were excluded by the dataset authors).

### Ethical / bias points for the report
- Dataset is from ~2007; language and product mix reflect that period.
- Electronics domain may over-represent certain product types (cameras, audio, computers, etc.).
- Binary conversion from star ratings discards the middle (3-star) nuance.
- Possible selection bias in which reviews were sampled or surfaced by Amazon.

### Files to commit to your branch
- `data_preparation.py`
- `train.csv`, `val.csv`, `test.csv`
- `data_preparation_summary.txt`
- `README_RoleA.md` (this file)
