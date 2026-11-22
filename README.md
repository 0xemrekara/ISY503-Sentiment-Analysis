# Data Preparation – Electronics Sentiment Reviews

## Dataset source
Johns Hopkins Multi-Domain Sentiment Dataset (Blitzer et al., 2007)  
http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html  

Only the **Electronics** category labelled files are used:  
- `sorted_data_acl/electronics/positive.review`  
- `sorted_data_acl/electronics/negative.review`  

## Required Python libraries
- `beautifulsoup4`  
- `lxml`  

Install with:  
```bash
pip install beautifulsoup4 lxml
```

## How to obtain the input dataset
```bash
mkdir -p /tmp/amazon_reviews
cd /tmp/amazon_reviews
curl -L -o domain_sentiment_data.tar.gz \
  "http://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz"
tar -xzf domain_sentiment_data.tar.gz
```
The script expects the extracted folder at `/tmp/amazon_reviews/sorted_data_acl`.  
Edit `DATA_ROOT` at the top of `prepare_data.py` if you place the data elsewhere.

## How to run the script
```bash
cd data
python prepare_data.py
```
The script writes `train.csv`, `val.csv`, `test.csv` and `data_preparation_summary.txt` into the same `data/` folder.

## Main cleaning and filtering steps
1. Parse the pseudo-XML files and extract title + review body.  
2. Light cleaning only:  
   - normalise whitespace  
   - convert to lowercase  
   - remove non-essential special characters  
   - **preserve** sentiment-bearing words (`not`, `never`, `bad`, `no`, etc.)  
3. Remove empty texts, exact duplicate texts, and reviews shorter than 5 words.  
4. Remove clear label/rating mismatches (positive file with rating ≤ 2, or negative file with rating ≥ 4). None were found in the Electronics data.  
5. Keep the original sentiment label (1 = positive, 0 = negative).

## Train / validation / test splitting method
- Fixed random seed: **42**  
- Stratified 80 % / 10 % / 10 % split by label so positive/negative proportions are maintained in every set.  
- Exact duplicates are removed before splitting, so no review appears in more than one set.

## Expected output files and structure
Three CSV files, each with exactly two columns:

| Column | Description |
|--------|-------------|
| `text` | Cleaned review text |
| `label` | 0 = negative, 1 = positive |

| File | Approximate rows |
|------|------------------|
| `train.csv` | ~1 572 |
| `val.csv` | ~195 |
| `test.csv` | ~199 |

A short numeric summary is also written to `data_preparation_summary.txt`.
