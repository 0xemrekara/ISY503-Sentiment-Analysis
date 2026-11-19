# Electronics Review Sentiment Analysis

ISY503 Assessment 3 | Group NLP project

**Project status:** Integration in progress. This README will be updated after the data preparation script, Streamlit interface, and end-to-end testing are complete.

## Overview

This project uses a convolutional neural network (CNN) to classify Amazon Electronics reviews as `Positive review` or `Negative review`. It uses the Electronics category of the [Johns Hopkins Multi-Domain Sentiment Dataset](http://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html).

## Repository structure

- `data/`: dataset description and, after integration, Ezra's data-preparation script. Review CSVs are excluded from this public copy pending redistribution permission review
- `notebooks/train_and_evaluate.ipynb`: CNN training and evaluation workflow (model contributor)
- `model/sentiment_model.keras`: selected saved model
- `model/predict.py`: reusable `predict_sentiment(text)` function for the interface
- `results/`: recorded outputs; final interpretation will be written after integration
- `docs/`: experiment notes, ethical considerations and presentation outline (drafts)
- `app.py`: Streamlit interface (to be added by the interface contributor)

## Setup

From the repository root, install dependencies:

```bash
python -m pip install -r requirements.txt
```

Test the provided model without retraining:

```bash
python -c "from model.predict import predict_sentiment; print(predict_sentiment('The sound quality is excellent.'))"
```

After placing the audited team CSVs under `data/`, run the training notebook from the **repository root**. For standalone Colab, upload the audited team dataset ZIP at the prompt. The notebook saves a new training run under `outputs/` rather than overwriting the supplied model.

Once `app.py` is integrated, run:

```bash
streamlit run app.py
```

## Dataset

CSV files use the fields `text` and `label`; `0` denotes negative and `1` positive. The team's audited split contains 1,572 training, 195 validation and 198 test reviews. **The CSV files are not committed to this public repository.** Acquire the dataset from the original source and use the team's approved preprocessing outputs. The final test set excludes one near-duplicate discovered during the data audit. See `data/README.md`; the original preparation script still needs to be integrated. To reproduce the current notebook, place the three team CSVs under `data/` locally, or upload the team data ZIP when prompted in Google Colab.

## Evaluation and limitations

**To be finalized after integration.** Model comparison notes and recorded figures are retained under `results/` and `docs/` to support transparent reporting. Raw misclassified review texts are kept out of the public repository pending dataset permissions review. The fixed test split was consulted in earlier development, so its observed score must not be presented as an entirely untouched estimate.

## Group contributions

- Data preparation: Ezra (script pending)
- CNN and evaluation: model contributor (included)
- Streamlit interface: interface contributor (`app.py` pending)

All contributions will be reviewed through Git branches and pull requests. Individual contribution reports are submitted separately and should not be published in this repository.

## Referencing and AI-use acknowledgement

**To be completed before submission.** The group will add appropriate APA references, ethical considerations and the required individual AI-use declarations in the relevant submission materials.

### Running the training notebook in Google Colab

A notebook opened on its own in Colab does not automatically include the
repository's `data/` folder. Run all cells and, at the upload prompt, choose
`03_TEAM_DATA_AUDITED_v1.zip` (or this complete project ZIP). The notebook
reads the same three audited CSV files without requiring the author's Drive
account. A standalone Colab run retrains a model; it does not alter the
selected application model already included in `model/`.


## Training notebook and result files

Run `notebooks/train_and_evaluate.ipynb` to train a new CNN on the prepared CSVs. The notebook shows dataset counts, layer summary, a per-epoch table, training and validation curves, classification metrics, a confusion matrix and sample misclassifications. It saves output under `outputs/` and, in standalone Colab, downloads `training_run_outputs.zip`.

The tracked `model/sentiment_model.keras` and files under `results/` are from the earlier selected v8 experiment (test accuracy 90.40%, macro F1 0.9039); **they are not produced by a new notebook run**. Because the held-out test set was inspected during earlier experiments, report its performance as exploratory.
