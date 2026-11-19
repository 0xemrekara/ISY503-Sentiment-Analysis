# Group collaboration

Use one repository and separate branches. Commit under your own GitHub account.

- `model` branch: CNN training notebook, saved model, prediction function and results.
- `data` branch: original dataset-preparation script and any data-specific documentation.
- `app` branch: Streamlit `app.py`, UI-specific changes, and optional screenshots after testing.

Create pull requests into `main` so the group can review changes before merging.
For integration fixes, make new commits and note the contributor in commit history; do not rewrite another member's history.

## Prediction interface

`from model.predict import predict_sentiment`

Call `predict_sentiment(text: str) -> str`. It returns exactly `Positive review` or `Negative review`. Empty input raises a `ValueError`; handle this in the UI. The model file must remain in `model/sentiment_model.keras`.

## Do not commit

- Private Drive credentials, authentication tokens or `.env` files.
- Personally identifying student IDs or individual contribution reports.
- Temporary notebook checkpoints, local caches or new training artifacts from `outputs/`.
