"""Streamlit interface for the ISY503 Electronics review sentiment classifier."""

import streamlit as st
from model.predict import predict_sentiment

st.set_page_config(
    page_title="Electronics Review Sentiment Analysis",
    page_icon="💬",
    layout="centered",
)

st.title("Electronics Review Sentiment Analysis")
st.write(
    "Enter an electronics product review below. Our trained CNN will classify "
    "its sentiment as positive or negative."
)

with st.form("sentiment_form", clear_on_submit=False):
    review = st.text_area(
        "Product review",
        placeholder="e.g. The battery life is excellent and the sound is clear.",
        height=150,
        max_chars=10000,
    )
    submitted = st.form_submit_button("Analyse Sentiment", type="primary")

if submitted:
    if not review.strip():
        st.warning("Please enter a review before analysing its sentiment.")
    else:
        try:
            with st.spinner("Analysing review..."):
                prediction = predict_sentiment(review)
        except Exception:
            st.error("The sentiment model could not complete the prediction. Please check the application logs.")
            st.stop()

        if prediction == "Positive review":
            st.success(f"Prediction: {prediction}")
        elif prediction == "Negative review":
            st.error(f"Prediction: {prediction}")
        else:
            st.warning(f"Prediction: {prediction}")

st.divider()
st.caption(
    "ISY503 group project · CNN trained on the Johns Hopkins Multi-Domain "
    "Sentiment Dataset (Electronics). Predictions may be incorrect, especially "
    "for ambiguous or out-of-domain reviews."
)
