
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
try:
    data = pd.read_csv("news.csv")
    data = data.drop(columns=["Unnamed: 0"], errors="ignore")  # Drop unnecessary column
except FileNotFoundError:
    st.error("Error: news.csv file not found!")
    st.stop()

# Ensure correct columns
if "title" not in data.columns or "label" not in data.columns:
    st.error("Error: CSV file must contain 'title' and 'label' columns!")
    st.stop()

# Prepare data
data["label"] = data["label"].str.lower()  # Standardize labels
x = data["title"].astype(str).values
y = data["label"].astype(str).values

cv = CountVectorizer()
x = cv.fit_transform(x)

# Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train model
model = MultinomialNB()
model.fit(x_train, y_train)

# Evaluate model
accuracy = accuracy_score(y_test, model.predict(x_test))

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detection System")
st.markdown("### Enter a news headline to check if it's real or fake.")

# Sidebar Information
st.sidebar.image("logo.jfif", use_container_width=True)
st.sidebar.subheader("Model Accuracy")
st.sidebar.write(f"{accuracy * 100:.2f}%")
st.sidebar.markdown("### About")
st.sidebar.info("This tool uses a Naive Bayes model trained on a dataset of news headlines to predict whether a given headline is real or fake.")

# Input field with enhanced UI
user_input = st.text_area("✍️ Enter News Headline:", "")

# Prediction Button with styled output
if st.button("🔍 Predict"):
    if user_input.strip():
        input_data = cv.transform([user_input])
        prediction = model.predict(input_data)[0]
        
        # Display Prediction
        if prediction == "real":
            st.success("🟢 This news is classified as REAL.")
        else:
            st.error("🔴 This news is classified as FAKE.")
    else:
        st.warning("⚠️ Please enter a news headline.")

# Footer
st.markdown("---")
st.markdown("💡 *Developed using Streamlit and Scikit-learn* 🚀")
