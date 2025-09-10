import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load your trained model and vectorizer
model = pickle.load(open("news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Preprocessing function
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

# Streamlit UI
st.title("📰 News Article Classifier")
st.write("Paste any news article below to classify it into categories like **business, politics, sports, tech, entertainment**.")

user_input = st.text_area("Enter your article text here:")

if st.button("Classify"):
    if user_input.strip() != "":
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        st.success(f"Predicted Category: **{prediction}**")
    else:
        st.warning("Please enter some text before classifying.")
