import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Load the trained model, vectorizer, and label encoder
try:
    model = joblib.load('logistic_regression_sms_spam_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    le = joblib.load('label_encoder.pkl')
except FileNotFoundError:
    st.error("Model, vectorizer, or label encoder files not found.")
    st.stop()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_sms(sms_text):
    # Convert to lowercase
    sms_text = sms_text.lower()

    # Remove special characters and numbers
    cleaned_text = re.sub(r'[^a-z\s]', '', sms_text)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # Tokenization
    tokens = nltk.word_tokenize(cleaned_text)

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]

    return ' '.join(tokens)

def predict_sms(sms_text, model, vectorizer, label_encoder):
    processed_sms = preprocess_sms(sms_text)

    if not processed_sms:
        return "Invalid input"

    vectorized_sms = vectorizer.transform([processed_sms])
    prediction = model.predict(vectorized_sms)

    return label_encoder.inverse_transform(prediction)[0]

# Streamlit App
st.title("📩 SMS Spam Detection")

st.write("Enter an SMS message below to check if it is Spam or Ham.")

sms_input = st.text_area("Enter SMS Message:", height=150)

if st.button("Predict"):
    if sms_input.strip():
        prediction = predict_sms(sms_input, model, tfidf_vectorizer, le)

        if prediction == 'spam':
            st.error("🚨 Prediction: SPAM")
        else:
            st.success("✅ Prediction: HAM")
    else:
        st.warning("Please enter an SMS message.")

