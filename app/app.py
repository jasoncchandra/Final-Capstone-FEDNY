import streamlit as st
import re
import os
import string
import joblib
from nltk.tokenize import WordPunctTokenizer
from nltk.data import LazyLoader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Check if the files exist in the current directory
st.write("Current directory:", os.getcwd())
st.write("Files in current directory:", os.listdir(os.getcwd()))

# Load the vectorizer and LDA model
try:
    vectorizer = joblib.load('vectorizer.pkl')
    st.success("Vectorizer loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Vectorizer file not found: {e}")

try:
    lda_model = joblib.load('lda_model.pkl')
    st.success("LDA model loaded successfully.")
except FileNotFoundError as e:
    st.error(f"LDA model file not found: {e}")

# Define the preprocessing function
def preprocess_text(text):
    # Initialize NLP tools
    word_tokenizer = WordPunctTokenizer()
    sent_tokenizer = LazyLoader("tokenizers/punkt/english.pickle")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - {'.'}
    punctuation = set(string.punctuation) - {'.'}

    # Tokenize sentences
    sentences = sent_tokenizer.tokenize(text)
    cleaned_sentences = []

    for sentence in sentences:
        # Tokenize words
        words = word_tokenizer.tokenize(sentence)
        
        # Remove stop words and punctuation (except full stops)
        words = [word for word in words if word.lower() not in stop_words and word not in punctuation]
        
        # Lemmatization
        words = [lemmatizer.lemmatize(word) for word in words]
        
        cleaned_sentences.append(" ".join(words))

    return " ".join(cleaned_sentences)

def infer_topics(text):
    preprocessed_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([preprocessed_text])
    topic_distribution = lda_model.transform(text_vectorized)
    return topic_distribution

# Streamlit app
st.title('Text Preprocessing and Topic Modeling App')

# File uploader
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    try:
        # Read file
        text = uploaded_file.read().decode("utf-8")

        # Preprocess text
        cleaned_text = preprocess_text(text)

        # Infer topics
        topic_distribution = infer_topics(text)

        # Display cleaned text
        st.subheader('Cleaned Text')
        st.write(cleaned_text)

        # Display topic distribution
        st.subheader('Topic Distribution')

        # Assuming you have a mapping of topics for the 3-topic model
        topic_dict = {
            0: 'Economic Indicators',
            1: 'Financial Institutions',
            2: 'Monetary Policy'
        }

        for idx, score in enumerate(topic_distribution[0]):
            st.write(f"Topic {idx} ({topic_dict.get(idx, 'Unknown')}): {score:.4f}")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a text file to preprocess.")
