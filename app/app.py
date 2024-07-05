import streamlit as st
import re
import string
from nltk.tokenize import WordPunctTokenizer
from nltk.data import LazyLoader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download necessary NLTK data
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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

# Streamlit app
st.title('Text Preprocessing App')

# File uploader
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    try:
        # Read file
        text = uploaded_file.read().decode("utf-8")

        # Preprocess text
        cleaned_text = preprocess_text(text)

        # Display cleaned text
        st.subheader('Cleaned Text')
        st.write(cleaned_text)
    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a text file to preprocess.")
