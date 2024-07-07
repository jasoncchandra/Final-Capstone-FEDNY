import streamlit as st
import re
import os
import string
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
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

def plot_topic_distribution(topic_distribution, topic_dict):
    # Bar plot for topic distribution
    topics = list(topic_dict.values())
    scores = topic_distribution[0]

    fig, ax = plt.subplots()
    sns.barplot(x=scores, y=topics, ax=ax)
    ax.set_title('Topic Distribution')
    ax.set_xlabel('Score')
    ax.set_ylabel('Topics')

    st.pyplot(fig)

def plot_word_clouds(lda_model, vectorizer, topic_dict):
    from wordcloud import WordCloud

    feature_names = vectorizer.get_feature_names_out()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    for idx, topic in enumerate(topic_dict.keys()):
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=10,
            contour_color='steelblue'
        ).generate_from_frequencies(
            {feature_names[i]: lda_model.components_[topic, i] for i in lda_model.components_[topic].argsort()[:-11:-1]}
        )
        axes[idx].imshow(wordcloud, interpolation='bilinear')
        axes[idx].set_title(topic_dict[topic])
        axes[idx].axis('off')

    st.pyplot(fig)

def plot_coherence_scores(coherence_scores):
    num_topics = list(coherence_scores.keys())
    scores = list(coherence_scores.values())

    fig, ax = plt.subplots()
    ax.plot(num_topics, scores, marker='o')
    ax.set_title('Coherence Scores by Number of Topics')
    ax.set_xlabel('Number of Topics')
    ax.set_ylabel('Coherence Score')
    ax.grid(True)

    st.pyplot(fig)

# Function to plot VIX data
def plot_vix_data():
    current_dir = os.path.dirname(__file__)
    vix_file_path = os.path.join(current_dir, 'data', 'VIX_History.csv')

    vix_data = pd.read_csv(vix_file_path, parse_dates=['DATE'])
    vix_data.set_index('DATE', inplace=True)

    fig = px.line(vix_data, x=vix_data.index, y='CLOSE', title='Historical VIX Data')
    st.plotly_chart(fig)

# Streamlit app
st.title('VIX Predictor via Text')

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select a Page:', ['Introduction', 'Training Process', 'Topic Inference', 'VIX Historical Data'])

# Introduction Page
if page == 'Introduction':
    st.header('Welcome to the VIX Predictor via Text')
    st.write('This application uses Latent Dirichlet Allocation (LDA) to perform topic modeling on textual data. '
             'It helps in understanding the themes present in the text and how they relate to different economic indicators, '
             'financial institutions, and monetary policies.')

# Training Process Page
elif page == 'Training Process':
    st.header('Training Process')
    st.write('We use Latent Dirichlet Allocation (LDA) to identify topics in the text data. Here are the coherence scores for different numbers of topics:')

    coherence_scores = {
        3: 0.09004151822108149,
        4: 0.0046645116832365555,
        5: -0.0045957082288738925,
        6: 0.0017441943397127069,
        7: 0.008529514229556334,
        8: 0.01629442206421782,
        9: -0.011536801725663837
    }

    for num_topics, score in coherence_scores.items():
        st.write(f"Num Topics = {num_topics}, Coherence Score = {score}")

    st.write("Based on the coherence scores, we decided to use 3 topics. Here are the topics and their top words:")

    topics_3 = {
        "Topic 0": ['inflation', 'rate', 'committee', 'economic', 'price', 'participant', 'policy', 'federal', 'growth', 'quarter'],
        "Topic 1": ['bank', 'financial', 'risk', 'reserve', 'federal', 'banking', 'institution', 'credit', 'community', 'capital'],
        "Topic 2": ['rate', 'policy', 'inflation', 'price', 'economy', 'growth', 'monetary', 'economic', 'productivity', 'term']
    }

    for topic, words in topics_3.items():
        st.write(f"{topic}: {', '.join(words)}")

    # Plot coherence scores
    st.subheader('Coherence Scores Graph')
    plot_coherence_scores(coherence_scores)

    # Plot word clouds for each topic
    st.subheader('Word Clouds for Each Topic')
    topic_dict = {0: 'Economic Indicators', 1: 'Financial Institutions', 2: 'Monetary Policy'}
    plot_word_clouds(lda_model, vectorizer, topic_dict)

# Topic Inference Page
elif page == 'Topic Inference':
    st.header('Topic Inference')

    # File uploader with progress bar
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    progress_bar = st.progress(0)

    if uploaded_file is not None:
        try:
            # Read file
            progress_bar.progress(10)
            text = uploaded_file.read().decode("utf-8")
            progress_bar.progress(30)

            # Preprocess text
            cleaned_text = preprocess_text(text)
            progress_bar.progress(50)

            # Infer topics
            topic_distribution = infer_topics(text)
            progress_bar.progress(70)

            # Display cleaned text
            st.subheader('Cleaned Text')
            st.write(cleaned_text)
            progress_bar.progress(80)

            # Display topic distribution
            st.subheader('Topic Distribution')

            topic_dict = {
                0: 'Economic Indicators',
                1: 'Financial Institutions',
                2: 'Monetary Policy'
            }

            for idx, score in enumerate(topic_distribution[0]):
                st.write(f"Topic {idx} ({topic_dict.get(idx, 'Unknown')}): {score:.4f}")

            # Plot topic distribution
            plot_topic_distribution(topic_distribution, topic_dict)
            progress_bar.progress(100)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            progress_bar.empty()
    else:
        st.info("Please upload a text file to preprocess.")

# VIX Historical Data Page
elif page == 'VIX Historical Data':
    st.header('Historical VIX Data')
    st.write('Below is an interactive chart of historical VIX data.')
    plot_vix_data()