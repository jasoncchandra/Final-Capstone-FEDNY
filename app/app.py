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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

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

# Load FinBERT model for sentiment analysis
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Define the preprocessing function
def preprocess_text(text):
    word_tokenizer = WordPunctTokenizer()
    sent_tokenizer = LazyLoader("tokenizers/punkt/english.pickle")
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english')) - {'.'}
    punctuation = set(string.punctuation) - {'.'}
    
    sentences = sent_tokenizer.tokenize(text)
    cleaned_sentences = []

    for sentence in sentences:
        words = word_tokenizer.tokenize(sentence)
        words = [word for word in words if word.lower() not in stop_words and word not in punctuation]
        words = [lemmatizer.lemmatize(word) for word in words]
        cleaned_sentences.append(" ".join(words))

    return cleaned_sentences

def infer_topics(sentences):
    preprocessed_sentences = [sentence for sentence in sentences if sentence]
    text_vectorized = vectorizer.transform(preprocessed_sentences)
    topic_distributions = lda_model.transform(text_vectorized)
    return topic_distributions

def plot_topic_distribution(topic_distribution, topic_dict):
    topics = list(topic_dict.values())
    scores = topic_distribution.mean(axis=0)

    fig, ax = plt.subplots()
    sns.barplot(x=scores, y=topics, ax=ax)
    ax.set_title('Topic Distribution')
    ax.set_xlabel('Average Score')
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

def analyze_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits.detach().numpy()
    sentiment = logits[0].argmax()
    sentiment_mapping = {0: -1, 1: 0, 2: 1}
    return sentiment_mapping[sentiment]

def aggregate_sentiment_by_topic(sentences, topic_distributions, sentiments, num_topics):
    aggregated_sentiment = {f"Topic_{i}": [] for i in range(num_topics)}

    for sentence, topic_dist, sentiment in zip(sentences, topic_distributions, sentiments):
        for topic_idx in range(num_topics):
            topic_weight = topic_dist[topic_idx]
            aggregated_sentiment[f"Topic_{topic_idx}"].append(sentiment * topic_weight)

    avg_sentiment = {topic: np.mean(scores) if scores else 0 for topic, scores in aggregated_sentiment.items()}
    return avg_sentiment

# Function to plot VIX historical data
def plot_vix_data(file_path):
    vix_data = pd.read_csv(file_path)
    vix_data['Date'] = pd.to_datetime(vix_data['Date'])
    fig = px.line(vix_data, x='Date', y='VIX Close', title='Historical VIX Data')
    st.plotly_chart(fig)

def display_key_sentences(sentences, topic_distributions, topic_dict, top_n=5):
    dominant_topics = topic_distributions.argmax(axis=1)
    topic_sentences = {topic: [] for topic in topic_dict.keys()}

    for i, topic in enumerate(dominant_topics):
        topic_sentences[topic].append((sentences[i], topic_distributions[i][topic]))

    for topic, sentences in topic_sentences.items():
        sentences = sorted(sentences, key=lambda x: -x[1])[:top_n]
        with st.expander(f"Key Sentences for Topic {topic} ({topic_dict[topic]}):"):
            for sentence, score in sentences:
                st.write(f"{sentence} (Score: {score:.4f})")

def search_sentences(sentences, query):
    results = [sentence for sentence in sentences if query.lower() in sentence.lower()]
    return results

# Streamlit app
st.title('VIX Predictor via Text')

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select a Page:', ['Introduction', 'Training Process', 'Topic Modeling & Sentiment Analysis', 'VIX Historical Data'])

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

# Topic Modeling & Sentiment Analysis Page
elif page == 'Topic Modeling & Sentiment Analysis':
    st.header('Topic Modeling & Sentiment Analysis')

    # File uploader with progress bar
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    progress_bar = st.progress(0)

    sentences = []

    if uploaded_file is not None:
        try:
            # Read file
            progress_bar.progress(10)
            text = uploaded_file.read().decode("utf-8")
            progress_bar.progress(30)

            # Preprocess text into sentences
            sentences = preprocess_text(text)
            progress_bar.progress(50)

            # Infer topics for each sentence
            topic_distributions = infer_topics(sentences)
            progress_bar.progress(70)

            # Determine the dominant topic for each sentence
            dominant_topics = topic_distributions.argmax(axis=1)

            # Aggregate topic counts
            topic_counts = pd.Series(dominant_topics).value_counts(normalize=True).sort_index()

            # Display cleaned text
            st.subheader('Cleaned Text')
            st.write(" ".join(sentences))
            progress_bar.progress(80)

            # Display topic distribution
            st.subheader('Sentence-Level Topic Distribution')

            topic_dict = {
                0: 'Economic Indicators',
                1: 'Financial Institutions',
                2: 'Monetary Policy'
            }

            for idx, count in topic_counts.items():
                st.write(f"Topic {idx} ({topic_dict.get(idx, 'Unknown')}): {count:.2%}")

            # Plot topic distribution
            plot_topic_distribution(topic_distributions, topic_dict)
            progress_bar.progress(90)

            # Display key sentences
            display_key_sentences(sentences, topic_distributions, topic_dict)
            progress_bar.progress(100)

            # Button to perform sentiment analysis
            if st.button('Perform Sentiment Analysis'):
                progress_bar = st.progress(0)
                sentiments = []
                for sentence in sentences:
                    sentiment = analyze_sentiment(sentence)
                    sentiments.append(sentiment)
                progress_bar.progress(50)

                avg_sentiment_by_topic = aggregate_sentiment_by_topic(sentences, topic_distributions, sentiments, num_topics=3)
                progress_bar.progress(100)

                # Display average sentiment by topic
                st.subheader('Sentiment Analysis by Topic')
                for topic, avg_sentiment in avg_sentiment_by_topic.items():
                    st.write(f"Average sentiment for {topic_dict[int(topic.split('_')[1])]}: {avg_sentiment:.2f}")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")
            progress_bar.empty()
    else:
        st.info("Please upload a text file to preprocess.")

    # Add a search box in the Topic Inference Page
    st.subheader('Search Sentences')
    query = st.text_input('Enter a term to search for:')
    if query:
        search_results = search_sentences(sentences, query)
        st.write(f"Found {len(search_results)} sentences containing '{query}':")
        for result in search_results:
            st.write(result)

# VIX Historical Data Page
elif page == 'VIX Historical Data':
    st.header('Historical VIX Data')
    st.write('Below is an interactive chart of historical VIX data.')
    plot_vix_data()
