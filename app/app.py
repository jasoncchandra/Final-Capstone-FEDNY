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
    vectorizer = joblib.load('vectorizer_v8.pkl')
    st.success("Vectorizer loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Vectorizer file not found: {e}")

try:
    lda_model = joblib.load('lda_model_8.pkl')
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
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)

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
        ax = axes[idx // 4, idx % 4]
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(topic_dict[topic])
        ax.axis('off')

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
    vix_data.columns = [col.strip() for col in vix_data.columns]
    vix_data['DATE'] = pd.to_datetime(vix_data['DATE'])
    fig = px.line(vix_data, x='DATE', y='CLOSE', title='Historical VIX Data')
    st.plotly_chart(fig)

def display_key_sentences(sentences, topic_distributions, topic_dict, top_n=5):
    dominant_topics = topic_distributions.argmax(axis=1)
    topic_sentences = {topic: [] for topic in topic_dict.keys()}

    for i, topic in enumerate(dominant_topics):
        topic_sentences[topic].append((sentences[i], topic_distributions[i][topic]))

    for topic in topic_dict.keys():
        sentences = sorted(topic_sentences[topic], key=lambda x: -x[1])[:top_n]
        with st.expander(f"Key Sentences for Topic {topic} ({topic_dict[topic]}):"):
            if sentences:
                for sentence, score in sentences:
                    st.write(f"{sentence} (Score: {score:.4f})")
            else:
                st.write("No sentences found for this topic.")

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
    st.write('We experimented with different numbers of topics to find the most relevant ones. It was a judgment call to select the optimal number of topics, balancing the need for distinct features with the desire to avoid having too many topics. Here are the coherence scores for different numbers of topics:')

    coherence_scores = {
        3: 0.09004151822108149,
        4: 0.0046645116832365555,
        5: -0.0045957082288738925,
        6: 0.0017441943397127069,
        7: 0.008529514229556334,
        8: 0.01629442206421782,
        9: -0.011536801725663837,
        10: -0.1372346118130309,
        11: -0.1595829856257721,
        12: -0.15113393781221862,
        13: -0.1431670808158615,
        14: -0.15929615987573326,
        15: -0.15891563765618963,
        16: -0.1583665063308525,
        17: -0.15529978708000322,
        18: -0.15851683435827796,
        19: -0.1515668846175406
    }

    # Plot coherence scores
    st.subheader('Coherence Scores Graph')
    plot_coherence_scores(coherence_scores)

    st.write("Based on the coherence scores, we decided to use 8 topics. Here are the topics and their top words:")

    topics_8 = {
        "Topic 0": ['payment', 'bank', 'reserve', 'service', 'federal', 'financial', 'check', 'currency', 'electronic', 'consumer'],
        "Topic 1": ['financial', 'crisis', 'liquidity', 'capital', 'stress', 'firm', 'asset', 'fund', 'requirement', 'regulation'],
        "Topic 2": ['friedman', 'gold', 'schwartz', 'hayek', 'carolina', '1929', 'charlotte', 'north', '1928', '1931'],
        "Topic 3": ['rate', 'price', 'economy', 'growth', 'policy', 'inflation', 'economic', 'productivity', 'country', 'capital'],
        "Topic 4": ['policy', 'financial', 'bank', 'reserve', 'federal', 'community', 'monetary', 'inflation', 'data', 'rate'],
        "Topic 5": ['inflation', 'price', 'growth', 'rate', 'economic', 'quarter', 'committee', 'policy', 'increase', 'recent'],
        "Topic 6": ['bank', 'risk', 'financial', 'banking', 'capital', 'management', 'institution', 'credit', 'organization', 'business'],
        "Topic 7": ['rate', 'participant', 'inflation', 'committee', 'economic', 'federal', 'policy', 'term', 'condition', 'security']
    }

    for topic, words in topics_8.items():
        st.write(f"{topic}: {', '.join(words)}")

    # Plot word clouds for each topic
    st.subheader('Word Clouds for Each Topic')
    topic_dict = {0: 'Payment Systems and Services', 1: 'Financial Crisis and Liquidity', 2: 'Historical Economic Figures',
                  3: 'Economic Growth and Policy', 4: 'Monetary Policy and Inflation', 5: 'Inflation and Economic Growth',
                  6: 'Banking and Risk Management', 7: 'Federal Reserve and Monetary Policy'}
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

            # Display cleaned text in a scrollable box
            st.subheader('Cleaned Text')
            st.text_area("", "\n".join(sentences), height=200)
            progress_bar.progress(80)

            # Display topic distribution
            st.subheader('Sentence-Level Topic Distribution')

            topic_dict = {
                0: 'Payment Systems and Services',
                1: 'Financial Crisis and Liquidity',
                2: 'Historical Economic Figures',
                3: 'Economic Growth and Policy',
                4: 'Monetary Policy and Inflation',
                5: 'Inflation and Economic Growth',
                6: 'Banking and Risk Management',
                7: 'Federal Reserve and Monetary Policy'
            }

            for idx in range(8):  # Ensure all topics are displayed
                count = topic_counts.get(idx, 0)
                st.write(f"Topic {idx} ({topic_dict.get(idx, 'Unknown')}): {count:.2%}")

            # Plot topic distribution
            plot_topic_distribution(topic_distributions, topic_dict)
            progress_bar.progress(90)

            # Display key sentences
            display_key_sentences(sentences, topic_distributions, topic_dict)
            progress_bar.progress(100)

            # Button to display sentiment analysis
            if st.button('Display Sentiment Analysis'):
                sentiment_progress = st.progress(0)
                sentiments = []
                for i, sentence in enumerate(sentences):
                    sentiment = analyze_sentiment(sentence)
                    sentiments.append(sentiment)
                    sentiment_progress.progress((i + 1) / len(sentences))

                avg_sentiment_by_topic = aggregate_sentiment_by_topic(sentences, topic_distributions, sentiments, num_topics=8)

                # Display average sentiment by topic
                st.subheader('Sentiment Analysis by Topic')
                for topic, avg_sentiment in avg_sentiment_by_topic.items():
                    st.write(f"Average sentiment for {topic_dict[int(topic.split('_')[1])]}: {avg_sentiment:.2f}")

                # Explanation of Sentiment Analysis
                st.subheader('Sentiment Analysis Primer')
                st.write('''
                The sentiment analysis is performed at the sentence level using the FinBERT model, which is fine-tuned for financial text. 
                Each sentence is assigned a sentiment score: -1 for negative, 0 for neutral, and 1 for positive. 

                We aggregate the sentiment scores for each topic, assigning higher weights to sentences with higher similarity scores to the topic. 
                This ensures that sentences more relevant to a topic have a greater influence on the overall sentiment score for that topic.
                ''')

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
    plot_vix_data('data/vix_data.csv')
