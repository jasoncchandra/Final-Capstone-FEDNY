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

# Define the return_self function
def return_self(x):
    return x

# Check if the files exist in the current directory
st.write("Current directory:", os.getcwd())
st.write("Files in current directory:", os.listdir(os.getcwd()))

# Load the vectorizer and LDA model
try:
    vectorizer = joblib.load('count_vectorizer.pkl')
    st.success("Vectorizer loaded successfully.")
except FileNotFoundError as e:
    st.error(f"Vectorizer file not found: {e}")

try:
    lda_model = joblib.load('lda_model_6_topics.pkl')
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
    ax.set_title('Average Topic Distribution (Weighted)')
    ax.set_xlabel('Average Topic Distribution (Weighted)')
    ax.set_ylabel('Topics')
    st.pyplot(fig)

def plot_word_clouds(lda_model, vectorizer, topic_dict):
    from wordcloud import WordCloud

    feature_names = vectorizer.get_feature_names_out()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)

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
        ax = axes[idx // 3, idx % 3]
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(topic_dict[topic])
        ax.axis('off')

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
                st.write("No sentences found where this topic is dominant.")

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
    st.write('In this project, we performed topic modeling on Federal Reserve meeting minutes to extract key themes. '
             'We used Latent Dirichlet Allocation (LDA) for topic modeling. Here are the identified topics and their descriptions:')

    topic_descriptions = {
        "Topic 0": "Quarterly Economic Performance\n\n"
                   "Representation: Discussions about quarterly economic performance, loan activities, and credit markets.\n"
                   "Justification: The FOMC reviews quarterly economic data and considers the state of the credit markets for economic stability and growth.",

        "Topic 1": "Committee Deliberations and Policies​\n\n"
                   "Representation: Focus on inflation and the committee's policy decisions.\n"
                   "Justification: Inflation control is a primary objective of the FOMC, involving policy measures to manage inflation.",

        "Topic 2": "Inflation and Prices​\n\n"
                   "Representation: Covers price levels, inflation measurements, and labor market conditions.\n"
                   "Justification: Monitoring price levels and labor market conditions helps assess economic health and inflationary pressures.",

        "Topic 3": "Economic Growth and Business Activity​\n\n"
                   "Representation: Related to overall economic growth, forecasts, and projections.\n"
                   "Justification: Economic forecasts and growth projections are essential for planning and adjusting monetary policy.",

        "Topic 4": "Credit Markets and Financial Conditions\n\n"
                   "Representation: Focus on market conditions and financial stability.\n"
                   "Justification: Ensuring stable market conditions and a sound financial system is a core responsibility of the FOMC.",

        "Topic 5": "Economic Forecasts and Projections​\n\n"
                   "Representation: Covers business activities and consumer spending.\n"
                   "Justification: Understanding business activities and consumer spending patterns helps gauge economic momentum and support sustainable growth."
    }

    for topic, description in topic_descriptions.items():
        st.subheader(topic)
        st.write(description)
        
    # Display the topic-word table as a DataFrame
    topic_word_data = {
        "Topic 0": ["quarter", "loan", "increas", "real", "remain", "declin", "continu", "credit", "sale", "consum"],
        "Topic 1": ["inflat", "committe", "particip", "polici", "would", "econom", "member", "rate", "feder", "risk"],
        "Topic 2": ["inflat", "price", "labor", "month", "rate", "measur", "percent", "year", "increas", "remain"],
        "Topic 3": ["growth", "econom", "project", "year", "quarter", "economi", "rate", "forecast", "gdp", "staff"],
        "Topic 4": ["market", "period", "rate", "intermeet", "bank", "financi", "treasuri", "fund", "secur", "remain"],
        "Topic 5": ["particip", "busi", "spend", "price", "growth", "note", "continu", "recent", "sector", "market"]
    }

    topic_word_df = pd.DataFrame(topic_word_data)
    st.dataframe(topic_word_df)

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
            st.subheader('Dominant Topic Sentence-Level Distribution')

            topic_dict = {
                0: 'Quarterly Economic Performance',
                1: 'Committee Deliberations and Policies',
                2: 'Inflation and Prices',
                3: 'Economic Growth and Business Activity',
                4: 'Credit Markets and Financial Conditions',
                5: 'Economic Forecasts and Projections​'
            }

            # Calculate and display the topic percentages based on average topic distribution
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

                avg_sentiment_by_topic = aggregate_sentiment_by_topic(sentences, topic_distributions, sentiments, num_topics=6)

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
