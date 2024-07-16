import streamlit as st
import os
import joblib
import pandas as pd
import plotly.express as px
import nltk
import string
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import WordPunctTokenizer
from nltk.data import LazyLoader
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from corpusutils import CorpusPreProcess, Document, Corpus
from featureutils import FeatureProcessor, find_closest

# Download necessary NLTK data
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

# Streamlit app
st.title('VIX Predictor via Text')

# Sidebar for navigation
st.sidebar.title('Navigation')
page = st.sidebar.selectbox('Select a Page:', ['Introduction', 'Training Process', 'VIX Historical Data', 'FED Minutes Analysis'])

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

# VIX Historical Data Page
elif page == 'VIX Historical Data':
    st.header('Historical VIX Data')
    st.write('Below is an interactive chart of historical VIX data.')

    def plot_vix_data(file_path):
        try:
            vix_data = pd.read_csv(file_path)
            fig = px.line(vix_data, x='DATE', y='CLOSE', title='Historical VIX Data')
            st.plotly_chart(fig)
        except FileNotFoundError as e:
            st.error(f"VIX data file not found: {e}")

    plot_vix_data('data/vix_data.csv')
    
# FED Minutes Analysis Page
elif page == 'FED Minutes Analysis':
    st.title("FED Minutes Analysis")

    # User input for directory and file pattern
    directory = st.text_input("Enter the directory path:", value=os.path.join(os.getcwd(), "data"))
    file_pattern = st.text_input("Enter the file pattern (e.g., '*.txt'):", value='*.txt')

    # Initialize progress bar
    progress_bar = st.progress(0)
    progress_step = 0

    # Directory settings
    current_directory = os.getcwd()
    root = directory
    word_tokenizer = WordPunctTokenizer()
    sent_tokenizer = LazyLoader("tokenizers/punkt/english.pickle")
    category_pattern = r'(\d{4})/*'
    file_pattern = file_pattern
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english') + list(string.punctuation) + ['u', '.', 's', '--', '-', '."', ',"', '.)', ')-', '".', '—', '),']

    # Display the directory and file pattern
    st.write(f"Analyzing text files in directory: {directory} with pattern: {file_pattern}")

    # Button to start analysis
    if st.button("Run Analysis"):
        # Progress step: Data Preparation
        progress_step += 10
        progress_bar.progress(progress_step)

        fed_mins = CorpusPreProcess(root=root, file_extension=file_pattern,
                                    category_pattern=category_pattern,
                                    file_pattern=file_pattern,
                                    word_tokenizer=word_tokenizer,
                                    sent_tokenizer=sent_tokenizer,
                                    stemmer=stemmer,
                                    lemmatizer=lemmatizer,
                                    stop_words=stop_words
                                    )

        # Load the LDA model and the vectorizer
        try:
            lda_model = joblib.load('lda_model_6_topics.pkl')
            vectorizer = joblib.load('count_vectorizer.pkl')
            st.success("LDA model and vectorizer loaded successfully.")
        except Exception as e:
            st.error(f"Error loading LDA model or vectorizer: {e}")

        # Progress step: Load Models
        progress_step += 20
        progress_bar.progress(progress_step)

        # Renaming topics
        topic_dict = {
            0: "Quarterly Economic Performance",
            1: "Committee Deliberations and Policies",
            2: "Inflation and Prices",
            3: "Economic Growth and Business Activity",
            4: "Credit Markets and Financial Conditions",
            5: "Economic Forecasts and Projections"
        }

        st.write("Renamed topics:")
        for key, value in topic_dict.items():
            st.write(f"Topic {key}: {value}")

        # Load tokenizer and model
        transformer_tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        transformer_model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert", output_hidden_states=True, output_attentions=False)

        # Progress step: Load Transformer Model
        progress_step += 20
        progress_bar.progress(progress_step)

        # Process the input files
        latest_minutes = fed_mins.get_paras(flatten=True, stem=True)
        tokenizer_settings = dict(is_split_into_words=True, max_length=350, padding='max_length', truncation=True, return_tensors="pt")

        document_feat = FeatureProcessor(latest_minutes,
                                         transformer_model=transformer_model,
                                         transformer_tokenizer=transformer_tokenizer,
                                         tokenizer_settings=tokenizer_settings,
                                         lda_model=lda_model,
                                         lda_vec=vectorizer,
                                         lda_topic_dict=topic_dict, batch_size=30)

        # Transform documents to get topic distributions
        transformed_docs = vectorizer.transform(latest_minutes)
        topic_distributions = lda_model.transform(transformed_docs)

        # Convert topic distributions to DataFrame
        topic_distributions_df = pd.DataFrame(topic_distributions, columns=[f'Topic {i}' for i in range(len(topic_dict))])

        # Calculate mean topic distribution
        mean_topic_distribution = topic_distributions_df.mean()

        # Display mean topic distribution
        st.header('Mean Topic Distribution for the Entire Corpus')
        mean_topic_distribution.index = [topic_dict[i] for i in range(len(topic_dict))]
        st.write(mean_topic_distribution)

        # Plot mean topic distribution
        fig = px.bar(mean_topic_distribution, x=mean_topic_distribution.index, y=mean_topic_distribution.values, title='Mean Topic Distribution')
        st.plotly_chart(fig)

        # Progress step: Topic Distribution Displayed
        progress_step += 30
        progress_bar.progress(progress_step)

        # Find top 3 examples for each topic
        top_examples = {}
        for topic_idx in range(len(topic_dict)):
            top_docs_idx = topic_distributions_df[f'Topic {topic_idx}'].nlargest(3).index
            top_examples[topic_idx] = [latest_minutes[i] for i in top_docs_idx]

        st.header('Top 3 Examples for Each Topic')
        for topic_idx, examples in top_examples.items():
            with st.expander(f'Topic {topic_idx}: {topic_dict[topic_idx]}'):
                for i, example in enumerate(examples):
                    st.write(f'Example {i+1}:')
                    st.write(example)
                    st.write('---')

        # Continue with sentiment analysis
        latest_minutes = document_feat.get_features(sentiment=True, embedding=True, topic=True)

        def corpus_stats(corpus):
            ids = [(n, f.category_id, f.file_id) for n, f in enumerate(corpus)]
            df_ids = pd.DataFrame(ids, columns=['idx', 'category_id', 'file_id'])
            start_idx = df_ids.drop_duplicates(['category_id', 'file_id'], keep='first')
            end_idx = df_ids.drop_duplicates(['category_id', 'file_id'], keep='last')

            idx = start_idx.merge(end_idx, on=['category_id', 'file_id'], suffixes=('_start', '_end'))

            stats = []
            corpus_sent = corpus.extract_features('sentiment')
            corpus_topic = corpus.extract_features('topics')

            for s, cat_id, file_id, e in idx.values:
                net_tone = [s['logits'][0] - s['logits'][1] for s in corpus_sent[s:e]]
                topic_dist = [s['topic_dist'] for s in corpus_topic[s:e]]
                topic_sentiment = np.asarray(topic_dist) * np.asarray(net_tone).reshape(-1, 1)
                topic_mean_sent = topic_sentiment.mean(axis=0).tolist()

                stats.append((cat_id, file_id, *topic_mean_sent))
            return stats

        latest_stats = corpus_stats(latest_minutes)

        # Progress step: Calculate Statistics
        progress_step += 10
        progress_bar.progress(progress_step)

        # Create DataFrame with the correct columns
        columns = ['category_id', 'file_id'] + list(topic_dict.values())
        latest_stats_df = pd.DataFrame(latest_stats, columns=columns)

        # Remove file extensions before converting to datetime
        latest_stats_df['file_id'] = latest_stats_df['file_id'].str.replace('.txt', '')

        # Convert file_id to datetime
        try:
            latest_stats_df['month'] = pd.to_datetime(latest_stats_df['file_id'], format='%Y%m%d').dt.to_period('M')
            latest_stats_df.set_index(['month'], inplace=True)
        except Exception as e:
            st.error(f"Error converting file_id to datetime: {e}")

        # Plot using matplotlib
        st.header('Average net-tone per topic for all files')
        fig, ax = plt.subplots()
        latest_stats_df.drop(['category_id', 'file_id'], axis=1).plot(kind='bar', ax=ax, figsize=(10, 5))
        st.pyplot(fig)

        # Progress step: Complete Analysis
        progress_step += 10
        progress_bar.progress(progress_step)

        # Output Analysis
        def output_analysis():
            st.header("Analysis Completed")
            st.write("The analysis of the FED minutes has been completed. The results are displayed above.")
            st.balloons()

        output_analysis()
