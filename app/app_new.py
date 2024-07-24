import streamlit as st
import os
import joblib
import pandas as pd
import plotly.express as px
import nltk
import string
import lime
from lime.lime_tabular import LimeTabularExplainer
import xgboost as xgb
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
import streamlit.components.v1 as components
import tempfile
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
page = st.sidebar.selectbox('Select a Page:', ['Introduction', 'Training Process', 'VIX Historical Data', 'Topic Modelling LDA Interaction', 'Historical Sentiment by Topic', 'FED Minutes Analysis', 'Feature Vectors', 'VIX Inference'])

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

# Topic Modelling LDA Interaction
elif page == 'Topic Modelling LDA Interaction':
    st.header('Topic Modelling LDA')
    st.write('Below is an interactive chart of how we split up the topic model')

    # Load the HTML file and display it
    with open('data/lda_vis.html', 'r') as f:
        html_data = f.read()
        
    components.html(html_data, width=1200, height=800, scrolling=True)

# New Streamlit Tab for Sentiment by Topic
elif page == 'Historical Sentiment by Topic':
    st.header('Sentiment by Topic Over Time')

    # Load the CSV file
    file_path = 'data/Final Sentiment by Topic.csv'
    try:
        sentiment_data = pd.read_csv(file_path)
        st.success("Sentiment data loaded successfully.")
    except FileNotFoundError as e:
        st.error(f"Sentiment data file not found: {e}")

    # Display the first few rows of the dataframe to understand its structure
    st.write("Preview of the Sentiment Data:")
    st.write(sentiment_data.head())

    # Set the 'month' column as the index for time series plotting
    sentiment_data['month'] = pd.to_datetime(sentiment_data['month'])
    sentiment_data.set_index('month', inplace=True)

    # Plotting the sentiment scores for each topic
    fig, ax = plt.subplots(figsize=(14, 8))

    for column in sentiment_data.columns[2:]:
        ax.plot(sentiment_data.index, sentiment_data[column], label=column)

    ax.set_title('Sentiment Scores by Topic Over Time')
    ax.set_xlabel('Time')
    ax.set_ylabel('Sentiment Score')
    ax.legend(loc='upper right')
    ax.grid(True)

    st.pyplot(fig)

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

        # Save the DataFrame to a temporary file
        temp_file_path = os.path.join(tempfile.gettempdir(), 'latest_stats_df.pkl')
        joblib.dump(latest_stats_df, temp_file_path)

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

# Additional Page for Displaying and Inputting Feature Vectors
elif page == 'Feature Vectors':
    st.header('Feature Vectors')

    # Define the topic_dict here for use in this page
    topic_dict = {
        0: "Quarterly Economic Performance",
        1: "Committee Deliberations and Policies",
        2: "Inflation and Prices",
        3: "Economic Growth and Business Activity",
        4: "Credit Markets and Financial Conditions",
        5: "Economic Forecasts and Projections"
    }

    # Option to use the most recent results or input new ones
    use_recent = st.radio("Select Option:", ("Use Most Recent Results", "Input New Results"))

    if use_recent == "Use Most Recent Results":
        st.subheader('Most Recent Feature Vectors')
        try:
            latest_stats_df = joblib.load(os.path.join(tempfile.gettempdir(), 'latest_stats_df.pkl'))
            st.dataframe(latest_stats_df)
        except Exception as e:
            st.error(f"Error loading recent feature vectors: {e}")
    else:
        st.subheader('Input Your Feature Vectors')

        # Define input fields
        input_category_id = st.text_input("Category ID")
        input_file_id = st.text_input("File ID (in YYYYMMDD format)")
        input_features = {}
        for topic in topic_dict.values():
            input_features[topic] = st.number_input(f"{topic} Feature Value", value=0.0)

        # Button to submit input
        if st.button("Submit Feature Vector"):
            # Create a new dataframe row with the input values
            new_data = {
                'category_id': input_category_id,
                'file_id': input_file_id,
                **input_features
            }

            new_df = pd.DataFrame([new_data])

            # Remove file extensions before converting to datetime
            new_df['file_id'] = new_df['file_id'].str.replace('.txt', '')

            # Convert file_id to datetime
            try:
                new_df['month'] = pd.to_datetime(new_df['file_id'], format='%Y%m%d').dt.to_period('M')
                new_df.set_index(['month'], inplace=True)
                
                # Append new data to existing dataframe and display it
                latest_stats_df = joblib.load(os.path.join(tempfile.gettempdir(), 'latest_stats_df.pkl'))
                latest_stats_df = pd.concat([latest_stats_df, new_df])
                joblib.dump(latest_stats_df, os.path.join(tempfile.gettempdir(), 'latest_stats_df.pkl'))
                st.success("Feature vector added successfully.")
                st.dataframe(latest_stats_df)
            except Exception as e:
                st.error(f"Error converting file_id to datetime: {e}")

# VIX Inference Page
elif page == 'VIX Inference':
    st.header('VIX Inference')

    # Load the necessary files for inference
    try:
        bst = xgb.Booster()
        bst.load_model('xgb_model.json')
        scaler_topics = joblib.load('xgb_scaler_topics.pkl')
        st.success("Model and scaler loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")

    # Load the most recent feature vectors
    try:
        latest_stats_df = joblib.load(os.path.join(tempfile.gettempdir(), 'latest_stats_df.pkl'))
        st.write("Latest Feature Vectors:")
        st.dataframe(latest_stats_df)
    except Exception as e:
        st.error(f"Error loading recent feature vectors: {e}")

    # Select the most recent row for lag features
    if not latest_stats_df.empty:
        last_row = latest_stats_df.iloc[-1]

        # Input new data for inference
        st.subheader('Input New Data for VIX Inference')

        input_data = {
            'file_id': st.text_input("Date (YYYYMMDD format)", value="20240128"),
            'Quarterly Economic Performance': st.number_input("Quarterly Economic Performance", value=0.0),
            'Committee Deliberations and Policies': st.number_input("Committee Deliberations and Policies", value=0.0),
            'Inflation and Prices': st.number_input("Inflation and Prices", value=0.0),
            'Economic Growth and Business Activity': st.number_input("Economic Growth and Business Activity", value=0.0),
            'Credit Markets and Financial Conditions': st.number_input("Credit Markets and Financial Conditions", value=0.0),
            'Economic Forecasts and Projections': st.number_input("Economic Forecasts and Projections", value=0.0)
        }
        if st.button("Predict VIX"):
                # Allow the user to select which row to use for lagged features
            st.subheader("Select a Feature Vector for Lagged Features")
            selected_index = st.selectbox("Select the index of the feature vector to use for lagged features:", latest_stats_df.index)

            # Create a DataFrame for the new data
            new_data = pd.DataFrame([input_data])

            # Prepare the input data for inference
            def prepare_input_data(new_data, last_row):
                # Create lagged features for each of the topic vectors
                for column in new_data.columns[1:]:
                    new_data[f'{column}_lag_1'] = last_row[column]
                
                # Scale the data
                X_topics_scaled = scaler_topics.transform(new_data.iloc[:, 1:])
                
                return X_topics_scaled

            # Select the specified row for lagged features
            last_row = latest_stats_df.loc[selected_index]

            X_topics_scaled = prepare_input_data(new_data, last_row)

            # Convert to DMatrix
            dtest = xgb.DMatrix(X_topics_scaled)

            # Predict the VIX value
            y_pred = bst.predict(dtest)

            st.write(f'Predicted VIX: {y_pred[0]}')

            # Confidence Interval Calculation
            st.header('Confidence Interval')
            y_preds = []
            num_samples = 1000  # Number of bootstrap samples

            for _ in range(num_samples):
                bootstrap_sample = np.random.choice(X_topics_scaled.flatten(), size=X_topics_scaled.shape[1], replace=True)
                dtest_bootstrap = xgb.DMatrix(bootstrap_sample.reshape(1, -1))
                y_pred_bootstrap = bst.predict(dtest_bootstrap)
                y_preds.append(y_pred_bootstrap[0])

            lower_bound = np.percentile(y_preds, 2.5)
            upper_bound = np.percentile(y_preds, 97.5)

            st.write(f'95% Confidence Interval for Predicted VIX: [{lower_bound}, {upper_bound}]')

            # Additional plots
            st.header('Distribution of Predictions')
            fig, ax = plt.subplots()
            ax.hist(y_preds, bins=30, edgecolor='k', alpha=0.7)
            ax.axvline(np.mean(y_preds), color='r', linestyle='--', label='Mean Prediction')
            ax.axvline(lower_bound, color='g', linestyle='--', label='2.5th Percentile')
            ax.axvline(upper_bound, color='b', linestyle='--', label='97.5th Percentile')
            ax.set_xlabel('Predicted VIX')
            ax.set_ylabel('Frequency')
            ax.legend()
            st.pyplot(fig)
