import os
import pandas as pd
import numpy as np
import re
import string
import nltk
import sys
from nltk.tokenize import WordPunctTokenizer
from nltk.data import LazyLoader
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import warnings

warnings.filterwarnings('ignore')

def main():
    # Step 1: Mount Google Colab Drive
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)  # Force remount if already mounted

    # Step 2: Navigate to your project folder
    project_dir = '/content/drive/MyDrive/Capstone - NY FED/Final-Capstone-FEDNY'
    if not os.path.exists(project_dir):
        raise FileNotFoundError(f"The directory {project_dir} does not exist. Please check the path.")

    os.chdir(project_dir)

    # Adjust Python path to include the utils directory and the project directory
    sys.path.append(project_dir)
    sys.path.append(os.path.join(project_dir, 'utils'))

    # Verify the files in the utils directory
    utils_path = os.path.join(project_dir, 'utils')
    print("Files in utils directory:")
    print(os.listdir(utils_path))

    from corpusutils import CorpusPreProcess, Document, Corpus
    from featureutils import FeatureProcessor, find_closest

    # Download NLTK data
    nltk.download('stopwords')
    nltk.download('punkt')
    
    # Set up paths and configurations
    current_directory = os.getcwd()
    root = os.path.join(current_directory, "Data_Training_Minutes")
    word_tokenizer = WordPunctTokenizer()
    sent_tokenizer = LazyLoader("tokenizers/punkt/english.pickle")
    category_pattern = r'(\d{4})/*'
    file_extension = r'*.txt'
    file_pattern = r'(\d{8})/*'
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words('english') + list(string.punctuation) + ['u', '.', 's', '--', '-', '."', ',"', '.)', ')-', '".', '—', '),']

    # Initialize CorpusPreProcess
    fed_mins = CorpusPreProcess(root=root, file_extension=file_extension,
                                category_pattern=category_pattern,
                                file_pattern=file_pattern,
                                word_tokenizer=word_tokenizer,
                                sent_tokenizer=sent_tokenizer,
                                stemmer=stemmer,
                                lemmatizer=lemmatizer,
                                stop_words=stop_words)
    
    # Truncate text based on specified regex patterns
    start_regex = r"""(?i)(staff\sreview\sof\sthe\seconomic|the\sinformation\s[\s]?(?:reviewed|received|provided)|the\scommittee\sthen\sturned\sto\sa\sdiscussion\sof\sthe\seconomic\soutlook|in\sthe\scommittee[\']?s\sdiscussion\sof\scurrent\sand\sprospective\seconomic)"""
    end_regex = r"""(?i)(at\sthe\sconclusion\sof\s[\s]?(?:the|this)\s(?:discussion|meetings)|the\scommitte\svoted\sto\sauthorize|the\svote\sencompassed\sapproval\sof)"""
    fed_mins.truncate_text(start_regex, end_regex)

    # LDA model preparation
    paras = fed_mins.get_paras(flatten=True, stem=True)

    def return_self(x): return x

    vectorizer = CountVectorizer(tokenizer=return_self, lowercase=False, preprocessor=return_self)
    bag_of_words = vectorizer.fit_transform((doc.stem for doc in paras))

    num_topics = 6

    lda_model = LatentDirichletAllocation(n_components=num_topics,
                                          max_iter=20,
                                          learning_method='online',
                                          random_state=10,
                                          n_jobs=-1)

    lda_transform = lda_model.fit_transform(bag_of_words)

    # Summarize topics
    def summarize_topics(model, feature_names, no_top_words):
        topics = pd.DataFrame()
        for topic_idx, topic in enumerate(model.components_):
            top_words = [feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]
            topics["Topic %d:" % (topic_idx)] = top_words
        return topics

    df_results = summarize_topics(lda_model, vectorizer.get_feature_names_out(), 10)
    print(df_results)

if __name__ == "__main__":
    main()
