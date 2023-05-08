from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from joblib import load
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter
import numpy as np

#Function to get Cluster ID of the data
def cluster(df):
    with open('model.pkl', 'rb') as f:
        kmeans_model, vectorizer = load(f)

    # Use the loaded KMeans model and vectorizer for prediction
    X = vectorizer.transform(df['Normalized Text'])
    cluster = kmeans_model.predict(X)
    df['Cluster ID'] = cluster
    return df

#Function to get Topic ID from the data
def topic(df):
    with open('lda_model.pkl', 'rb') as f:
        lda, vectorizer2 = load(f)

    # Transform text data using the loaded vectorizer
    X = vectorizer2.transform(df['Normalized Text'])

    # Get the topic distribution for each document
    doc_topic_dist = lda.transform(X)

    # Get the top two topics for each document based on topic distribution
    top_two_topics = np.argsort(-doc_topic_dist, axis=1)[:, :2]

    # Create topic IDs for the top two topics and add them to the dataframe
    topic_ids = ['T' + str(topic_idx) for topic_idx in top_two_topics.flatten()]
    topic_ids = np.array(topic_ids)

    df['Topic_ids'] = np.split(topic_ids, len(df))
    return df

#Function to get keywords from normalized data
def keywords_text(text, max_words = 4):
    tokens = word_tokenize(text)
    # Get a set of stop words from the nltk corpus
    stop_words = set(stopwords.words('english'))

    # Clean the tokens by removing stop words and non-alphabetic tokens
    cleaned_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.isalpha()]

    # Create a stemmer object
    stemmer = PorterStemmer()

    # Stem the cleaned tokens
    stemmed_tokens = [stemmer.stem(token) for token in cleaned_tokens]

    # Get the frequency of each stemmed token
    word_freq = Counter(stemmed_tokens)

    # Get the total number of words in the text and the number of unique words
    total_words = sum(word_freq.values())
    important_words = [word for word, freq in word_freq.items() if freq > total_words/len(word_freq)]
    
    return important_words[:max_words]

#Function to get summary from normalized data
def summary(text, max_words = 4):
    # Tokenize the input text
    tokens = word_tokenize(text)
    # Get a set of stop words from the nltk corpus
    stop_words = set(stopwords.words('english'))
    cleaned_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and token.isalpha()]

    # Stem the tokens using the PorterStemmer
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in cleaned_tokens]

    # Counting the frequency of each stemmed token
    word_freq = Counter(stemmed_tokens)

    # Calculating the total number of words and the frequency threshold for important words
    total_words = sum(word_freq.values())
    important_words = [word for word, freq in word_freq.items() if freq > total_words/len(word_freq)]
    
    return ' '.join(important_words[:max_words])    