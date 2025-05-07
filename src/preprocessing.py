import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from src import config

def preprocess_data(df):
    """
    Preprocess the input DataFrame by filling missing values, encoding labels,
    and vectorizing text data.

    Returns:
        X_train: Training feature vectors
        X_test: Testing feature vectors
        y_train: Training labels
        y_test: Testing labels
        vectorizer: Fitted TfidfVectorizer object
        le: Fitted LabelEncoder object
    """
    # Fill NA and lowercase titles
    df['Product Title'] = df['Product Title'].fillna("").str.lower()
    # Remove punctuation, numbers, and extra whitespace
    df['Product Title'] = df['Product Title'].apply(lambda x: re.sub(r'\W+|\d+', ' ', x).strip())

    # Encode target labels
    le = LabelEncoder()
    df['Category Encoded'] = le.fit_transform(df['Category Label'])

    # TF-IDF vectorization of Product Titles with improved parameters
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.95,
        min_df=5,
        ngram_range=(1, 2),
        max_features=5000
    )
    X = vectorizer.fit_transform(df['Product Title'])
    y = df['Category Encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE
    )

    return X_train, X_test, y_train, y_test, vectorizer, le