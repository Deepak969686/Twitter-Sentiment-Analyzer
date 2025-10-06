# File: train.py

import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from tqdm import tqdm

# --- Text Preprocessing ---
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
port_stem = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

# --- Training Function ---
def train_and_save_model():
    column_names = ['target', 'ids', 'date', 'flag', 'user', 'text']
    dataset_path = 'training.1600000.processed.noemoticon.csv'

    df = pd.read_csv(dataset_path, names=column_names, encoding='ISO-8859-1')
    df = df[['target', 'text']]
    df['target'] = df['target'].replace(4, 1)

    sample_df = df.sample(n=1600000, random_state=42)

    tqdm.pandas()
    sample_df['stemmed_content'] = sample_df['text'].progress_apply(stemming)

    X = sample_df['stemmed_content'].values
    Y = sample_df['target'].values
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, Y_train)

    # Evaluation
    train_pred = model.predict(X_train_tfidf)
    test_pred = model.predict(X_test_tfidf)
    print("Train Accuracy:", accuracy_score(Y_train, train_pred))
    print("Test Accuracy:", accuracy_score(Y_test, test_pred))

    # Save model
    with open('trained_model.pkl', 'wb') as f:
        pickle.dump({'model': model, 'vectorizer': vectorizer}, f)

if __name__ == '__main__':
    train_and_save_model()
