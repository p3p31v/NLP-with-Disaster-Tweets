# src/train.py

# Import Libraries
import joblib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import re #regular expressions
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.corpus import stopwords
import string
import nltk
from gensim.models import Word2Vec
import gensim.downloader as api
from sklearn.decomposition import PCA
# Stopwords set defined
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

#Function to remove irrelevant data
def clean_tweet(tweet):

    """
    Regex expressions website https://regex101.com/
    """

    # Capital letters to lowercase
    tweet = tweet.lower()
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove user mentions
    tweet = re.sub(r'@[^\s]+', '', tweet)
    # Remove hashtags
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    # Remove punctuation
    tweet = re.sub('[%s]' % re.escape(string.punctuation), '', tweet)
    # Remove #RT
    tweet = re.sub(r'\b(rt)\b','',tweet)
    # Remove words containing numbers
    tweet = re.sub('\w*\d\w*' , '', tweet)
    # Remove single characters
    tweet = re.sub(r'\b[a-zA-Z]\b','',tweet)
    # Remove stopwords
    tweet = ' '.join([word for word in tweet.split() if word not in stop_words])

    return tweet

# Import data
df = pd.read_csv("./input/nlp-getting-started/train.csv")

# We apply the function to remove irrelevant data to the text column of the train dataset
df['text'] = [clean_tweet(tweet) for tweet in df['text']]
    
# We create a new column called kfold and fill it with -1
df["kfold"] = -1

# The next step is to randomize the rows of the data 
df = df.sample(frac=1,random_state=42).reset_index(drop=True)

# Split data
X_train = df.text
y_train = df.target

# Preprocess text
X_train_preprocessed = [clean_tweet(tweet) for tweet in X_train]

# Initialize CountVectorizer and fit count_vec on training data
count_vec = CountVectorizer(tokenizer=word_tokenize,token_pattern=None,ngram_range=(1,2))
X_train_count_vec = count_vec.fit_transform(X_train_preprocessed)

# Load a pre-trained Word2Vec model
word2vec_model = api.load("word2vec-google-news-300")

# Define a function to convert a sentence to its Word2Vec representation
def sentence_to_w2v(sentence, model):
    words = sentence.split()
    word_vectors = [model[word] for word in words if word in model]
    if word_vectors:
        return np.mean(word_vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
    
# Convert text data to Word2Vec representations
X_train_w2v = np.array([sentence_to_w2v(sentence, word2vec_model) for sentence in X_train_preprocessed])

# Initialize PCA with the desired number of components (e.g., 300)
pca = PCA(n_components=300)

# Fit and transform the Word2Vec representations
X_train_w2v_reduced = pca.fit_transform(X_train_w2v)

# Initialize RandomForest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model with training data
model.fit(X_train_w2v_reduced, y_train)

# Initiate the kfold class from model_selection module
kf = StratifiedKFold(n_splits=5)

# Calculate f1 score
f1 = cross_val_score(model, X_train_w2v, y_train, scoring="f1", cv=kf).mean()
print(f"F1 score with Word2Vec: {f1}")

# save the model
#joblib.dump(model,f"../models/model_{fold_}_{datetime.now()}")

# Import sample submission
sample_submission = pd.read_csv("./input/nlp-getting-started/sample_submission.csv")

# Import test competition data
test = pd.read_csv("./input/nlp-getting-started/test.csv")

# Split data
X_test = test.text

# Preprocess text
X_test_preprocessed = [clean_tweet(tweet) for tweet in X_test]

# Convert test competition data to vectors
X_test_count_vec = count_vec.transform(X_test_preprocessed)

# Predict 
sample_submission["target"] = model.predict(X_test_count_vec)

# Create submission file
sample_submission.to_csv("submission.csv", index=False)