# src/train.py

# Import Libraries
import joblib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import re #regular expressions
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk import ngrams
from nltk.corpus import stopwords
import string
import nltk

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

<<<<<<< HEAD:src/train.py
# Import data and randomize the rows of the data 
df = pd.read_csv("../input/nlp-getting-started/train.csv")
=======
# Import data
df = pd.read_csv("./input/nlp-getting-started/train.csv")

# We apply the function to remove irrelevant data to the text column of the train dataset
df['text'] = [clean_tweet(tweet) for tweet in df['text']]
    
# We create a new column called kfold and fill it with -1
df["kfold"] = -1

# The next step is to randomize the rows of the data 
>>>>>>> 5d1fa94b18092abde3c8e68eedb6df3d6970a1fe:src/__main__.py
df = df.sample(frac=1,random_state=42).reset_index(drop=True)

# Split data
X_train = df.text
y_train = df.target

# Preprocess text
X_train_preprocessed = [clean_tweet(tweet) for tweet in X_train]

# Initialize CountVectorizer and fit count_vec on training data
count_vec = CountVectorizer(tokenizer=word_tokenize,token_pattern=None,ngram_range=(1,2))
X_train_count_vec = count_vec.fit_transform(X_train_preprocessed)

# Initialize logistic regression model
model = LogisticRegression(solver='lbfgs', max_iter=1000).fit(X_train_count_vec, y_train)

# Initiate the kfold class from model_selection module
kf = StratifiedKFold(n_splits=5)

# Calculate f1 score
f1 = cross_val_score(model, X_train_count_vec, y_train, scoring="f1", cv=kf).mean()
print (f"f1 score: {f1}")

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