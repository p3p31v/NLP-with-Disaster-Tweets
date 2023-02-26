# src/train.py

# Import Libraries
import joblib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import feature_extraction, linear_model, model_selection, preprocessing
from datetime import datetime

# Import data
train_df = pd.read_csv("../input/nlp-getting-started/train.csv")
test_df = pd.read_csv("../input/nlp-getting-started/test.csv")

# Building vectors
count_vectorizer = feature_extraction.text.CountVectorizer()

# Now let's create vectors for all of our tweets.
train_vectors = count_vectorizer.fit_transform(train_df["text"])

"""
    note that we're NOT using .fit_transform() here. Using just .transform() makes sure
    that the tokens in the train vectors are the only ones mapped to the test vectors - 
    i.e. that the train and test vectors use the same set of tokens.
"""

test_vectors = count_vectorizer.transform(test_df["text"])

# Our model
"""
    Our vectors are really big, so we want to push our model's weights
    toward 0 without completely discounting different words - ridge regression 
    is a good way to do this.
"""

clf = linear_model.RidgeClassifier()

# Cross-validation and scores
scores = model_selection.cross_val_score(clf, train_vectors, train_df["target"], cv=3, scoring="f1")
scores

# Predictions
clf.fit(train_vectors, train_df["target"])

# save the model
# joblib.dump(clf,f"../models/model_{datetime.now()}.bin")

# Read submission file
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

# Fill the target column
sample_submission["target"] = clf.predict(test_vectors)
sample_submission.head()

# Export submission to csv file

sample_submission.to_csv("submission.csv", index=False)