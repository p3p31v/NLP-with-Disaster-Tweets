# src/train.py

# Import Libraries
import joblib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import linear_model, model_selection, metrics
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from nltk.tokenize import word_tokenize

# Import data
df = pd.read_csv("../input/nlp-getting-started/train.csv")

# We create a new column called kfold and fill it with -1
df["kfold"] = -1

# The next step is to randomize the rows of the data 
df = df.sample(frac=1).reset_index(drop=True)

# Fetch labels
y = df["target"].values

# Initiate the kfold class from model_selection module
kf = model_selection.StratifiedKFold(n_splits=5)

# Fill the new fold column
for f,(t_,v_) in enumerate(kf.split(X=df,y=y)):
    df.loc[v_,"kfold"] = f

# We go over the folds created
for fold_ in range(5):
    # temporary dataframes from train and test
    train_df = df[df.kfold != fold_].reset_index(drop=True)
    test_df = df[df.kfold == fold_].reset_index(drop=True)

    # Initialize CountVectorizer with NLTK's word_tokenize
    # Function as tokenizer
    count_vec = CountVectorizer(tokenizer=word_tokenize,token_pattern=None)

    # Fit count_vec on training data
    count_vec.fit(train_df["text"])

    # Transform training and validation data 
    xtrain = count_vec.transform(train_df["text"])
    xtest = count_vec.transform(test_df["text"])

    # Initialize logistic regression model
    model = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000)
    
    # Fit the model on training data and target
    model.fit(xtrain, train_df["target"])

    # Make preditions on test data
    # Threshold for predictions is 0.5
    preds = model.predict(xtest)

    # Calculate f1
    f1 = metrics.f1_score(test_df["target"],preds)

    print(f"Fold: {fold_}")
    print(f"f1 score = {f1}")
    print("")

# Import sample submission
sample_submission = pd.read_csv("../input/nlp-getting-started/sample_submission.csv")

# Import test competition data
test = pd.read_csv("../input/nlp-getting-started/test.csv")

# Convert test competition data to vectors
test = count_vec.transform(test["text"])

# Predict 
sample_submission["target"] = model.predict(test)

# Create submission file
sample_submission.to_csv("submission.csv", index=False)