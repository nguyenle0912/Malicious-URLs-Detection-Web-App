import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle

# Reading data from csv file
data = pd.read_csv("urldata.csv")
data = data.drop('Unnamed: 0',axis=1)

# Labels
y = data["label"]

# Features
url_list = data["url"]

# Using Tokenizer
vectorizer = TfidfVectorizer()

# Store vectors into X variable as Our XFeatures
X = vectorizer.fit_transform(url_list)

# save the vectorizer to disk
with open('vectorizer.pk', 'wb') as fin:
    pickle.dump(vectorizer, fin)

# Split into training and testing dataset 80:20 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building using logistic regression
logit = LogisticRegression()
logit.fit(X_train, y_train)
# Accuracy of Our Model
print("Accuracy of our model is: ", logit.score(X_test, y_test))

# save the model to disk
pickle.dump(logit, open('detect_mal_url_model.sav', 'wb'))