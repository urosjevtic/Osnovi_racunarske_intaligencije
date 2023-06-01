import pandas as pd
import numpy as np
# from sklearn.base import accuracy_score
# import spacy # biblioteka za procesiranje teksta
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
import re

df = pd.read_csv('south_park_train.csv')
print(df)

df['Line'] = df['Line'].apply(lambda x: x.lower())
df['Line'] = df['Line'].apply(lambda x: re.sub(r"[^\w\s]","", x))
df['Line'] = df['Line'].apply(lambda x: " ".join([word for word in x.split() if len(word) >= 2]))

print(df)

df = df.dropna()

x_train, x_test, y_train, y_test = train_test_split(df['Line'], df['Character'], test_size = 0.3, random_state = 42)

vect = CountVectorizer(stop_words="english", ngram_range=((1,2)))
x_train = vect.fit_transform(x_train)
x_test = vect.transform(x_test)


#vect = TfidfVectorizer(stop_words='english', ngram_range=((1,1)))
#x_train = vect.fit_transform(x_train)
#x_test = vect.transform(x_test)

#nb = MultinomialNB().fit(x_train, y_train)
#y_pred = nb.predict(x_test)

mlp = MLPClassifier(hidden_layer_sizes=[256, 128, 64], max_iter=5, learning_rate_init=0.01, verbose=True).fit(x_train, y_train)
y_pred = mlp.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)
print(f'Accuracy: {accuracy}')