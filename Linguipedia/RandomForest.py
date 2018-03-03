# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset1 = pd.read_csv('train_set.csv')
dataset2 = pd.read_csv('test_tweets_set.csv')

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus1 = []
for i in range(0, 31962):
    Tweet = re.sub('[^a-zA-Z]', ' ', dataset1['tweet'][i])
    Tweet = Tweet.lower()
    Tweet = Tweet.split()
    ps = PorterStemmer()
    Tweet = [ps.stem(word) for word in Tweet if not word in set(stopwords.words('english'))]
    Tweet = ' '.join(Tweet)
    corpus1.append(Tweet)
 
corpus2 = []    
for i in range(0, 17197):
    Tweet = re.sub('[^a-zA-Z]', ' ', dataset2['tweet'][i])
    Tweet = Tweet.lower()
    Tweet = Tweet.split()
    ps = PorterStemmer()
    Tweet = [ps.stem(word) for word in Tweet if not word in set(stopwords.words('english'))]
    Tweet = ' '.join(Tweet)
    corpus2.append(Tweet)
    

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 21000)
X_train = cv.fit_transform(corpus1).toarray()
y_train = dataset1.iloc[:, 0].values

cv = CountVectorizer(max_features = 21000)
X_test = cv.fit_transform(corpus2).toarray()





from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 1000, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

'''# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)'''
