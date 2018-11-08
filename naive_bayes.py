from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics

from sklearn.model_selection import KFold
import os
from sklearn.metrics import accuracy_score
import pandas as pd
import sys

#file_path = '/Users/esteban/data/autopsy/data.csv'

print(sys.argv[1])

file_path = sys.argv[1]

data = pd.read_csv(file_path,
                          header=None, encoding='ISO-8859-1',
                          names=['label', 'text'])

data = data.loc[data['label'].isin(['Suicide','Homicide'])]
data.label = pd.Categorical(data.label)
data['label_code'] = data.label.cat.codes


X = data['text'].values

Y = data['label_code'].values
print(Y)
print(data['label'].values)

kfold_splits = 5
kf = KFold(n_splits=kfold_splits, shuffle=True)

history = []
for index, (train_indices, test_indices) in enumerate(kf.split(X)):
  xtrain, xtest = X[train_indices], X[test_indices]
  ytrain, ytest = Y[train_indices], Y[test_indices]


  vectorizer = CountVectorizer(stop_words='english')
  
  train_features = vectorizer.fit_transform(xtrain)
  test_features = vectorizer.transform(xtest)
  

  nb = MultinomialNB()
  nb.fit(train_features, ytrain)


  predictions = nb.predict(test_features)
  accuracy = accuracy_score(ytest,predictions)
  history.append(accuracy)
  print (accuracy)
print(history)

sum = 0.0
for acc in history:
  sum = sum + acc
print ('average accuracy:', (sum/(kfold_splits)))
