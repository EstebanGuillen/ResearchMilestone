import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.python.keras.layers import Input, RNN,LSTM, Bidirectional, Dense, Embedding, Dropout, SpatialDropout1D, GRU
from keras import optimizers

import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

from sklearn.model_selection import KFold
import os

file_path = '/Users/esteban/data/autopsy/data.csv'
data = pd.read_csv(file_path,
                          header=None, encoding='ISO-8859-1',
                          names=['label', 'text'])

data = data.loc[data['label'].isin(['Suicide','Homicide'])]
max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' ')
tokenizer.fit_on_texts(data['text'].values)
X = tokenizer.texts_to_sequences(data['text'].values)
X = pad_sequences(X)
Y = pd.get_dummies(data['label']).values

print(Y.shape)
print(X.shape)



maxlen = X.shape[1]
embedding_size=300
batch_size = 32
lstm_out = 196
epochs = 20

def make_model(batch_size=None):
  source = Input(shape=(maxlen,), batch_size=batch_size, dtype=tf.int32, name='Input')
  embedding = Embedding(input_dim=max_features, output_dim=embedding_size, input_length = X.shape[1],name='Embedding')(source)
  drop = SpatialDropout1D(0.5)(embedding)
  #rnn =  Bidirectional(LSTM(lstm_out, name = 'LSTM',dropout=0.50, recurrent_dropout=0.50))(drop)
  rnn =  RNN(lstm_out, name = 'RNN',dropout=0.40, recurrent_dropout=0.40)(drop)
  predicted_var = Dense(2, activation='sigmoid', name='Output')(rnn)
  model = tf.keras.Model(inputs=[source], outputs=[predicted_var])
  model.compile(
      #optimizer='rmsprop',
      optimizer=tf.keras.optimizers.RMSprop(decay=1e-3),
      loss = 'categorical_crossentropy',
      metrics=['acc'])
  return model


history_list = []

kfold_splits = 10
kf = KFold(n_splits=kfold_splits, shuffle=True)

for index, (train_indices, val_indices) in enumerate(kf.split(X)):
  xtrain, xval = X[train_indices], X[val_indices]
  ytrain, yval = Y[train_indices], Y[val_indices]
  
  tf.keras.backend.clear_session()
  training_model = None
  training_model = make_model(batch_size = batch_size)


  
  history = training_model.fit(xtrain, ytrain,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_data=(xval,yval))
  
  history_list.append(history)

  accuracy_history = history.history['acc']
  val_accuracy_history = history.history['val_acc']
  print('*****************************************************************************')
  print('*****************************************************************************')
  print('*****************************************************************************')
  print ("Last training accuracy: " + str(accuracy_history[-1]) + ", last validation accuracy: " + str(val_accuracy_history[-1]) )
  print('*****************************************************************************')
  print('*****************************************************************************')
  print('*****************************************************************************')
  
  #eval_history = tpu_model.evaluate(xval, yval, batch_size=batch_size * num_tpu)

sum = 0.0
for h in history_list:
  val_accuracy_history = h.history['val_acc']
  final_val_accuracy = val_accuracy_history[-1]
  sum = sum + final_val_accuracy

print('')
print('*****************************************************************************')
print('*****************************************************************************')
print('*****************************************************************************')
print("average accuracy:", (sum/10.0))
print('*****************************************************************************')
print('*****************************************************************************')
print('*****************************************************************************')
