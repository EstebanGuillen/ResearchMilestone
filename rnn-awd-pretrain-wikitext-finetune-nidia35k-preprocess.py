
from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
from pathlib import Path

path_clas = Path('/home/ubuntu/data/autopsy')
path_lm = Path('/home/ubuntu/data/medical/nidia27k_preprocess')

batch_size=32
epochs=2
drop_mult=0.1
learning_rate=1e-3
wd=1e-4
num_folds = 5

folds = ['data_suicide_homicide_k_1.csv','data_suicide_homicide_k_2.csv','data_suicide_homicide_k_3.csv','data_suicide_homicide_k_4.csv','data_suicide_homicide_k_5.csv']

accuracy_list = []
i = 0
for f in folds:
  i = i + 1
  print("\nFold: " + str(i))
  data_lm = TextLMDataBunch.from_csv(path_lm, f, classes=['neg','pos'])
  data_clas = TextClasDataBunch.from_csv(path_clas,f, vocab=data_lm.train_ds.vocab, classes=['Suicide','Homicide'])

  learn = text_classifier_learner(data_clas, drop_mult=drop_mult)
  learn.load_encoder('enc_nidia_pretrained')
  learn.unfreeze()
  learn.fit(epochs,learning_rate, wd=wd)
  
  acc = (learn.validate())[1].item()
  accuracy_list.append(acc)

print('\nAccuracy List')
print(accuracy_list)

print("\nAverage Accuracy")
print( (sum(accuracy_list))/ (float(num_folds)  ))
