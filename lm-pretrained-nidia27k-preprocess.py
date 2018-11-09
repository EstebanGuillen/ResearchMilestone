from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
from pathlib import Path

path_lm = Path('/home/ubuntu/data/medical/nidia27k_preprocess')

batch_size = 16
epochs=20
drop_mult=0.75


data_lm = TextLMDataBunch.from_csv(path_lm,'documents-preprocess-valid.csv', classes=['neg','pos'])


learn = language_model_learner(data_lm, drop_mult=0.3, pretrained_model=URLs.WT103)

learn.unfreeze()
learn.fit(epochs, slice(1e-4,1e-2))

learn.save_encoder('enc_20_epochs_nidia27k_preprocess')


