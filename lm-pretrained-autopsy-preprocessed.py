from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
import fastai.datasets

path_lm = '/home/ubuntu/data/autopsy'

batch_size = 32
epochs=20
drop_mult=0.5

data_lm = TextLMDataBunch.from_csv(path_lm)
data_lm.train_dl.dl.bs = batch_size
data_lm.valid_dl.dl.bs = batch_size


learn = RNNLearner.language_model(data_lm, pretrained_model=URLs.WT103, drop_mult=drop_mult)
learn.unfreeze()
learn.fit(epochs, slice(1e-4,1e-3))

learn.save_encoder('enc_20_epochs_autopsy_preprocess')


