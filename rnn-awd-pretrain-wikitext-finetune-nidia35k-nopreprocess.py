from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
import fastai.datasets

path_lm = '/home/ubuntu/data/medical'
path_clas = '/home/ubuntu/data/autopsy'


data_lm = TextLMDataBunch.from_csv(path_lm)
data_lm.train_dl.dl.bs = 32
data_lm.valid_dl.dl.bs = 32

data_clas = TextClasDataBunch.from_csv(path_clas, vocab=data_lm.train_ds.vocab)
data_clas.train_dl.dl.bs = 32
data_clas.valid_dl.dl.bs = 32


learn = RNNLearner.classifier(data_clas, drop_mult=0.1)
learn.load_encoder('enc_20_epochs_0_5_drop_mult_lr_e-4_to_e-3')

learn.unfreeze()
learn.fit(20, 1e-3, wd=1e-4)

