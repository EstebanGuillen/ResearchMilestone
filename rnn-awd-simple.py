from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
import fastai.datasets

path_clas = '/home/ubuntu/data/autopsy'



data_clas = TextClasDataBunch.from_csv(path_clas, max_vocab=2000)
data_clas.train_dl.dl.bs = 32
data_clas.valid_dl.dl.bs = 32


learn = RNNLearner.classifier(data_clas, drop_mult=0.5, emb_sz=300, nh=198, nl=1)
learn.unfreeze()
learn.fit(20, 1e-3, wd=1e-4)

