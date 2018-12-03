
from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
from pathlib import Path

from shutil import copyfile


path_lm = Path('/home/ubuntu/data/autopsy')



batch_size = 32
epochs=50
drop_mult=0.30


data_lm = TextLMDataBunch.from_csv(path_lm,'documents_and_autopsy.csv', classes=['Suicide','Homicide'], bs=batch_size)


learn = language_model_learner(data_lm, drop_mult=drop_mult, emb_sz=300, nh=198, nl=1)

learn.unfreeze()
learn.fit(epochs, slice(1e-4,1e-2))

learn.save_encoder('enc_documents_and_autopsy')


