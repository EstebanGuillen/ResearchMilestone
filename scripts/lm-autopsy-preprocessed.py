
from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
from pathlib import Path

path_lm = Path('/home/ubuntu/data/autopsy')

batch_size = 32
epochs=12
drop_mult=0.5


data_lm = TextLMDataBunch.from_csv(path_lm,'data_suicide_homicide_k_1.csv', classes=['Suicide','Homicide'])


learn = language_model_learner(data_lm, drop_mult=drop_mult)

learn.unfreeze()
learn.fit(epochs, slice(1e-4,1e-2))

learn.save_encoder('enc_autopsy_not_pretrained')


