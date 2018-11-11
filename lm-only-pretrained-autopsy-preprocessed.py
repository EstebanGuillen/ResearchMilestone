from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
from pathlib import Path

path_lm = Path('/home/ubuntu/data/autopsy')


data_lm = TextLMDataBunch.from_csv(path_lm,'data_suicide_homicide_k_1.csv', classes=['Suicide','Homicide'])


learn = language_model_learner(data_lm, pretrained_model=URLs.WT103)

learn.save_encoder('enc_autopsy_only_pretrained')
