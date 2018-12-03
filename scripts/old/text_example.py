from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality


path = untar_data(URLs.IMDB_SAMPLE)
print(path)

df = pd.read_csv(path/'texts.csv', header=None)
print(df.head())


data_lm = TextLMDataBunch.from_csv(path, 'texts.csv')
data_clas = TextClasDataBunch.from_csv(path, 'texts.csv', vocab=data_lm.train_ds.vocab, bs=42)

learn = language_model_learner(data_lm, pretrained_model=URLs.WT103)
learn.unfreeze()
learn.fit(2, slice(1e-4,1e-2))


learn.save_encoder('enc')

learn = text_classifier_learner(data_clas)
learn.load_encoder('enc')
learn.fit(3, 1e-3)

