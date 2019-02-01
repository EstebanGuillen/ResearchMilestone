from fastai.text import *
from pathlib import Path
import pandas as pd

import os
import time


#usage: python run_experiments.py <num_layers> <embedding_size> <num_hidden> <architecture> <path-to-data> <data-set> <gpu-id> <epochs> <path-to-weights> <path-to-vocab> 

num_layers = int(sys.argv[1])

embedding_size = int(sys.argv[2])

num_hidden_units = int(sys.argv[3])

architecture = str(sys.argv[4])

path_to_data = str(sys.argv[5])

data_set = str(sys.argv[6])

gpu_id = int(sys.argv[7])

epochs = int(sys.argv[8])

path_to_weights = str(sys.argv[9])

path_to_vocab = str(sys.argv[10])

torch.cuda.set_device(gpu_id)



learning_rates_with_pre = [slice(1e-5,1e-3),slice(1e-4,1e-2)]
learning_rates_without_pre = [5e-4,5e-3]
learning_rates = []
drop_out_values = [0.2, 0.5, 0.8]
batch_size=32

num_folds = 5

folds_s_h = ['data_suicide_homicide_k_1.csv','data_suicide_homicide_k_2.csv','data_suicide_homicide_k_3.csv','data_suicide_homicide_k_4.csv','data_suicide_homicide_k_5.csv']

folds_s_h_a = ['data_suicide_homicide_accident_k_1.csv','data_suicide_homicide_accident_k_2.csv','data_suicide_homicide_accident_k_3.csv','data_suicide_homicide_accident_k_4.csv','data_suicide_homicide_accident_k_5.csv']

folds = []
classes = []
if data_set == 's_h':
    folds = folds_s_h
    classes = ['Suicide','Homicide']
else:   
    folds = folds_s_h_a
    classes = ['Suicide','Homicide','Accident']

using_qrnn = False
if architecture == 'qrnn':
    using_qrnn = True

using_pre_trained = False
if path_to_weights != 'NA' and path_to_weights != 'NA':
    using_pre_trained = True

path = Path(path_to_data)
data_lm = TextLMDataBunch.from_csv(path, 'data.csv', classes=classes, bs=batch_size)

learn = language_model_learner(data_lm, qrnn=using_qrnn, emb_sz=embedding_size, nh=num_hidden_units, nl=num_layers)

if using_pre_trained:
    #no pretrained model
    weights = path_to_weights
    vocab = path_to_vocab 

    learn.load_pretrained(weights,vocab)

    learn.freeze()


    learn.fit_one_cycle(2, 1e-2, moms=(0.8,0.7))

    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))

    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))

    learn.unfreeze()
    learn.fit_one_cycle(10, 1e-3, moms=(0.8,0.7))
else:
    #no pretrained model
    learn.unfreeze()
    learn.fit_one_cycle(10, 1e-2, moms=(0.8,0.7))



#name of the encoder

lm_encoder_name = 'encoder' + '_' + architecture + '_' + str(num_layers) + '_' + data_set

learn.save_encoder(lm_encoder_name)




#loop over hyperparams and folds
if using_pre_trained:
    learning_rates = learning_rates_with_pre
else:
    learning_rates = learning_rates_without_pre

count = 0
for lr in learning_rates:
    for drop in drop_out_values:
        print('')
        print('STARTING CROSS VAL:',architecture,str(num_layers),data_set,lr, str(drop) )
        for fold in folds:
            print(fold)
            data_clas = TextClasDataBunch.from_csv(path, fold, vocab=data_lm.train_ds.vocab, classes=classes, bs=batch_size)
            learn = text_classifier_learner(data_clas, drop_mult=drop, qrnn=using_qrnn, emb_sz=embedding_size, nh=num_hidden_units, nl=num_layers)



            learn.load_encoder(lm_encoder_name)

            learn.freeze()

            learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))

            learn.freeze_to(-2)
            learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))


            learn.freeze_to(-3)
            learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))


            learn.unfreeze()

            learn.fit_one_cycle(epochs, lr, moms=(0.8,0.7))
            #TODO save all results of each fold into a file
            
            
            classifier_name = architecture + '_' + str(num_layers) + '_' + str(count)
            learn.save(classifier_name)
            count = count + 1
        print('ENDING CROSS VAL:',architecture,str(num_layers),data_set,lr, str(drop) )
        print('')

