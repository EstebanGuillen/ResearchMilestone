from fastai.text import *
from pathlib import Path
import pandas as pd
import os
import time
import timeit

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


learning_rates = [slice(3e-4,3e-3)]
drop_out_values = [ 0.5]

#TODO if there is time add mom values to the hyperparameter search
mom_values = [(.95,.85),(.8,.7)]

batch_size=32
num_folds = 1
folds_s_h_a_n = ['data_plus_case_id.csv']


folds = folds_s_h_a_n
classes = ['Suicide','Homicide','Accident','Natural'] 


using_qrnn= False
if architecture == 'qrnn':
    using_qrnn = True


using_pre_trained = False
if path_to_weights != 'NA' and path_to_vocab != 'NA':
    using_pre_trained = True

path = Path(path_to_data)


df = pd.read_csv(path/'data_plus_case_id.csv')

valid_df =  df.loc[df['is_valid']==True]
train_df = df.loc[df['is_valid']==False]

data_lm = TextLMDataBunch.from_df(path, train_df=train_df, valid_df=valid_df, classes=classes,bs=batch_size)


learn = language_model_learner(data_lm, qrnn=using_qrnn, emb_sz=embedding_size, nh=num_hidden_units, nl=num_layers)

if using_pre_trained:
    #pretrained model
    print('Using pre-trained model')
    weights = path_to_weights
    vocab = path_to_vocab 

    learn.load_pretrained(weights,vocab)

    learn.freeze()

    #do some gradual unfreezing, so we don't lose the pretrained info
    learn.fit_one_cycle(2, 1e-2, moms=(0.8,0.7))

    learn.freeze_to(-2)
    learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))

    learn.freeze_to(-3)
    learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))

    learn.unfreeze()
    learn.fit_one_cycle(8, 1e-3, moms=(0.8,0.7))
else:
    #no pretrained model, so we just unfreeze it
    learn.unfreeze()
    
#name of the encoder
pre_trained_string = ''
if using_pre_trained:
    pre_trained_string = 'pre_trained'
else:
    pre_trained_string = 'not_pre_trained'

lm_encoder_name =  architecture + '_' + str(num_layers) + '_' + data_set + '_' + pre_trained_string + '_' + 'explain_y_n'

learn.save_encoder(lm_encoder_name)

start = timeit.default_timer()

for lr in learning_rates:
    for drop in drop_out_values:
        print('')
        print('STARTING CROSS VAL:',architecture,str(num_layers),data_set,lr, str(drop) )
        fold_id = 1 
        for fold in folds:
            print(fold)
           

            df = pd.read_csv(path/fold)

            valid_df =  df.loc[df['is_valid']==True]
            train_df = df.loc[df['is_valid']==False]

            data_clas = TextClasDataBunch.from_df(path, train_df=train_df, valid_df=valid_df, vocab=data_lm.train_ds.vocab,classes=classes,bs=batch_size)  


            learn = text_classifier_learner(data_clas, drop_mult=drop, qrnn=using_qrnn, emb_sz=embedding_size, nh=num_hidden_units, nl=num_layers)



            learn.load_encoder(lm_encoder_name)
            
            if using_pre_trained:
                learn.freeze()

                learn.fit_one_cycle(1, 2e-2, moms=(0.8,0.7))

                learn.freeze_to(-2)
                learn.fit_one_cycle(1, slice(1e-2/(2.6**4),1e-2), moms=(0.8,0.7))


                learn.freeze_to(-3)
                learn.fit_one_cycle(1, slice(5e-3/(2.6**4),5e-3), moms=(0.8,0.7))


            learn.unfreeze()

            learn.fit_one_cycle(epochs, lr, moms=(0.8,0.7))
            
            import csv
            header = ['correct','prediction','actual','case_id','COD','report']
            with open('explain_3_layer_awd_y_n.csv', 'w') as f: 
                writer = csv.writer(f, delimiter=',')
                writer.writerow(header) 
                #iterate over the validation set and save the miss classifications 
                for index, row in valid_df.iterrows():
                    pred_cat,pred_id,probs = learn.predict(row['text'])
                    actual = row['label']
                    pred = str(pred_cat)
                    if pred != actual:
                        #print('pred:',pred,'actual:',actual,'COD:',row['cod'])
                        row = ['n',pred,actual,row['case_id'],row['cod'],row['text']]
                        writer.writerow(row)
                    else:
                        row = ['y',pred,actual,row['case_id'],row['cod'],row['text']]
                        writer.writerow(row)
            f.close() 

        print('ENDING CROSS VAL:',architecture,str(num_layers),data_set,lr, str(drop) )
        print('')

stop = timeit.default_timer()

print('Time for full CROSS VAL run: ', stop - start) 
