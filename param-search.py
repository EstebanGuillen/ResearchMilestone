from fastai import *        # Quick access to most common functionality
from fastai.text import *   # Quick access to NLP functionality
from pathlib import Path
from fastai.vision import *
import os
import time


folds = ['data_suicide_homicide_k_1.csv','data_suicide_homicide_k_2.csv','data_suicide_homicide_k_3.csv','data_suicide_homicide_k_4.csv','data_suicide_homicide_k_5.csv']

def fit_without_pretraining(batch_size,learning_rate,epochs,wd,drop_mult,path_clas, emb_sz=400, nh=1150, nl=3):
    accuracy_list = []
    for f in folds:
        data_clas = TextClasDataBunch.from_csv(path_clas,f, classes=['Suicide','Homicide'],bs=batch_size)
        learn = text_classifier_learner(data_clas, drop_mult=drop_mult,emb_sz=emb_sz,nh=nh,nl=nl)
        learn.unfreeze()
        learn.fit(epochs,learning_rate, wd=wd)
        
        accuracy_list.append(learn.validate()[1].item())
        avg = sum(accuracy_list)/len(accuracy_list)
    return (avg,accuracy_list)

def fit_with_pretraining(batch_size,learning_rate,epochs,wd,drop_mult,path_clas,data_lm,enc,emb_sz=400, nh=1150, nl=3):
    accuracy_list = []
    for f in folds:
        data_clas = TextClasDataBunch.from_csv(path_clas,f, vocab=data_lm.train_ds.vocab, classes=['Suicide','Homicide'], bs=batch_size)

        learn = text_classifier_learner(data_clas, drop_mult=drop_mult,emb_sz=emb_sz,nh=nh,nl=nl)
        learn.load_encoder(enc)

        learn.freeze()
        learn.fit(4,learning_rate, wd=wd)
        learn.unfreeze()
        learn.fit(epochs,learning_rate, wd=wd)
        
        accuracy_list.append(learn.validate()[1].item())
        avg = sum(accuracy_list)/len(accuracy_list)
    return (avg,accuracy_list)



path_clas = Path('/home/ubuntu/data/autopsy')
path_lm_autopsy = Path('/home/ubuntu/data/autopsy')
path_lm_nidia = Path('/home/ubuntu/data/medical/nidia27k_preprocess')


batch_size=32
#drop_mult=[0.1,0.3,0.5,0.7]
drop_mult=[0.7]
learning_rate=[1e-4,1e-3,1e-2]
wd=1e-7
epochs=20

data_lm_autopsy = TextLMDataBunch.from_csv(path_lm_autopsy, 'data_suicide_homicide_combined_train_test.csv', classes=['Suicide','Homicide'])
data_lm_nidia = TextLMDataBunch.from_csv(path_lm_nidia,'documents-preprocess-valid.csv', classes=['neg','pos'], bs=batch_size)

timestr = time.strftime("%Y%m%d-%H%M%S")
f = open("hyper-search" + "-" + timestr + ".csv", "x")

start_time = time.time()

for drop in drop_mult:
    for lr in learning_rate:
        #SIMPLE-AWD (no pretraining)
        print('')
        print("SIMPLE-AWD: " + "drop=" + str(drop) + " " + "lr=" + str(lr))
        avg, accuracy_list = fit_without_pretraining(batch_size,lr,epochs,wd,drop,path_clas,emb_sz=300,nh=198,nl=1)
        f.write("SIMPLE-AWD" + ',' + str(drop) + ',' + str(lr) + ',' + str(avg) + ',' + str(accuracy_list) + '\n')
        f.flush()
        os.fsync(f.fileno())
        elapsed_time = time.time() - start_time
        print('total elapsed time:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        #Simple-AWD-finetuned-autopsy
        print('')
        print("SIMPLE-AWD-Finetuned-Autopsy: " + "drop=" + str(drop) + " " + "lr=" + str(lr))
        avg, accuracy_list = fit_with_pretraining(batch_size,lr,epochs,wd,drop,path_clas,data_lm_autopsy,'enc_autopsy_not_pretrained_simple',emb_sz=300,nh=198,nl=1)
        f.write("SIMPLE-AWD-Finetuned-Autopsy" + ',' + str(drop) + ',' + str(lr) + ',' + str(avg) + ',' + str(accuracy_list) + '\n')
        f.flush()
        os.fsync(f.fileno())
        elapsed_time = time.time() - start_time
        print('total elapsed time:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        #Simple-AWD-finetuned-nidia
        print('')
        print("SIMPLE-AWD-Finetuned-Nidia: " + "drop=" + str(drop) + " " + "lr=" + str(lr))
        avg, accuracy_list = fit_with_pretraining(batch_size,lr,epochs,wd,drop,path_clas,data_lm_nidia,'enc_nidia_not_pretrained_simple',emb_sz=300,nh=198,nl=1)
        f.write("SIMPLE-AWD-Finetuned-Nidia" + ',' + str(drop) + ',' + str(lr) + ',' + str(avg) + ',' + str(accuracy_list) + '\n')
        f.flush()
        os.fsync(f.fileno())
        elapsed_time = time.time() - start_time
        print('total elapsed time:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        #AWD (no pretraining)
        print('')
        print("AWD: " + "drop=" + str(drop) + " " + "lr=" + str(lr))
        avg, accuracy_list = fit_without_pretraining(batch_size,lr,epochs,wd,drop,path_clas)
        f.write("AWD" + ',' + str(drop) + ',' + str(lr) + ',' + str(avg) + ',' + str(accuracy_list) + '\n')
        f.flush()
        os.fsync(f.fileno())
        elapsed_time = time.time() - start_time
        print('total elapsed time:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        #AWD-pretrained-Wiki
        print('')
        print("AWD-Pretrained-Wiki: " + "drop=" + str(drop) + " " + "lr=" + str(lr))
        avg, accuracy_list = fit_with_pretraining(batch_size,lr,epochs,wd,drop,path_clas,data_lm_autopsy,'enc_autopsy_only_pretrained')
        f.write("AWD-Pretrained-Wiki" + ',' + str(drop) + ',' + str(lr) + ',' + str(avg) + ',' + str(accuracy_list) + '\n')
        f.flush()
        os.fsync(f.fileno())
        elapsed_time = time.time() - start_time
        print('total elapsed time:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        #AWD-finetuned-autopsy
        print('')
        print("AWD-Finetuned-Autopsy: " + "drop=" + str(drop) + " " + "lr=" + str(lr))
        avg, accuracy_list = fit_with_pretraining(batch_size,lr,epochs,wd,drop,path_clas,data_lm_autopsy,'enc_autopsy_not_pretrained')
        f.write("AWD-Finetuned-Autopsy" + ',' + str(drop) + ',' + str(lr) + ',' + str(avg) + ',' + str(accuracy_list) + '\n')
        f.flush()
        os.fsync(f.fileno())
        elapsed_time = time.time() - start_time
        print('total elapsed time:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        #AWD-finetuned-nidia
        print('')
        print("AWD-Finetuned-Nidia: " + "drop=" + str(drop) + " " + "lr=" + str(lr))
        avg, accuracy_list = fit_with_pretraining(batch_size,lr,epochs,wd,drop,path_clas,data_lm_nidia,'enc_nidia_not_pretrained')
        f.write("AWD-Finetuned-Nidia" + ',' + str(drop) + ',' + str(lr) + ',' + str(avg) + ',' + str(accuracy_list) + '\n')
        f.flush()
        os.fsync(f.fileno())
        elapsed_time = time.time() - start_time
        print('total elapsed time:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        #AWD-pretrained-Wiki-finetuned-autopsy
        print('')
        print("AWD-Pretrained-Wiki-Finetuned-Autopsy: " + "drop=" + str(drop) + " " + "lr=" + str(lr))
        avg, accuracy_list = fit_with_pretraining(batch_size,lr,epochs,wd,drop,path_clas,data_lm_autopsy,'enc_autopsy_pretrained')
        f.write("AWD-Pretrained-Wiki-Finetuned-Autopsy" + ',' + str(drop) + ',' + str(lr) + ',' + str(avg) + ',' + str(accuracy_list) + '\n')
        f.flush()
        os.fsync(f.fileno())
        elapsed_time = time.time() - start_time
        print('total elapsed time:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

        #AWD-pretrained-Wiki-finetuned-nidia
        print('')
        print("AWD-Pretrained-Wiki-Finetuned-Nidia: " + "drop=" + str(drop) + " " + "lr=" + str(lr))
        avg, accuracy_list = fit_with_pretraining(batch_size,lr,epochs,wd,drop,path_clas,data_lm_nidia,'enc_nidia_pretrained')
        f.write("AWD-Pretrained-Wiki-Finetuned-Nidia" + ',' + str(drop) + ',' + str(lr) + ',' + str(avg) + ',' + str(accuracy_list) + '\n')
        f.flush()
        os.fsync(f.fileno())
        elapsed_time = time.time() - start_time
        print('total elapsed time:',time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))


f.close()

