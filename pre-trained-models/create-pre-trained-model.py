import sys



#usage: python create-pre-trained-model.py <num_layer> <embedding_size> <num_hidden_units> <architecture> <path_to_data> <gpu_id>


num_layers = int(sys.argv[1])

embedding_size = int(sys.argv[2])

num_hidden_units = int(sys.argv[3])

architecture = str(sys.argv[4])

path_to_data = str(sys.argv[5])

gpu_id = int(sys.argv[6])



print('STARTING:',num_layers,embedding_size,num_hidden_units,architecture,path_to_data,gpu_id)



from fastai.text import * 
from fastai import *

torch.cuda.set_device(gpu_id)

path = Path(path_to_data)


def istitle(line):
    return len(re.findall(r'^ = [^=]* = $', line)) != 0


def process_unk(s):
    return UNK if s == '<unk>' else s

def read_file(filename):
    articles = []
    with open(filename, encoding='utf8') as f:
        lines = f.readlines()
    current_article = ''
    for i,line in enumerate(lines):
        current_article += line
        if i < len(lines)-2 and lines[i+1] == ' \n' and istitle(lines[i+2]):
            articles.append(current_article)
            current_article = ''
    articles.append(current_article)
    return np.array(articles)


train = read_file(path/'wiki.train.tokens')
valid = read_file(path/'wiki.valid.tokens')
test =  read_file(path/'wiki.test.tokens')


all_texts = np.concatenate([valid, train,test])
df = pd.DataFrame({'texts':all_texts})


del train
del valid
del test


data = (TextList.from_df(df, path, cols='texts')
                .split_by_idx(range(0,60))
                .label_for_lm()
                .databunch())
data.save()


data = TextLMDataBunch.load(path, bs=80)

use_qrnn = False
if architecture == 'qrnn':
    use_qrnn = True

learn = language_model_learner(data, drop_mult=0.5, emb_sz=embedding_size, nh=num_hidden_units, nl=num_layers, qrnn=use_qrnn, clip=0.12)
learn.fit_one_cycle(1,5e-3, moms=(0.8,0.7))

save_name = architecture +  '_' + str(num_layers)

learn.save(save_name)

print('DONE:',num_layers,embedding_size,num_hidden_units,architecture,path_to_data,gpu_id)


