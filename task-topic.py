# load startup file
with open("/home/yu/OneDrive/App/Settings/jupyter + R + Python/python_startup.py", 'r') as _:
    exec(_.read())


# working directory
ROOT_DIR = '/home/yu/OneDrive/CC'
DATA_DIR = f'{ROOT_DIR}/data'

print(f'ROOT_DIR: {ROOT_DIR}')
print(f'DATA_DIR: {DATA_DIR}')

import os
os.chdir(ROOT_DIR)

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.matutils import Sparse2Corpus
from gensim.corpora import Dictionary


# -------------- Use ALL text to learn vocabulary ------------------

ld('texttoken', force=True)

vectorizer = CountVectorizer(preprocessor=lambda x: x,
                             tokenizer=lambda x: x,
                             lowercase=False,
                             ngram_range=(1,1),
                             min_df=100,
                             max_df=0.8)

vectorizer.fit([t for t in texttoken['a'].values() if len(t)>0]);
vectorizer.fit([t for t in texttoken['q'].values() if len(t)>0]);
vectorizer.fit([t for t in texttoken['md'].values() if len(t)>0]);

# create idx-word map
id2word = {v:k for k, v in vectorizer.vocabulary_.items()}

print(f'Vocab size: {len(id2word)}')


# ------------------- train model -------------
from gensim.models import LdaModel, LdaMulticore

def train(texttoken):

    # convert to DTM
    dtm = vectorizer.transform(texttoken)
    print(f'N_doc:{dtm.shape[0]}, N_feature:{dtm.shape[1]}')

    # convert to gensim corpus
    corpus = Sparse2Corpus(dtm, documents_columns=False)

    # Train LDA model.
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,

        num_topics=50,
        # workers=8,
        passes=8,
        iterations=500,
        chunksize=30000,

        alpha='auto',
        eta='auto',
        eval_every=2, # slow down the traning, only for debugging,
        per_word_topics=True
    )
    # save model
    
    return model, corpus

# choose model type
model_type = 'md_ngram1'
if not os.path.exists(f'data/ldamodel/{model_type}'):
    os.mkdir(f'data/ldamodel/{model_type}')
print(f'Training: {model_type}')

# start training
model, corpus = train([t for t in texttoken[f'{model_type.split("_")[0]}'].values() if len(t)>0])

# save
model.save(f'data/ldamodel/{model_type}/{model_type}')
sv('corpus', svname=f'corpus_{model_type}', path=f'data/ldamodel/{model_type}')