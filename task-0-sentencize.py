import pandas as pd
import spacy
from spacy.lang.en import English
from tqdm import tqdm

ROOT_DIR = 'C:/Users/rossz/OneDrive/CC'
DATA_DIR = f'{ROOT_DIR}/data'

targets_df = pd.read_feather(f'{DATA_DIR}/f_sue_keydevid_car_finratio_transcriptid_text.feather') 

# spacy model
nlp = English()  
sentencizer = nlp.create_pipe("sentencizer")
nlp.add_pipe(sentencizer)

def sentencize(df:str, sentencizer, text_field):
    transcriptids = df['transcriptid']
    texts = df[text_field]
    
    res = []
    texts_sentencized = nlp.pipe(texts, batch_size=1000, n_process=1)
    for tid, text in tqdm(zip(transcriptids, texts_sentencized), total=len(texts)):
        for sid, sent in enumerate(text.sents):
            res.append((tid, sid, sent.text))
        
    return res

# text_all_sentencized = sentencize(targets_df, nlp, 'text_all')
# pd.DataFrame(text_all_sentencized, columns=['transcriptid', 'sentenceid', 'text']).to_feather(f'{DATA_DIR}/text_all_sentencized.feather')


# text_present_sentencized = sentencize(targets_df, nlp, 'text_present')
# pd.DataFrame(text_present_sentencized, columns=['transcriptid', 'sentenceid', 'text']).to_feather(f'{DATA_DIR}/text_present_sentencized.feather')

text_qa_sentencized = sentencize(targets_df, nlp, 'text_qa')
pd.DataFrame(text_qa_sentencized, columns=['transcriptid', 'sentenceid', 'text']).to_feather(f'{DATA_DIR}/text_qa_sentencized.feather')