from transformers import BertModel, BertTokenizer, BertTokenizerFast

ROOT_DIR = 'C:/Users/rossz/Onedrive/CC'
DATA_DIR = f'{ROOT_DIR}/data'

tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained(f'{DATA_DIR}/finBERT', return_dict=True)
model.to('cuda');

tokens = ['okay okay '*1000]*12
tokens = tokenizer(tokens, return_tensors='pt', padding=True, truncation=True, max_length=512).to('cuda')
tokens.input_ids.shape

y = model(**tokens)
print(y)