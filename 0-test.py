import datatable as dt

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datatable import f
from tqdm.auto import tqdm

dt.init_styles()

import spacy
from spacy.attrs import ORTH
from spacy.tokens import Doc, Span

# --------------------  Register Extension -----------------

# Use the simpliest pipeline
nlp = spacy.load("en_core_web_sm", disable=['tok2vec', 'ner', 'parser', 'morphologizer', 'tagger', 'lemmatizer', 'attribute_ruler', 'transformer'])

# Add a simple sentencizer
nlp.add_pipe('sentencizer')

# add [EOC] as special case in tokenization
special_case = [{ORTH: "[EOC]"}]
nlp.tokenizer.add_special_case("[EOC]", special_case)

# register extension for Span
Span.set_extension('transcriptid', default=None, force=True)
Span.set_extension('componentid', default=None, force=True)
Span.set_extension('componentorder', default=None, force=True)
Span.set_extension('componenttypeid', default=None, force=True)
Span.set_extension('speakerid', default=None, force=True)
Span.set_extension('speakertypeid', default=None, force=True)
Span.set_extension('is_component', default=False, force=True)


# --------------------  Load Data -----------------

ld('text_component_sp500', ldname='text_component', force=True)

# conver to tuples
text_component = dt.Frame(text_component)

text_component = text_component[:1000,:].to_tuples()
text_component = [(line[6], 
                   {'transcriptid': line[2],
                    'componentid': line[0],
                    'componenttypeid': line[4],
                    'componentorder': line[3],
                    'speakerid': int(line[5]) if line[5]!=None else None,
                    'speakertypeid': int(line[1]) if line[1]!=None else None
                   }) for line in text_component]

text_component_grouped = {}
for text, context in text_component:
    tid = context['transcriptid']
    if tid in text_component_grouped:
        text_component_grouped[tid].append((text, context))
    else:
        text_component_grouped[tid] = [(text, context)]



# --------------------  Build Doc -----------------
# Final output holder
# docs = []

# Iterate through every transcriptid
# for line in tqdm(text_component_grouped.values(), total=len(text_component_grouped)):

def parse_one_doc(line):
    # '''
    # line: A list of tuples, each tuple contains one component
    # '''

    # Output holder
    components = []

    # Within every transcriptid, iterature through every component
    for component, context in nlp.pipe(line, as_tuples=True):
        
        # Assign component-level attributes
        component[:]._.is_component = True
        component[:]._.transcriptid = context['transcriptid']
        component[:]._.componentid = context['componentid']
        component[:]._.componenttypeid = context['componenttypeid']
        component[:]._.componentorder = context['componentorder']
        component[:]._.speakerid = context['speakerid']
        component[:]._.speakertypeid = context['speakertypeid']

        # Assign sentence-level attributes
        for sent in component.sents:
            sent._.componentid = context['componentid']

        # return
        components.append(component)

    # join components into one Doc
    doc = Doc.from_docs(components)

    # create SpanGroup "components" for doc
    spans_component = []
    for k, v in doc.user_data.items():
        if k[1]=='is_component':
            if v==True:
                spans_component.append(doc.char_span(k[2], k[3]))

    doc.spans['components'] = spans_component 

    assert type(doc)==Doc   

    # docs.append(doc)
    return doc

with ProcessPoolExecutor(max_workers=1) as executer:
    docs = executer.map(parse_one_doc, text_component_grouped.values())

print(type(docs))
print(list(docs))