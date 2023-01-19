import pickle

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def convert_text_to_tokens(text_a,max_seq_length=100,tokenizer=tokenizer):
    # print(text_a)
    if len(text_a)>100:
        text_a = text_a[:100]
    tokens_a = tokenizer.tokenize(text_a)
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(1)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(0)
    # print(tokens)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    # print(input_ids)
#     tokens_tensor = torch.tensor([indexed_tokens])
    input_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
    return torch.tensor([input_ids]),torch.tensor([input_mask]),torch.tensor([segment_ids])

def convert_examples_to_features(text_a, max_seq_length=100, tokenizer=tokenizer):
    input_ids = []
    for i in range(len(text_a)):
        input_id,input_mask,segment_id = convert_text_to_tokens(text_a[i],max_seq_length=max_seq_length,tokenizer=tokenizer)
        input_ids.append(input_id)
    return torch.cat(input_ids, dim=0)


def tuple_to_tensor(t1):
    for i in range(len(t1)):
        t1[i] = list(map(float, t1[i]))

def load_pickle(file_name):
	f = open(file_name, "rb+")
	data = pickle.load(f)
	f.close()
	return data