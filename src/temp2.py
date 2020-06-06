import src.utils as utils
import torch
from transformers import BertModel, BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
path = '../data/cnndm-pj/deepmind/train_set/'
train = utils.load_pickle(path+'train.pkl')
path2 = '../bert_data/'
train2  = torch.load(path2+'cnndm.test.2.bert.pt')
# print(train2[0].keys())
# print(len(train2[0]['src']))
# print(len(train2[0]['clss']), len(train2[0]['labels']))
# print(train2[0])

for data in train2:
    if 'david' in data['src_txt']:
        print(data['src_txt'])
# print(train2[0])
# print(train[0][0][1])
# print(train2[3])
# data = train2[3]['src_txt']
# print("Segs", len(train2[3]['segs']))
# print("src", len(train2[3]['src']))
# print("Labels", len(train2[9]['labels']))
# total_sentences = []
# counter = 0
# for d in data:
# 	lines = d.split('.')
# 	counter += len(lines)

# print(counter)
#
# granola_ids = tokenizer.encode(train2[3]['src_txt'], return_tensors='pt')
# print(granola_ids)
# print(tokenizer.convert_ids_to_tokens(granola_ids[0]))