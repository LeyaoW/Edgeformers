import os
import random
import json
import pickle
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from transformers import BertTokenizerFast

random.seed(0)

dataset = 'Appliances'  # Electronics, movie, CDs, Apps
data_name = 'Appliances_5_train_FT' # reviews_Electronics_5, reviews_CDs_and_Vinyl_5, reviews_Apps_for_Android_5
output_dir='E_data'

# read raw data
data=[]
with open(f'{dataset}/{data_name}.json') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        data.append(json.loads(line))

train_bound=len(data)
print(train_bound)

with open(f'{dataset}/{dataset}_5_val.json') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        data.append(json.loads(line))
        
val_bound=len(data)
print(val_bound)

with open(f'{dataset}/{dataset}_5_test.json') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        data.append(json.loads(line))

print(len(data))
print(data[14])

# text processing function
def text_process(text):
    p_text = ' '.join(text.split('\r\n'))
    p_text = ' '.join(text.split('\n\r'))
    p_text = ' '.join(text.split('\n'))
    p_text = ' '.join(p_text.split('\t'))
    p_text = ' '.join(p_text.split('\rm'))
    p_text = ' '.join(p_text.split('\r'))
    p_text = ''.join(p_text.split('$'))
    p_text = ''.join(p_text.split('*'))

    return p_text


rate_dict = defaultdict(int)
user_id2idx = {}
item_id2idx = {}

for d in tqdm(data):
    rate_dict[d['overall']] += 1
    if d['reviewerID'] not in user_id2idx:
        user_id2idx[d['reviewerID']] = len(user_id2idx)
    if d['asin'] not in item_id2idx:
        item_id2idx[d['asin']] = len(item_id2idx)

print(rate_dict)


samples = []

for d in tqdm(data):
    samples.append((text_process(d['reviewText']), user_id2idx[d['reviewerID']], item_id2idx[d['asin']], d['overall']-1))
    

## split train/val/test as 7:1:2 or 8:1:1
### user_pos_reviews/user_neg_reviews: key<-userID, value<-list(reviews)
### item_pos_reviews/item_neg_reviews: key<-productID, value<-list(reviews)
### train_user_neighbor: key<-userID, value<-list(tuple(reviews,p/n))
### train_item_neighbor: key<-userID, value<-list(tuple(reviews,p/n))

sample_num = len(samples)
random.seed(0)

# train_bound = int(sample_num * 0.7)
# val_bound = int(sample_num * 0.8)
print(train_bound, val_bound - train_bound, sample_num - val_bound)




# generate and save train file

with open(f'{dataset}/{output_dir}/train.tsv','w') as fout:
    for s in tqdm(samples[:train_bound]):
        fout.write(s[0]+'\$\$'+str(s[1])+'\$\$'+str(s[2])+'\$\$'+str(int(s[3]))+'\n')
        
# generate and save val file

with open(f'{dataset}/{output_dir}/val.tsv','w') as fout:
    for s in tqdm(samples[train_bound:val_bound]):
        fout.write(s[0]+'\$\$'+str(s[1])+'\$\$'+str(s[2])+'\$\$'+str(int(s[3]))+'\n')
        
 # generate and save test file

with open(f'{dataset}/{output_dir}/test.tsv','w') as fout:
    for s in tqdm(samples[val_bound:]):
        fout.write(s[0]+'\$\$'+str(s[1])+'\$\$'+str(s[2])+'\$\$'+str(int(s[3]))+'\n')   
        
        
# save side files

pickle.dump(user_id2idx,open(f'{dataset}/{output_dir}/user_id2idx.pkl','wb'))
pickle.dump(item_id2idx,open(f'{dataset}/{output_dir}/item_id2idx.pkl','wb'))
pickle.dump([len(user_id2idx),len(item_id2idx),5],open(f'{dataset}/{output_dir}/node_num.pkl','wb'))
        