from utils import load_json_data
import os
import json
import pickle
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from transformers import BertTokenizerFast
from utils import *
import secrets


secrets.SystemRandom().seed(0)

dataset="Appliances"

# output_dir="processed_N" #llm
# output_dir="finetuned_N"
# output_dir="original_N"
# output_dir="knn_N"
output_dir="llm_N" #llm


train_data = []
val_data = []
test_data = []

train_data=load_json_data(train_data,"Appliances_5_train_llm",dataset)
# traindata=load_json_data(train_data,"Appliances_5_train_FT",dataset)
# traindata=load_json_data(train_data,"Appliances_5_train_knn",dataset)
# train_data=load_json_data(train_data,"Appliances_5_train_dropout_0.1",dataset)

val_data=load_json_data(val_data,"Appliances_5_val",dataset)

test_data=load_json_data(test_data,"Appliances_5_test",dataset)




rate_dict = defaultdict(int)

data=train_data +val_data +test_data 
for d in tqdm(data):
    rate_dict[d['overall']] += 1
    
print(rate_dict)


## user/item statistics
### we see 5 score as positive edge(review), 1-4 as negative ones.
### user_pos_reviews/user_neg_reviews: key<-userID, value<-list(reviews)
### item_pos_reviews/item_neg_reviews: key<-productID, value<-list(reviews)
### user_reviews_dict/item_reviews_dict: key<-userID/productID, value<-list(tuple(reviews,p/n))

train_user_pos_reviews, train_user_neg_reviews,  train_item_pos_reviews, train_item_neg_reviews, train_user_set, train_item_set, train_user_reviews_dict, train_item_reviews_dict = process_data_N(train_data,rate_dict)
_, _,  _, _, _, _, val_user_reviews_dict, _ = process_data_N(val_data,rate_dict)
_, _,  _, _, _, _, test_user_reviews_dict, _ = process_data_N(test_data,rate_dict)



# split train/val/test as 7:1:2 or 8:1:1
## user_pos_reviews/user_neg_reviews: key<-userID, value<-list(reviews)
## item_pos_reviews/item_neg_reviews: key<-productID, value<-list(reviews)
## train_user_neighbor: key<-userID, value<-list(tuple(reviews,p/n))
## train_item_neighbor: key<-userID, value<-list(tuple(reviews,p/n))


train_item_set, train_tuples, user_id2idx, item_id2idx,train_user_pos_neighbor , train_user_neg_neighbor, train_item_pos_neighbor , train_item_neg_neighbor= get_train_tuples(train_user_reviews_dict)

val_tuples,user_id2idx=get_val_test_tuples(val_user_reviews_dict,user_id2idx)

test_tuples,user_id2idx=get_val_test_tuples(test_user_reviews_dict,user_id2idx)
        
print(f'Number of item appearing in train_set:{len(train_item_set)} or {len(item_id2idx)}') # Number of item appearing in train_set:47 or 47
print(f'Train/Val/Test size:{len(train_tuples)},{len(val_tuples)},{len(test_tuples)}') # Train/Val/Test size:1593,229,455




# generate and save train file
## user pos neighbor: 8/2, user neg neighbor: 8/2
## item pos neighbor: 10/5, item neg neighbor: 10/5

upos = 3
uneg = 3
ipos = 5
ineg = 5

secrets.SystemRandom().seed(0)
with open(f'{dataset}/{output_dir}/train.tsv','w+') as fout:
    for d in tqdm(train_tuples):
        
        # prepare sample pool for user and item
        user_pos_pool, user_neg_pool, item_pos_pool, item_neg_pool=get_pool(d, "train", train_user_pos_neighbor,train_user_neg_neighbor,train_item_pos_neighbor,train_item_neg_neighbor)
        save_tsv_N(fout,upos, uneg,ipos, ineg, user_pos_pool, user_neg_pool, item_pos_pool, item_neg_pool,item_id2idx,user_id2idx,d )



secrets.SystemRandom().seed(0)
valid_dev_edges = 0
with open(f'{dataset}/{output_dir}/val.tsv','w+') as fout:
    for d in tqdm(val_tuples):
        # if item not in train item set, continue
        if d[1][1] not in train_item_set:
            continue

        # counting
        valid_dev_edges += 1
        
        user_pos_pool, user_neg_pool, item_pos_pool, item_neg_pool=get_pool(d, "val", train_user_pos_neighbor,train_user_neg_neighbor,train_item_pos_neighbor,train_item_neg_neighbor)
        save_tsv_N(fout,upos, uneg,ipos, ineg, user_pos_pool, user_neg_pool, item_pos_pool, item_neg_pool, item_id2idx, user_id2idx,d)
        
print(f'Number of Valid Dev Edges:{valid_dev_edges} | Total:{len(val_tuples)}')
    
    
    
    
secrets.SystemRandom().seed(0)
valid_test_edges = 0
with open(f'{dataset}/{output_dir}/test.tsv','w+') as fout:
    for d in tqdm(test_tuples):
        # if item not in train item set, continue
        if d[1][1] not in train_item_set:
            continue

        # counting
        valid_test_edges += 1
        
        user_pos_pool, user_neg_pool, item_pos_pool, item_neg_pool=get_pool(d, "test", train_user_pos_neighbor,train_user_neg_neighbor,train_item_pos_neighbor,train_item_neg_neighbor)
        save_tsv_N(fout,upos, uneg,ipos, ineg, user_pos_pool, user_neg_pool, item_pos_pool, item_neg_pool, item_id2idx, user_id2idx,d )
        
        
print(f'Number of Valid Test Edges:{valid_test_edges} | Total:{len(test_tuples)}')


# save side files
pickle.dump([upos,uneg,ipos,ineg],open(f'{dataset}/{output_dir}/neighbor_sampling.pkl','wb+'))
pickle.dump(user_id2idx,open(f'{dataset}/{output_dir}/user_id2idx.pkl','wb+'))
pickle.dump(item_id2idx,open(f'{dataset}/{output_dir}/item_id2idx.pkl','wb+'))
pickle.dump([len(user_id2idx),len(item_id2idx),2],open(f'{dataset}/{output_dir}/node_num.pkl','wb+'))


# save neighbor file
pickle.dump(train_user_pos_neighbor,open(f'{dataset}/{output_dir}/neighbor/train_user_pos_neighbor.pkl','wb+'))
pickle.dump(train_user_neg_neighbor,open(f'{dataset}/{output_dir}/neighbor/train_user_neg_neighbor.pkl','wb+'))
pickle.dump(train_item_pos_neighbor,open(f'{dataset}/{output_dir}/neighbor/train_item_pos_neighbor.pkl','wb+'))
pickle.dump(train_item_neg_neighbor,open(f'{dataset}/{output_dir}/neighbor/train_item_neg_neighbor.pkl','wb+'))
