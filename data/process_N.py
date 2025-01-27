import os
import random
import json
import pickle
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

import numpy as np
from transformers import BertTokenizerFast
from parser import args

random.seed(0)


dataset = f'{args.dataset}'  

if args.mode=="MISS":
    data_name = f'{args.data_name}_train_rr_{args.review_rate}'
elif args.mode=="ORIG":
    data_name = f'{args.data_name}_train'
else :
    data_name = f'{args.data_name}_train_rr_{args.review_rate}_{args.mode}'

output_dir=f'/{args.dataset}_rr_{args.review_rate}/{args.mode}'



if not os.path.exists(f'{dataset}/{output_dir}'):
    os.makedirs(f'{dataset}/{output_dir}')

train_data = []
val_data = []
test_data = []

with open(f'{dataset}/{args.data_name}.json') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        train_data.append(json.loads(line))

print(len(train_data))

with open(f'{dataset}/{args.data_name}_val.json') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        val_data.append(json.loads(line))
        


with open(f'{dataset}/{args.data_name}_test.json') as f:
    readin = f.readlines()
    for line in tqdm(readin):
        test_data.append(json.loads(line))

data=train_data +val_data +test_data 
print(len(data))

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
    
    ## rate distribution

rate_dict = defaultdict(int)

for d in tqdm(data):
    rate_dict[d['overall']] += 1
    
print(rate_dict)







user_set = set()
item_set = set()


train_user_pos_reviews = defaultdict(list)
train_user_neg_reviews = defaultdict(list)
train_item_pos_reviews = defaultdict(list)
train_item_neg_reviews = defaultdict(list)
train_user_reviews_dict = defaultdict(list)
train_item_reviews_dict = defaultdict(list)

blank_review_cnt = 0




for d in tqdm(train_data):
    if 'reviewText' not in d:
        blank_review_cnt += 1
        continue
    
    text = text_process(d['reviewText'])
    #text = text_process(d['summary'])
    user_set.add(d['reviewerID'])
    item_set.add(d['asin'])
    
    if d['overall'] == 5.0:
        train_user_pos_reviews[d['reviewerID']].append(text)
        train_item_pos_reviews[d['asin']].append(text)
        
        train_user_reviews_dict[d['reviewerID']].append((text,d['asin'],1))
        train_item_reviews_dict[d['asin']].append((text,d['reviewerID'],1))
    elif d['overall'] in [1,2,3,4]:
        train_user_neg_reviews[d['reviewerID']].append(text)
        train_item_neg_reviews[d['asin']].append(text)
        
        train_user_reviews_dict[d['reviewerID']].append((text,d['asin'],0))
        train_item_reviews_dict[d['asin']].append((text,d['reviewerID'],0))
    else:
        raise ValueError('Error!')




test_user_pos_reviews = defaultdict(list)
test_user_neg_reviews = defaultdict(list)
test_item_pos_reviews = defaultdict(list)
test_item_neg_reviews = defaultdict(list)
test_user_reviews_dict = defaultdict(list)
test_item_reviews_dict = defaultdict(list)


for d in tqdm(test_data):
    if 'reviewText' not in d:
        blank_review_cnt += 1
        continue
    
    text = text_process(d['reviewText'])
    #text = text_process(d['summary'])
    user_set.add(d['reviewerID'])
    item_set.add(d['asin'])
    
    if d['overall'] == 5.0:
        test_user_pos_reviews[d['reviewerID']].append(text)
        test_item_pos_reviews[d['asin']].append(text)
        
        test_user_reviews_dict[d['reviewerID']].append((text,d['asin'],1))
        test_item_reviews_dict[d['asin']].append((text,d['reviewerID'],1))
    elif d['overall'] in [1,2,3,4]:
        test_user_neg_reviews[d['reviewerID']].append(text)
        test_item_neg_reviews[d['asin']].append(text)
        
        test_user_reviews_dict[d['reviewerID']].append((text,d['asin'],0))
        test_item_reviews_dict[d['asin']].append((text,d['reviewerID'],0))
    else:
        raise ValueError('Error!')



val_user_pos_reviews = defaultdict(list)
val_user_neg_reviews = defaultdict(list)
val_item_pos_reviews = defaultdict(list)
val_item_neg_reviews = defaultdict(list)
val_user_reviews_dict = defaultdict(list)
val_item_reviews_dict = defaultdict(list)




for d in tqdm(val_data):
    if 'reviewText' not in d:
        blank_review_cnt += 1
        continue
    
    text = text_process(d['reviewText'])
    #text = text_process(d['summary'])
    user_set.add(d['reviewerID'])
    item_set.add(d['asin'])
    
    if d['overall'] == 5.0:
        val_user_pos_reviews[d['reviewerID']].append(text)
        val_item_pos_reviews[d['asin']].append(text)
        
        val_user_reviews_dict[d['reviewerID']].append((text,d['asin'],1))
        val_item_reviews_dict[d['asin']].append((text,d['reviewerID'],1))
    elif d['overall'] in [1,2,3,4]:
        val_user_neg_reviews[d['reviewerID']].append(text)
        val_item_neg_reviews[d['asin']].append(text)
        
        val_user_reviews_dict[d['reviewerID']].append((text,d['asin'],0))
        val_item_reviews_dict[d['asin']].append((text,d['reviewerID'],0))
    else:
        raise ValueError('Error!')







random.seed(0)

train_tuples = []
val_tuples = []
test_tuples = []
train_item_set = set()
user_id2idx = {}
item_id2idx = {}
train_user_pos_neighbor = defaultdict(list)
train_user_neg_neighbor = defaultdict(list)
train_item_pos_neighbor = defaultdict(list)
train_item_neg_neighbor = defaultdict(list)

c1 = 0
c2 = 0
c3 = 0

for uid in tqdm(train_user_reviews_dict):
    if uid not in user_id2idx:
        user_id2idx[uid] = len(user_id2idx)
    random.shuffle(train_user_reviews_dict[uid])
    
    for i in range(int(len(train_user_reviews_dict[uid]))):
    #for i in range(int(len(user_reviews_dict[uid])*0.8)):
        train_tuples.append((uid,train_user_reviews_dict[uid][i]))
        train_item_set.add(train_user_reviews_dict[uid][i][1])
        
        # add to item_id2idx
        if train_user_reviews_dict[uid][i][1] not in item_id2idx:
            item_id2idx[train_user_reviews_dict[uid][i][1]] = len(item_id2idx)

        # add to train_user_neighbor/train_item_neighbor
        if train_user_reviews_dict[uid][i][2] == 1:
            train_user_pos_neighbor[uid].append(train_user_reviews_dict[uid][i])
            train_item_pos_neighbor[train_user_reviews_dict[uid][i][1]].append((train_user_reviews_dict[uid][i][0],uid,train_user_reviews_dict[uid][i][2]))
        elif train_user_reviews_dict[uid][i][2] == 0:
            train_user_neg_neighbor[uid].append(train_user_reviews_dict[uid][i])
            train_item_neg_neighbor[train_user_reviews_dict[uid][i][1]].append((train_user_reviews_dict[uid][i][0],uid,train_user_reviews_dict[uid][i][2]))
        else:
            raise ValueError('Error!')

for uid in tqdm(val_user_reviews_dict):
    if uid not in user_id2idx:
        user_id2idx[uid] = len(user_id2idx)
    random.shuffle(val_user_reviews_dict[uid])  
           
    for i in range(int(len(val_user_reviews_dict[uid]))):
        val_tuples.append((uid,val_user_reviews_dict[uid][i]))

for uid in tqdm(test_user_reviews_dict):
    if uid not in user_id2idx:
        user_id2idx[uid] = len(user_id2idx)
    random.shuffle(test_user_reviews_dict[uid])  
    for i in range(int(len(test_user_reviews_dict[uid]))):
    #for i in range(int(len(user_reviews_dict[uid])*0.9),len(user_reviews_dict[uid])):
        test_tuples.append((uid,test_user_reviews_dict[uid][i]))
        
print(f'Number of item appearing in train_set:{len(train_item_set)} or {len(item_id2idx)}')
print(f'Train/Val/Test size:{len(train_tuples)},{len(val_tuples)},{len(test_tuples)}')


# generate and save train file
## user pos neighbor: 8/2, user neg neighbor: 8/2
## item pos neighbor: 10/5, item neg neighbor: 10/5

upos = 3
uneg = 3
ipos = 5
ineg = 5

random.seed(0)

with open(f'{dataset}/{output_dir}/train.tsv','w+') as fout:
    for d in tqdm(train_tuples):
        
        # prepare sample pool for user and item
        user_pos_pool = set(deepcopy(train_user_pos_neighbor[d[0]]))
        user_neg_pool = set(deepcopy(train_user_neg_neighbor[d[0]]))
        item_pos_pool = set(deepcopy(train_item_pos_neighbor[d[1][1]]))
        item_neg_pool = set(deepcopy(train_item_neg_neighbor[d[1][1]]))
        
        if d[1][2] == 1:
            user_pos_pool.remove(d[1])
            item_pos_pool.remove((d[1][0],d[0],d[1][2]))
        elif d[1][2] == 0:
            user_neg_pool.remove(d[1])
            item_neg_pool.remove((d[1][0],d[0],d[1][2]))
        else:
            raise ValueError('Error!')
        
        user_pos_pool = list(user_pos_pool)
        item_pos_pool = list(item_pos_pool)
        user_neg_pool = list(user_neg_pool)
        item_neg_pool = list(item_neg_pool)
        random.shuffle(user_pos_pool)
        random.shuffle(user_neg_pool)
        random.shuffle(item_pos_pool)
        random.shuffle(item_neg_pool)
        
        # sample for user
        if len(user_pos_pool) >= upos:
            user_pos_samples = user_pos_pool[:upos]
        else:
            user_pos_samples = user_pos_pool + [('',-1)] * (upos-len(user_pos_pool))
        
        if len(user_neg_pool) >= uneg:
            user_neg_samples = user_neg_pool[:uneg]
        else:
            user_neg_samples = user_neg_pool + [('',-1)] * (uneg-len(user_neg_pool))
        
        # sample for item
        if len(item_pos_pool) >= ipos:
            item_pos_samples = item_pos_pool[:ipos]
        else:
            item_pos_samples = item_pos_pool + [('',-1)] * (ipos-len(item_pos_pool))
        
        if len(item_neg_pool) >= ineg:
            item_neg_samples = item_neg_pool[:ineg]
        else:
            item_neg_samples = item_neg_pool + [('',-1)] * (ineg-len(item_neg_pool))
        
        # prepare for writing file
        user_pos_text = '\t'.join([up[0] for up in user_pos_samples])
        user_pos_neighbor = '\t'.join([str(item_id2idx[up[1]]) if up[1] != -1 else str(-1) for up in user_pos_samples])
        user_neg_text = '\t'.join([un[0] for un in user_neg_samples])
        user_neg_neighbor = '\t'.join([str(item_id2idx[un[1]]) if un[1] != -1 else str(-1) for un in user_neg_samples])
        
        item_pos_text = '\t'.join([ip[0] for ip in item_pos_samples])
        item_pos_neighbor = '\t'.join([str(user_id2idx[ip[1]]) if ip[1] != -1 else str(-1) for ip in item_pos_samples])
        item_neg_text = '\t'.join([inn[0] for inn in item_neg_samples])
        item_neg_neighbor = '\t'.join([str(user_id2idx[inn[1]]) if inn[1] != -1 else str(-1) for inn in item_neg_samples])
        
        user_line = str(user_id2idx[d[0]]) + '\*\*' + user_pos_text + '\*\*' + user_neg_text + '\*\*' + user_pos_neighbor + '\*\*' + user_neg_neighbor
        item_line = str(item_id2idx[d[1][1]]) + '\*\*' + item_pos_text + '\*\*' + item_neg_text + '\*\*' + item_pos_neighbor + '\*\*' + item_neg_neighbor
        
        fout.write(user_line+'\$\$'+item_line+'\$\$'+str(d[1][2])+'\n')
        
        
# generate and save val file (make sure to delete items that are not in train set)

random.seed(0)

valid_dev_edges = 0

with open(f'{dataset}/{output_dir}/val.tsv','w+') as fout:
    for d in tqdm(val_tuples):
        # if item not in train item set, continue
        if d[1][1] not in train_item_set:
            continue

        # counting
        valid_dev_edges += 1

        # prepare sample pool for user and item
        user_pos_pool = deepcopy(train_user_pos_neighbor[d[0]])
        user_neg_pool = deepcopy(train_user_neg_neighbor[d[0]])
        item_pos_pool = deepcopy(train_item_pos_neighbor[d[1][1]])
        item_neg_pool = deepcopy(train_item_neg_neighbor[d[1][1]])
        
        random.shuffle(user_pos_pool)
        random.shuffle(user_neg_pool)
        random.shuffle(item_pos_pool)
        random.shuffle(item_neg_pool)
        
        # sample for user
        if len(user_pos_pool) >= upos:
            user_pos_samples = user_pos_pool[:upos]
        else:
            user_pos_samples = user_pos_pool + [('',-1)] * (upos-len(user_pos_pool))
        
        if len(user_neg_pool) >= uneg:
            user_neg_samples = user_neg_pool[:uneg]
        else:
            user_neg_samples = user_neg_pool + [('',-1)] * (uneg-len(user_neg_pool))
        
        # sample for item
        if len(item_pos_pool) >= ipos:
            item_pos_samples = item_pos_pool[:ipos]
        else:
            item_pos_samples = item_pos_pool + [('',-1)] * (ipos-len(item_pos_pool))
        
        if len(item_neg_pool) >= ineg:
            item_neg_samples = item_neg_pool[:ineg]
        else:
            item_neg_samples = item_neg_pool + [('',-1)] * (ineg-len(item_neg_pool))
        
        # prepare for writing file
        user_pos_text = '\t'.join([up[0] for up in user_pos_samples])
        user_pos_neighbor = '\t'.join([str(item_id2idx[up[1]]) if up[1] != -1 else str(-1) for up in user_pos_samples])
        user_neg_text = '\t'.join([un[0] for un in user_neg_samples])
        user_neg_neighbor = '\t'.join([str(item_id2idx[un[1]]) if un[1] != -1 else str(-1) for un in user_neg_samples])
        
        item_pos_text = '\t'.join([ip[0] for ip in item_pos_samples])
        item_pos_neighbor = '\t'.join([str(user_id2idx[ip[1]]) if ip[1] != -1 else str(-1) for ip in item_pos_samples])
        item_neg_text = '\t'.join([inn[0] for inn in item_neg_samples])
        item_neg_neighbor = '\t'.join([str(user_id2idx[inn[1]]) if inn[1] != -1 else str(-1) for inn in item_neg_samples])
        
        user_line = str(user_id2idx[d[0]]) + '\*\*' + user_pos_text + '\*\*' + user_neg_text + '\*\*' + user_pos_neighbor + '\*\*' + user_neg_neighbor
        item_line = str(item_id2idx[d[1][1]]) + '\*\*' + item_pos_text + '\*\*' + item_neg_text + '\*\*' + item_pos_neighbor + '\*\*' + item_neg_neighbor
        
        fout.write(user_line+'\$\$'+item_line+'\$\$'+str(d[1][2])+'\n')

print(f'Number of Valid Dev Edges:{valid_dev_edges} | Total:{len(val_tuples)}')






# generate and save test file (make sure to delete items that are not in train set)

random.seed(0)

valid_test_edges = 0

with open(f'{dataset}/{output_dir}/test.tsv','w+') as fout:
    for d in tqdm(test_tuples):
        # if item not in train item set, continue
        if d[1][1] not in train_item_set:
            continue

        # counting
        valid_test_edges += 1

        # prepare sample pool for user and item
        user_pos_pool = deepcopy(train_user_pos_neighbor[d[0]])
        user_neg_pool = deepcopy(train_user_neg_neighbor[d[0]])
        item_pos_pool = deepcopy(train_item_pos_neighbor[d[1][1]])
        item_neg_pool = deepcopy(train_item_neg_neighbor[d[1][1]])
        
        random.shuffle(user_pos_pool)
        random.shuffle(user_neg_pool)
        random.shuffle(item_pos_pool)
        random.shuffle(item_neg_pool)
        
        # sample for user
        if len(user_pos_pool) >= upos:
            user_pos_samples = user_pos_pool[:upos]
        else:
            user_pos_samples = user_pos_pool + [('',-1)] * (upos-len(user_pos_pool))
        
        if len(user_neg_pool) >= uneg:
            user_neg_samples = user_neg_pool[:uneg]
        else:
            user_neg_samples = user_neg_pool + [('',-1)] * (uneg-len(user_neg_pool))
        
        # sample for item
        if len(item_pos_pool) >= ipos:
            item_pos_samples = item_pos_pool[:ipos]
        else:
            item_pos_samples = item_pos_pool + [('',-1)] * (ipos-len(item_pos_pool))
        
        if len(item_neg_pool) >= ineg:
            item_neg_samples = item_neg_pool[:ineg]
        else:
            item_neg_samples = item_neg_pool + [('',-1)] * (ineg-len(item_neg_pool))
        
        # prepare for writing file
        user_pos_text = '\t'.join([up[0] for up in user_pos_samples])
        user_pos_neighbor = '\t'.join([str(item_id2idx[up[1]]) if up[1] != -1 else str(-1) for up in user_pos_samples])
        user_neg_text = '\t'.join([un[0] for un in user_neg_samples])
        user_neg_neighbor = '\t'.join([str(item_id2idx[un[1]]) if un[1] != -1 else str(-1) for un in user_neg_samples])
        
        item_pos_text = '\t'.join([ip[0] for ip in item_pos_samples])
        item_pos_neighbor = '\t'.join([str(user_id2idx[ip[1]]) if ip[1] != -1 else str(-1) for ip in item_pos_samples])
        item_neg_text = '\t'.join([inn[0] for inn in item_neg_samples])
        item_neg_neighbor = '\t'.join([str(user_id2idx[inn[1]]) if inn[1] != -1 else str(-1) for inn in item_neg_samples])
        
        user_line = str(user_id2idx[d[0]]) + '\*\*' + user_pos_text + '\*\*' + user_neg_text + '\*\*' + user_pos_neighbor + '\*\*' + user_neg_neighbor
        item_line = str(item_id2idx[d[1][1]]) + '\*\*' + item_pos_text + '\*\*' + item_neg_text + '\*\*' + item_pos_neighbor + '\*\*' + item_neg_neighbor
        
        fout.write(user_line+'\$\$'+item_line+'\$\$'+str(d[1][2])+'\n')

print(f'Number of Valid Test Edges:{valid_test_edges} | Total:{len(test_tuples)}')



# save side files


pickle.dump([upos,uneg,ipos,ineg],open(f'{dataset}/{output_dir}/neighbor_sampling.pkl','wb+'))
pickle.dump(user_id2idx,open(f'{dataset}/{output_dir}/user_id2idx.pkl','wb+'))
pickle.dump(item_id2idx,open(f'{dataset}/{output_dir}/item_id2idx.pkl','wb+'))
pickle.dump([len(user_id2idx),len(item_id2idx),2],open(f'{dataset}/{output_dir}/node_num.pkl','wb+'))

# save neighbor file
if not os.path.exists(f'{dataset}/{output_dir}/neighbor'):
    os.makedirs(f'{dataset}/{output_dir}/neighbor')
if not os.path.exists(f'{dataset}/{output_dir}/ckpt'):
    os.makedirs(f'{dataset}/{output_dir}/ckpt')

pickle.dump(train_user_pos_neighbor,open(f'{dataset}/{output_dir}/neighbor/train_user_pos_neighbor.pkl','wb+'))
pickle.dump(train_user_neg_neighbor,open(f'{dataset}/{output_dir}/neighbor/train_user_neg_neighbor.pkl','wb+'))
pickle.dump(train_item_pos_neighbor,open(f'{dataset}/{output_dir}/neighbor/train_item_pos_neighbor.pkl','wb+'))
pickle.dump(train_item_neg_neighbor,open(f'{dataset}/{output_dir}/neighbor/train_item_neg_neighbor.pkl','wb+'))