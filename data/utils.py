import json
from tqdm import tqdm
from collections import defaultdict
from copy import deepcopy
import torch
import os
import secrets

def seed_everything(seed=0):
    secrets.SystemRandom().seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_json_data(data, file,dataset):
    with open(f'{dataset}/{file}.json') as f:
        readin = f.readlines()
        for line in tqdm(readin):
            data.append(json.loads(line))
    return data 


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




def process_data_N(data,rate_dict):
    secrets.SystemRandom().seed(0)
    user_pos_reviews = defaultdict(list)
    user_neg_reviews = defaultdict(list)
    item_pos_reviews = defaultdict(list)
    item_neg_reviews = defaultdict(list)
    user_set = set()
    item_set = set()

    user_reviews_dict = defaultdict(list)
    item_reviews_dict = defaultdict(list)

    blank_review_cnt = 0

    for d in tqdm(data):
        if 'reviewText' not in d:
        #if 'summary' not in d:
            blank_review_cnt += 1
            continue
        
        text = text_process(d['reviewText'])
        #text = text_process(d['summary'])
        user_set.add(d['reviewerID'])
        item_set.add(d['asin'])
        if d['overall'] == 5.0:
            user_pos_reviews[d['reviewerID']].append(text)
            item_pos_reviews[d['asin']].append(text)
            
            user_reviews_dict[d['reviewerID']].append((text,d['asin'],1))
            item_reviews_dict[d['asin']].append((text,d['reviewerID'],1))
        elif d['overall'] in [1,2,3,4]:
            user_neg_reviews[d['reviewerID']].append(text)
            item_neg_reviews[d['asin']].append(text)
            
            user_reviews_dict[d['reviewerID']].append((text,d['asin'],0))
            item_reviews_dict[d['asin']].append((text,d['reviewerID'],0))
        else:
            raise ValueError('Error!')
    
    
    print(f'Number of blank review:{blank_review_cnt}')
    print(f'Number of user:{len(user_set)}, Number of item:{len(item_set)}')
    print(f'user_pos_reviews.len:{len(user_pos_reviews)},user_neg_reviews.len:{len(user_neg_reviews)}')
    print(f'item_pos_reviews.len:{len(item_pos_reviews)},item_neg_reviews.len:{len(item_neg_reviews)}')
    print(f'user.avg.pos_review:{rate_dict[5]/len(user_set)},user.avg.neg_review:{(rate_dict[1]+rate_dict[2]+rate_dict[3]+rate_dict[4])/len(user_set)}')
    print(f'item.avg.pos_review:{rate_dict[5]/len(item_set)},item.avg.neg_review:{(rate_dict[1]+rate_dict[2]+rate_dict[3]+rate_dict[4])/len(item_set)}')
    
    return user_pos_reviews, user_neg_reviews,  item_pos_reviews, item_neg_reviews, user_set,item_set, user_reviews_dict, item_reviews_dict


def get_train_tuples(user_reviews_dict):
    secrets.SystemRandom().seed(0)
    train_item_set = set()
    train_tuples = []
    user_id2idx = {}
    item_id2idx = {}
    train_user_pos_neighbor = defaultdict(list)
    train_user_neg_neighbor = defaultdict(list)
    train_item_pos_neighbor = defaultdict(list)
    train_item_neg_neighbor = defaultdict(list)
    for uid in tqdm(user_reviews_dict):
        if uid not in user_id2idx:
            user_id2idx[uid] = len(user_id2idx)
        secrets.SystemRandom().shuffle(user_reviews_dict[uid])
            

        for i in range(len(user_reviews_dict[uid])):
            train_tuples.append((uid,user_reviews_dict[uid][i]))
            train_item_set.add(user_reviews_dict[uid][i][1])
            
            # add to item_id2idx
            if user_reviews_dict[uid][i][1] not in item_id2idx:
                item_id2idx[user_reviews_dict[uid][i][1]] = len(item_id2idx)

            #add to train_user_neighbor/train_item_neighbor
            if user_reviews_dict[uid][i][2] == 1:
                train_user_pos_neighbor[uid].append(user_reviews_dict[uid][i])
                train_item_pos_neighbor[user_reviews_dict[uid][i][1]].append((user_reviews_dict[uid][i][0],uid,user_reviews_dict[uid][i][2]))
            elif user_reviews_dict[uid][i][2] == 0:
                train_user_neg_neighbor[uid].append(user_reviews_dict[uid][i])
                train_item_neg_neighbor[user_reviews_dict[uid][i][1]].append((user_reviews_dict[uid][i][0],uid,user_reviews_dict[uid][i][2]))
            else:
                raise ValueError('Error!')
    return  train_item_set, train_tuples, user_id2idx, item_id2idx,train_user_pos_neighbor , train_user_neg_neighbor, train_item_pos_neighbor , train_item_neg_neighbor


def get_val_test_tuples(user_reviews_dict,user_id2idx):
    val_test_tuples = []
    for uid in tqdm(user_reviews_dict):
        if uid not in user_id2idx:
            user_id2idx[uid] = len(user_id2idx)
        secrets.SystemRandom().shuffle(user_reviews_dict[uid])
        for i in range(len(user_reviews_dict[uid])):
            # print(i,len(user_reviews_dict[uid]))
            val_test_tuples.append((uid,user_reviews_dict[uid][i]))
    return val_test_tuples,user_id2idx



def get_pool(d,mode, train_user_pos_neighbor,train_user_neg_neighbor,train_item_pos_neighbor,train_item_neg_neighbor):
    
    # prepare sample pool for user and item
    user_pos_pool = set(deepcopy(train_user_pos_neighbor[d[0]]))
    user_neg_pool = set(deepcopy(train_user_neg_neighbor[d[0]]))
    item_pos_pool = set(deepcopy(train_item_pos_neighbor[d[1][1]]))
    item_neg_pool = set(deepcopy(train_item_neg_neighbor[d[1][1]]))
    
    if mode=="train":
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
    
    return user_pos_pool, user_neg_pool, item_pos_pool, item_neg_pool




def save_tsv_N(fout,upos, uneg,ipos, ineg, user_pos_pool, user_neg_pool, item_pos_pool, item_neg_pool,item_id2idx,user_id2idx,d):

        
    secrets.SystemRandom().seed(0)
    

    secrets.SystemRandom().shuffle(user_pos_pool)
    secrets.SystemRandom().shuffle(user_neg_pool)
    secrets.SystemRandom().shuffle(item_pos_pool)
    secrets.SystemRandom().shuffle(item_neg_pool)
    
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
    
    
    
def form_prompt(label,dataset):
    dataset_name=f"Amazon {dataset}"
    text=f"{dataset} review"

    prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for generating {text}s from {dataset_name}. <|eot_id|>
<|start_header_id|>user<|end_header_id|>

Give me one {text} for {dataset_name} with a rating of {label} stars. Wrap up your answer with <START> in the beginning and <END> at the end. <|eot_id|> 
<|start_header_id|>assistant<|end_header_id|>
'''
    return prompt
