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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from utils import load_json_data,  generate_text
# from transformers import pipeline 

random.seed(0)



# args.dataset="Appliances"

data_name = f'{args.dataset}_rr_{args.review_rate}/{args.data_name}_train' 


args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if args.fine_tune=="False":
    model_name="meta-llama/Llama-3.2-3B-Instruct"
    args.saved_name="Appliances_5_train.json"
    
else:
    model_name=f"{args.dataset}_{data_name}_fine_tuning/{data_name}_FT"
    args.saved_name="Appliances_5_train_FT.json"

args.tokenizer = AutoTokenizer.from_pretrained(model_name, load_in_8bit=False,torch_dtype=torch.float16)
args.model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=False,torch_dtype=torch.float16, force_download=True ).bfloat16()

args.tokenizer.pad_token_id=args.tokenizer.eos_token_id
args.model.to(args.device)
print("Loaded the LLM model:",model_name)


data = []

data=load_json_data(data,f"{data_name}_miss",args.dataset)


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

def form_prompt_with_summary(label, summary, dataset):
    dataset_name=f"Amazon {dataset}"
    text=f"{dataset} review"

    prompt = f'''<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful AI assistant for generating {text}s from {dataset_name}. <|eot_id|>
<|start_header_id|>user<|end_header_id|>

Summerzied user preference: {summary}<|eot_id|> 
<|start_header_id|>user<|end_header_id|>

Based on summerzied user preference, give me one {text} for {dataset_name} with a rating of {label} stars. Wrap up your answer with <START> in the beginning and <END> at the end. <|eot_id|> 

<|start_header_id|>assistant<|end_header_id|>
'''
    return prompt



def impute(data):
    cnt=0
    for i in range(len(data)):
        
        if data[i]['reviewText']=="":
            label=data[i]['overall']
            cnt=cnt+1
            prompt=form_prompt(label,args.dataset)
            review=generate_text(prompt,args)
            data[i]['reviewText']=review
    print("Added reviews: ",cnt) #159
    return data


def impute_with_summary(data):
    with open(f'{args.dataset}/train_user_summary_dict.pkl', 'rb') as f:
        user_summary_dict = pickle.load(f)
        
    cnt=0
    for i in range(len(data)):
        
        if data[i]['reviewText']=="":
            label=data[i]['overall']
            user_id=data[i]["reviewerID"]
            summary=user_summary_dict[uid]
            prompt=form_prompt_with_summary(label,summary,args.dataset)
            review=generate_text(prompt,args)
            data[i]['reviewText']=review
            
            cnt=cnt+1
    print("Added reviews: ",cnt) #159
    return data




def save_data(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')





if args.mode=="LLM":
    impute(data)
    save_data(data, f"{args.dataset}/{data_name}_rr_{args.review_rate}_{args.mode}.json")
# else:
#     impute_with_summary(data)
#     save_data(data, f"{args.dataset}/{data_name}_summary.json")







