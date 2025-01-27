import json
from tqdm import tqdm
from collections import defaultdict
import random
from copy import deepcopy
import torch
import os
from parser import args

def seed_everything(seed=0):
    random.seed(seed)
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




def generate_text(prompt,args):

    inputs = args.tokenizer(prompt, return_tensors="pt").to(args.device)
    output = args.model.generate(inputs['input_ids'], max_new_tokens=250, num_return_sequences=1)

    generated_text = args.tokenizer.decode(output[0], skip_special_tokens=True)
    #print(generated_text)
    answer=generated_text.split("<START>")[-1]
    answer=answer.split('<END>')[0]
    tem=answer.split("\n")
    answer=" ".join(tem)
    # print(answer)
    return answer