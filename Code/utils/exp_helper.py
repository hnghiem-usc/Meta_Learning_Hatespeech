import json
from transformers import RobertaTokenizer, BertTokenizer 
from multi_learner import MAML_Unicorn

def create_tokenizer(tknz_args:dict):
    if tknz_args['tokenizer_type'] == 'roberta': 
        tokenizer = RobertaTokenizer.from_pretrained(tknz_args['tokenizer_dir'])
    elif tknz_args['tokenizer_type'] == 'bert': 
        tokenizer = BertTokenizer.from_pretrained(tknz_args['tokenizer_dir'])
    
    return tokenizer


def save_results(result, filename,  save_dir): 
    with open(save_dir + '/' + filename + '.json', 'w') as f :
        json.dump(result, f)
    
    
def get_learner_lookup():
    learner_lookup = {'MAML_Unicorn': MAML_Unicorn}
    return learner_lookup