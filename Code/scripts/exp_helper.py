# Helper functions for run_[LEARNER].py scripts 
import json
from transformers import RobertaTokenizer, BertTokenizer 
from multi_learner import MAML_Unicorn
from special_learner import Special_Learner
from ds_learner_orth import DS_Learner_Orth
from ds_learner_cov import DS_Learner_Cov
from joint_learner_cov import Joint_Learner_Cov

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
    learner_lookup = {'MAML_Unicorn': MAML_Unicorn, 'Special_Learner':Special_Learner,
                      'DS_Learner_Orth': DS_Learner_Orth, 'DS_Learner_Cov': DS_Learner_Cov, 
#                       'Joint_Learner_Orth': Joint_Learner_Orth,
                      'Joint_Learner_Cov': Joint_Learner_Cov}
    return learner_lookup