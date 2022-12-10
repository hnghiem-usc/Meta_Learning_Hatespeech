import os, sys 
sys.path.append("../utils/")

from learner import *
from loader import * 

import gc 
import torch
import json
import argparse 
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaForSequenceClassification
from math import ceil
from sklearn.model_selection import train_test_split

model_lookup = {'roberta-base': RobertaForSequenceClassification}

def create_tokenizer(tknz_args:dict):
    if tknz_args['tokenizer_type'] == 'roberta': 
        tokenizer = RobertaTokenizer.from_pretrained(tknz_args['tokenizer_dir'])
    
    return tokenizer


def create_model(model_args:dict, model_lookup:dict):
    model_dir = model_args['model_dir']
    num_labels = model_args['num_labels']
    model_class = model_lookup[model_args['model_class']]
    
    return model_class.from_pretrained(model_dir, num_labels=num_labels)


def save_results(result, filename,  save_dir): 
    with open(save_dir + '/' + filename + '.json', 'w') as f :
        json.dump(result, f)
        
        
def argumentParser(): 
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--config_file', type=str, default="../../Config/binary_config.json") 
    parser.add_argument('--config_key', type=str, default='binary_test')
    parser.add_argument('--holdout_set', type=str, default="olid")
    parser.add_argument('--tokenizer_dir', type=str, default="../../Model/RobertaTokenizer/")
    
    return parser


def compile_arguments(parser):
    args = parser.parse_args()
    
    with open(args.config_file,'r') as f: 
        config = json.load(f)[args.config_key]
    
    config['holdout_set'] = args.holdout_set
    config['config_key'] = args.config_key
    ## Append the tokenizer directory
    config['tokenizer_args']['tokenizer_dir'] = args.tokenizer_dir
    tokenizer = create_tokenizer(config['tokenizer_args'])

    config['max_len'] =  config['tokenizer_args']['max_len']
    
    del config['tokenizer_args']
    
    return tokenizer, config


def run_experiment(data_dir, train_filename, holdout_set, seeds, K_values, K_test, max_len, eval_interval, \
                  tokenize_function, model_args:dict, train_epochs, batch_size, val_args_dict, test_args_dict, \
                  config_key, save_dir, train_num_sample = None, train_val_size=0):
    
    df = pd.read_csv(data_dir + train_filename)
    df.rename(columns={'name':'domain', 'text_std':'text', 'label_bin':'labels'}, inplace=True)
    
    for seed in seeds:
        # LOAD DATA 
        df_train = df[~df.domain.isin([holdout_set])].sample(frac=1, random_state=seed)
        if train_num_sample is not None:
            df_train = df_train.sample(train_num_sample)
       
        select_vars = ['text', 'labels']
        train = to_torch_Dataset(df_train, select_vars, tokenize_function )
        if train_val_size > 0:
            df_train_val = df_train.sample(train_val_size, random_state=seed)
            train_val = to_torch_Dataset(df_train_val, select_vars, tokenize_function )
            
        seed_results = dict()

        df_test_all = df[df.domain.isin([holdout_set])]
        print("------------------------------ SEED : {} ------------------------------".format(seed))
        K_results = dict()
        
        for K_val in K_values: 
            print("---------------TRAINING FOR  K_val: {} ---------------".format(K_val)) 
            set_seed(seed)

            df_val = df_test_all.sample(K_val) 
            df_test = df_test_all.loc[~df_test_all.id.isin(df_val.id)].sample(K_test)

            print("Training domains:", df_train.domain.unique(), df_train.shape)
            if train_val_size > 0:
                print("Training validation domains:", df_train.domain.unique(), df_train_val.shape)
            print("Validation domains:", df_val.domain.unique(), df_val.shape)
            print("Test domains:", df_test.domain.unique(), df_test.shape)

            # Process to tensor dataset 
            val = to_torch_Dataset(df_val, select_vars, tokenize_function )
            test = to_torch_Dataset(df_test, select_vars, tokenize_function )

            # valdiate only on limited dataset
            model = create_model(model_args, model_lookup)
            output = binary_train(model=model, epochs=train_epochs, batch_size=batch_size,
                                eval_interval= eval_interval, train_dataset=train, 
                                val_dataset=train_val if train_val_size > 0 else None,
                                test_dataset=val, test_args=TrainingArguments(seed=seed, **val_args_dict), seed=seed, 
                                return_best_statedict=True)


            # Test on TRUE test set
            print("--------------- TESTING ---------------")
            best_statedict = output[seed]['best_statedict']
            best_model = deepcopy(model)
            best_model.load_state_dict(best_statedict)

            set_seed(seed*10)
            best_trainer, best_results = quick_eval(best_model, train_set=val, test_set = test,
                                               training_args=TrainingArguments(seed=seed, **test_args_dict), metric_func=None, 
                                               require_training=True,
                                               eval_while_train=False)

            final_output = {'train_result': output[seed]['report'], 'test_result':best_results}

            del model, output, best_model, best_statedict, best_trainer, best_results
            gc.collect()
            torch.cuda.empty_cache()

            # append results for seed    
            K_results[K_val] = final_output

        # append results for holdoutset
        seed_results[seed] = K_results
        
    save_results(seed_results, holdout_set + '_' +  config_key + '_results' , save_dir)    
    
    
def main(): 
    parser = argumentParser()
    tokenizer, config_dict = compile_arguments(parser)
    
    def tokenize_function(dt, max_len=config_dict['max_len'], var='text'): 
        return tokenizer(dt[var], max_length=max_len, padding='max_length', truncation=True, return_tensors='pt')
    run_experiment(tokenize_function=tokenize_function, **config_dict )

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print("END OF CODE, Duration: {}".format(end - start))

