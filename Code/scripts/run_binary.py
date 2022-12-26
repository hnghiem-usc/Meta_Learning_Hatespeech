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
    parser.add_argument('--holdout_sets', type=str, default=[], nargs='+')
    parser.add_argument('--test_sets',type=str, default=[], nargs='+')
    parser.add_argument('--tokenizer_dir', type=str, default="../../Model/RobertaTokenizer/")
    return parser


def compile_arguments(parser):
    args = parser.parse_args()
    
    with open(args.config_file,'r') as f: 
        config = json.load(f)[args.config_key]
    # parse sets into list 
    if args.holdout_sets != []: config['holdout_sets'] = args.holdout_sets
    if args.test_sets != []: config['test_sets'] = args.test_sets
    
    config['config_key'] = args.config_key
    config['statedict_to_save'] = {k: tuple(v) for k, v in config['statedict_to_save'].items()}
    
    # Append the tokenizer directory
    config['tokenizer_args']['tokenizer_dir'] = args.tokenizer_dir
    tokenizer = create_tokenizer(config['tokenizer_args'])

    config['max_len'] =  config['tokenizer_args']['max_len']
    
    del config['tokenizer_args']
    
    return tokenizer,  config


def run_experiment(data_dir, train_filename, holdout_sets, test_sets, seeds, K_values, K_test,\
                    max_len, eval_interval, \
                    tokenize_function, model_args:dict, train_epochs, batch_size, val_args_dict, test_args_dict, \
                    config_key, save_dir, train_num_sample = None, train_val_size=0, statedict_dir='', statedict_to_save=dict()
                  ):
    '''
    @param best_state_idx_to_save: list of tuples (seed, k_val): save the selected best state dict as JSON if the indices match
    '''
    df = pd.read_csv(data_dir + train_filename)
    df.rename(columns={'name':'domain', 'text_std':'text', 'label_bin':'labels'}, inplace=True)
    
    seed_results = {d:dict() for d in test_sets}
    for seed in seeds:
        print("------------------------------ SEED : {} ------------------------------".format(seed))
        print("-----------------------------------------------------------------------")
        print("                             TRAINING                             ")
        print("-----------------------------------------------------------------------")
        # LOAD DATA...df_test is dependent on seeds 
        set_seed(seed)
        select_vars = ['text', 'labels']
        df_train, df_test, val_data =  create_test_sets(df, K_values, holdout_sets=holdout_sets, test_sets=test_sets, \
                                                        select_vars=select_vars, tokenize_function=tokenize_function, \
                                                        seed=seed, k_test=K_test, to_torch=True)
        df_train = df_train.sample(frac=1, random_state=seed)
        if train_num_sample is not None:
            df_train = df_train.sample(train_num_sample)

        
        train = to_torch_Dataset(df_train, select_vars, tokenize_function )
        if train_val_size > 0:
            df_train_val = df_train.sample(train_val_size, random_state=seed)
            train_val = to_torch_Dataset(df_train_val, select_vars, tokenize_function )
            print("Training validation domains:", df_train.domain.unique(), df_train_val.shape)    
        
#         val_data = {k: to_torch_Dataset(v, select_vars, tokenize_function) for k, v in val_data.items()}
        model = create_model(model_args, model_lookup)
        output = binary_train(model=model, epochs=train_epochs, batch_size=batch_size,
                                eval_interval= eval_interval, train_dataset=train, 
                                val_dataset=train_val if train_val_size > 0 else None,
                                test_datasets=val_data, test_args=TrainingArguments(seed=seed, **val_args_dict), seed=seed, 
                                return_best_statedict=True)
#         
        print("-----------------------------------------------------------------------")
        print("                               TESTING                                 ")
        print("-----------------------------------------------------------------------")
        for domain in test_sets:
            print("---------- Testing on Domain {} ----------".format(domain.upper()))
            K_results = dict()
            for K_val in K_values: 
                print("\n.....Training using {} samples and test on {} samples at seed {}".format(K_val, K_test, seed)) 
                test = to_torch_Dataset(df_test[df_test.domain == domain] , select_vars, tokenize_function )

                best_statedict = output[seed]['best_statedict'][domain][K_val]
                # if seed-k_val is in the lookup, save
                if domain in statedict_to_save and statedict_to_save[domain] == (seed, K_val): 
                    best_statedict_name = '_'.join([model_args['model_class'], config_key, domain, str(seed), str(K_val), 'statedict.pth'])
                    torch.save(best_statedict, statedict_dir + best_statedict_name)
                best_model = deepcopy(model)
                best_model.load_state_dict(best_statedict)

                new_seed = seed * 10 
                set_seed(seed*10)
                best_trainer, best_results = quick_eval(best_model, train_set=val_data[domain][K_val], test_set = test,
                                                   training_args=TrainingArguments(seed=seed, **test_args_dict), metric_func=None, 
                                                   require_training=True,
                                                   eval_while_train=False)
                print("\n...Testing accuracy when trained on K_val = {}: {}".format(K_val, round(best_results['accuracy'],3)))
                # collect only train_acc, validation_f1, validation acc
                train_result = {'train_acc': output[seed]['train_acc'], 
                                'val_f1': output[seed]['val_f1'][domain][K_val],
                                'val_acc': output[seed]['val_acc'][domain][K_val]}
                final_output = {'train_result': train_result, 'test_result':best_results}
                # append results for seed    
                K_results[K_val] = final_output

                del best_model, best_trainer, best_results, best_statedict, train_result, final_output
                gc.collect()
                torch.cuda.empty_cache()
            
            # append results for holdoutset
            seed_results[domain][seed] = K_results

        del df_train, df_test, val_data, model
        gc.collect()
        torch.cuda.empty_cache()
        
    result_name = '_'.join([model_args['model_class'], config_key, *test_sets, 'result'])
    save_results(seed_results, result_name , save_dir) 
    
    
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

