import os, sys 
sys.path.append("../utils/")

from learner import *
from loader import * 

import gc 
import torch
import json
import argparse 
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaForSequenceClassification

learner_lookup = {'MAML_Learner': MAML_Learner}

def create_tokenizer(tknz_args:dict):
    if tknz_args['tokenizer_type'] == 'roberta': 
        tokenizer = RobertaTokenizer.from_pretrained(tknz_args['tokenizer_dir'])
    
    return tokenizer


def save_results(result, filename,  save_dir): 
    with open(save_dir + '/' + filename + '.json', 'w') as f :
        json.dump(result, f)
        
        
def argumentParser(): 
    parser = argparse.ArgumentParser()
    
    # Data arguments
    parser.add_argument('--config_file', type=str, default="../../Config/MAML_config.json") 
    parser.add_argument('--config_key', type=str, default='MAML_test')
    parser.add_argument('--holdout_set', type=str, default="olid")
    parser.add_argument('--tokenizer_dir', type=str, default="../../Model/RobertaTokenizer/")
    
    return parser
        
    
def compile_arguments(parser):
    args = parser.parse_args()
    
    with open(args.config_file,'r') as f: 
        config = json.load(f)[args.config_key]
    
    config['holdout_set'] = args.holdout_set
    ## Append the tokenizer directory
    config['tokenizer_args']['tokenizer_dir'] = args.tokenizer_dir
    tokenizer = create_tokenizer(config['tokenizer_args'])
    config['tokenizer'] = tokenizer
    config['max_len'] =  config['tokenizer_args']['max_len']
    
    del config['tokenizer_args']
    
    return config
 
        
def run_experiment(data_dir, train_filename, holdout_set, seeds, K_values, K_test, tokenizer, max_len, base_args, 
                   learner_class, learner_args, compute_metrics, test_args, save_dir):
    
    df = pd.read_csv(data_dir + train_filename)
    df.rename(columns={'name':'domain'}, inplace=True)
    
    seed_results = dict()
    for seed in seeds:
        # LOAD DATA 
        df_train = df[~df.domain.isin([holdout_set])]

        df_test_all = df[df.domain.isin([holdout_set])]
        print("------------------------------ SEED : {} ------------------------------".format(seed))
        K_results = dict()
        for K_val in K_values: 
            print("---------------TRAINING FOR  K_val = {} ---------------".format(K_val)) 
            set_seed(seed)

            df_val = df_test_all.sample(K_val)
            df_test = df_test_all.loc[~df_test_all.id.isin(df_val.id)].sample(K_test)

            df_test = pd.concat([df_val, df_test])
            df_val = pd.concat([df_val, df_val])     

#             del df_test_all
            print("Training domains:", df_train.domain.unique())
            print("Validation domains:", df_val.domain.unique(), df_val.shape)
            print("Test domains:", df_test.domain.unique(), df_test.shape)

            # Specify training arguments !!! COMPARE TO MAKE SURE ARGUMENTS ARE CONSISTENT
            args = TrainingArgs(base_args)
            learner = learner_lookup[learner_class](args) 
            print("--------------- META TRAINING ---------------")
            meta_output = meta_train(meta_learner=learner, meta_args=args, max_len=max_len,
                    df_train=df_train, df_test=df_val, tokenizer=tokenizer, 
                    test_k_support=K_val, test_k_query=K_val, val_metric_function=compute_metrics,
                    seed=seed,
                    **learner_args
                   )
            
            # Testing
            print("--------------- TESTING ---------------")
            best_statedict = meta_output[seed]['best_statedict']
            best_model = deepcopy(learner.model)
            best_model.load_state_dict(best_statedict)
            
            set_seed(seed*10)
            test_loader = MetaLoader(df_test, num_task = 1,
                                  k_support=K_val, k_query=K_test, max_len=max_len,
                                  tokenizer = tokenizer, batch_mode='fixed', verbose=True)
            train_ds, test_ds = create_meta_batches(test_loader, 1, split_to_train_test=True)
            testing_args = TrainingArguments(learning_rate=args.inner_update_lr,
                                          seed=seed,
                                          **test_args)
            
            best_trainer, best_results = quick_eval(best_model, train_set=train_ds, test_set = test_ds,
                                           training_args=testing_args, metric_func=None, 
                                           require_training=True,
                                           eval_while_train=False)
            
            final_output = {'train_result': meta_output[seed]['report'], 'test_result':best_results}

            del learner, best_statedict, best_model
            del test_loader, train_ds, test_ds, testing_args, best_trainer, best_results
            gc.collect()
            torch.cuda.empty_cache()

            # append results for seed    
            K_results[K_val] = final_output
            
        # append results for holdoutset
        seed_results[seed] = K_results

    save_results(seed_results, holdout_set + '_results' , save_dir)

    
def main(): 
    parser = argumentParser()
    config_dict = compile_arguments(parser)
    run_experiment(compute_metrics=compute_metrics, **config_dict )
    
if __name__=="__main__":
    main()
    print('END OF CODE')