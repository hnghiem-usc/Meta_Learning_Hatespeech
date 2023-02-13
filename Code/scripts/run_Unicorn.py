import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import os, sys 
sys.path.append("../utils/")
from multi_loader import * 
from multi_learner import *
from exp_helper import * 

import json, argparse

def argumentParser(): 
    parser = argparse.ArgumentParser()
    # Data arguments
    parser.add_argument('--config_file', type=str, default="../../Config/Unicorn_config.json") 
    parser.add_argument('--config_key', type=str, default='test')
    parser.add_argument('--holdout_sets', type=str, default=[], nargs='+')
    parser.add_argument('--test_sets',type=str, default=[], nargs='+')
    parser.add_argument('--tokenizer_dir', type=str, default="../../Model/RobertaTokenizer/")
    parser.add_argument('--select_vars', type=str, default=['domain', 'text', 'label_orig', 'label_target'], nargs='+')
    parser.add_argument('--label_vars', type=str, default=['label_orig', 'label_target'], nargs='+')
    parser.add_argument('--num_classifier', type=int, default=0)
    
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
    
    if args.num_classifier > 0:
        config['learner_args']['num_classifier'] = args.num_classifier
    config['label_vars'] = args.label_vars
    
    # Append the tokenizer directory
    config['tokenizer_args']['tokenizer_dir'] = args.tokenizer_dir
    tokenizer = create_tokenizer(config['tokenizer_args'])

    config['max_len'] =  config['tokenizer_args']['max_len']
    
    del config['tokenizer_args']
    
    return tokenizer,  config


# learner args: arguments to set up Meta Learner
def run_experiment(data_dir, train_filename, config_filename, holdout_sets, test_sets, seeds, K_values, K_test,
                   tokenizer, max_len, learner_class, learner_args, meta_train_args, save_dir, 
                   config_key, statedict_dir='', statedict_to_save=dict(), meta_test_args = dict(), 
                   select_vars = ['domain', 'text', 'label_orig', 'label_target'], 
                   label_vars = ['label_orig', 'label_target']
                  ):
    '''
    @param best_state_idx_to_save: list of tuples (seed, k_val): save the selected best state dict as JSON if the indices match
    '''
    df = pd.read_csv(data_dir + train_filename)
    df.rename(columns={'name':'domain'}, inplace=True)
    label_config = json.load(open(data_dir + config_filename, 'r'))
    print("Label Vars:", label_vars)
    
    seed_results = {d:dict() for d in test_sets}
    for seed in seeds:
        print("------------------------------ SEED : {} ------------------------------".format(seed))
        print("-----------------------------------------------------------------------")
        print("                             META TRAINING                             ")
        print("-----------------------------------------------------------------------")
        # LOAD DATA...df_test is dependent on seeds 
        set_seed(seed)
        df_train, df_test, val_data = create_test_sets(df, K_values,  holdout_sets=holdout_sets, test_sets=test_sets ,\
                                                       select_vars=select_vars, k_test=K_test, seed=seed)
#         return df_train
        args = TrainingArgs(**learner_args)
        learner = learner_lookup[learner_class](args)
        # METATRAIN and META VALIDATE that works throughout all K's
        meta_output = multi_meta_train(meta_learner=learner, meta_args=args, 
                            df_train=df_train, tokenizer=tokenizer, max_len = max_len, label_config=label_config,
                            test_datasets=val_data, seed=seed, skip_validation=False, return_best_statedict=True, 
                            label_vars=label_vars,**meta_train_args
                       )
        
        print("-----------------------------------------------------------------------")
        print("                               TESTING                                 ")
        print("-----------------------------------------------------------------------")
        for domain in test_sets: 
            print("\n---------- Testing on Domain {} ----------".format(domain.upper()))
            K_results = dict()
            for K_val in K_values: 
                print("\n.....Training using {} samples and test on {} samples at seed {}".format(K_val, K_test, seed)) 
                best_statedict = meta_output[seed]['best_statedict'][domain][K_val]
                # if seed-k_val is in the lookup, save
                if domain in statedict_to_save and statedict_to_save[domain] == (seed, K_val): 
                    best_statedict_name = '_'.join([learner_class, config_key, domain, str(seed), str(K_val), 'statedict.pth'])
                    print("Best statedict name:{}".format(best_statedict_name)) #D
                    torch.save(best_statedict, statedict_dir + best_statedict_name)
                
                # create temp model with correct architecture with best weights 
                best_model = deepcopy(learner)
                best_model.load_state_dict(best_statedict)
                
                new_seed = seed * 10
                set_seed(new_seed)
                # Train on limited K_val and test on actual K_test 
                # Note: val_data at this step is already doubly stacked, so take half only
                df_val_test = pd.concat([val_data[domain][K_val][:K_val], df_test])
                print("df_val_test_shape:", df_val_test.shape, df_val_test.columns)
                test_metaloader = MetaLoader(df_val_test, num_task = 1, k_support=K_val, k_query=K_test,
                                             max_len=max_len, tokenizer = tokenizer, label_config=label_config,   
                                             batch_mode = 'fixed', label_vars=label_vars, test_domains=None)
                _, best_results = meta_quick_eval(best_model, test_metaloader, **meta_test_args, include_scheduler=True)
                
                print("Testing result when trained on K_val = {}".format(K_val))
                for task, result in best_results.items():
                    print("\t{} \tAccuracy:{} \tMacro-F1:{} ".format(task, round(result['accuracy'],3),
                                                                   round(result['macro avg']['f1-score'],5)))
                         
                
                # collect only train_acc, validation_f1, validation acc
                train_result = {'train_acc': meta_output[seed]['train_acc'], 
                                'val_f1': meta_output[seed]['val_f1'][domain][K_val],
                                'val_acc': meta_output[seed]['val_acc'][domain][K_val]}
                final_output = {'train_result': train_result, 'test_result':best_results}
                
                # append results for seed    
                K_results[K_val] = final_output

                del df_val_test, best_statedict, best_model, test_metaloader, best_results
                cleanup()
                
            # append results for holdoutset
            seed_results[domain][seed] = K_results
            
        # clean up aftet colelcting results
        del df_train, df_test, val_data, args, learner, meta_output
        cleanup()
        
    result_name = '_'.join([learner_class, config_key, *test_sets, 'result'])
    print(result_name)
    # return seed_results 
    save_results(seed_results, result_name , save_dir)
       
        
learner_lookup = get_learner_lookup()
def main(): 
    parser = argumentParser()
    tokenizer, config_dict = compile_arguments(parser)
    run_experiment(tokenizer=tokenizer, **config_dict)
    
if __name__=="__main__":
    start = time.time()
    main()
    end = time.time()
    print('END OF CODE, Duration: {}'.format(end - start))