import os, sys, json
from copy import deepcopy
import pandas as pd, numpy as np

def flatten_results(res_dict:dict):
    '''
        Pull perf. metrics to the same level as keys in a dict
    '''
    out = dict()
    out['accuracy'] = res_dict['accuracy']
    for k, v in res_dict['macro avg'].items():
        out[k] = v
    return out


def distill_results(res_dict:dict):
    '''
        Distill complete results dict to select only accuracy and macroaverage metrics
    '''
    seed_results = dict()
    for seed, results in res_dict.items(): 
        k_dict = dict()
        for k, values in results.items(): 
            temp_values = deepcopy(values)
            del temp_values['train_result']

            temp_results = {k:v for k, v in temp_values['test_result'].items() if k in( 'accuracy', 'macro avg') }

            k_dict[k] = flatten_results(temp_results)

        seed_results[seed] = k_dict
    return seed_results 


def averge_results(res_dict:dict):
    '''
        Compute elememtwise averages and STD 
        @param res_dict: output of distill_results()
    '''
    res_list = list()
    columns = pd.DataFrame(res_dict[list(res_dict.keys())[0]]).columns
    index   = pd.DataFrame(res_dict[list(res_dict.keys())[0]]).index

    # stack results across seeds
    for _, res in res_dict.items():
        res_list.append(pd.DataFrame.from_dict(res))
    
    stacked_np = np.array([x.values for x in res_list])
    mean = pd.DataFrame(np.mean(stacked_np, axis=0), columns=columns, index=index)
    std  = pd.DataFrame(np.std(stacked_np, axis=0), columns=columns, index=index)
    return mean, std