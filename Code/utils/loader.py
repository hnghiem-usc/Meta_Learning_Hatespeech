# UDATED 12/31/2022: MODIFIED LOADER TO BE COMPATIBLE WITH MULTI-TASK (LABEL_ORIG, LABEL_TARGET) AND MULTI-LABEL SETTING

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json
from random import shuffle
import pandas as pd, numpy as np

import os
import torch
from copy import deepcopy
from torch.utils.data import Dataset
import collections
import random
import json, pickle
import gc

from torch.utils.data import TensorDataset
from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertForSequenceClassification
from transformers import TrainingArguments, Trainer

from datasets import load_dataset, load_metric, Dataset as hg_Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, f1_score
from math import ceil



import time
import logging


############################## META LOADER ##############################
def convert_label_format(x: list):
    '''
        Convert list of labels from str into tensor-compatible format 
        
        Parameters:
        x: list of string labels 
        
        Return: 
        list of int/floats
    '''
    data_type = set([type(d) for d in x])
    if data_type.issubset(set([int, float])):
        return list(x)
    elif np.char.str_len(x).max() == 1: 
        labels = [int(l) for l in x]
        return labels 
    else: 
        samples = [l[1:-1].split(',') for l in x]
        labels = [[int(l) for l in label] for label in samples]
        return labels


# Craete a custom dataset loader that returns a batch of [support-query] set
# Requires parameters to control the data in each batch: 
# number of tasks to sample, size of query, size of support, and argumentns for tokenzier
class MetaLoader(Dataset): 
    def __init__(self, df: pd.DataFrame, num_task, k_support, k_query,tokenizer, label_config:dict, max_len=128, truncation=True,
                 domain_var='domain', text_var='text_std', label_vars = 'label_bin', test_domains=[], to_dataset = False, 
                 batch_mode = 'random', label_dict={'positive':1, 'negative':0}, verbose=False,seed=123):
        
        self.data = dict()
        for d in df[domain_var].unique():
            self.data[d]  = df.loc[df[domain_var] == d, :].sample(frac=1).to_dict(orient='records')
        self.domain_list = list(self.data.keys())
        
        self.num_task  = num_task
        self.k_support = k_support
        self.k_query   = k_query 
        self.label_config = label_config
        self.label_dict = label_dict
        self.domain_var = domain_var 
        self.test_domains = test_domains
        self.text_var = text_var
        self.label_vars = [label_vars] if type(label_vars) is str else label_vars
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.truncation = truncation
        self.verbose = verbose
        self.batch_mode = batch_mode
        self.to_dataset = to_dataset

        print("Tokenizer Config:\nMax Lenght: {}, Truncation: {}".format(max_len, truncation))
        
        if self.batch_mode == 'fixed': 
            self.create_fixed_batch()
        elif self.batch_mode == 'disjoint':
            self.create_disjoint_batch()
        elif self.batch_mode == 'joint': 
            self.create_joint_batch()
        else:
            self.create_random_batch()
    
    
    def __reset_batch__(self):
        self.support = [] 
        self.query = []
 

    def create_random_batch(self):
        """
        Arrange data into random query and support test. 
        NOTE: support and query may not belong to the same domain
        param num_task: int, number of meta-samples per batch
        param test_domains: list, if not empty, the domains of which samples are chosen for train and test are mutually exclusive
        """
        self.__reset_batch__()
        if self.verbose: 
            print("Creating RANDOM batch...")
        if self.test_domains:
            train_domains = [d for d in self.domain_list if d not in self.test_domains]
        
        # random.seed(seed)
        for i in range(self.num_task):
            if self.test_domains:
                sample_train = [random.choice(self.data[random.choice(train_domains)]) for i in range(self.k_support)]
                sample_test  = [random.choice(self.data[random.choice(self.test_domains)]) for i in range(self.k_query)]
            else: 
                sample_train = [random.choice(self.data[random.choice(self.domain_list)]) for i in range(self.k_support)]
                sample_test  = [random.choice(self.data[random.choice(self.domain_list)]) for i in range(self.k_query)]
            self.support.append(sample_train)     
            self.query.append(sample_test)
    
    
    def create_joint_batch(self): 
        '''
        Arrange data such that support and query batches are from the same domain.
        NOTE: sample may still overlap between train and test set
        '''
        self.__reset_batch__()
        if self.verbose: 
            print("Creating JOINT batch...")
        if self.test_domains:
            train_domains = [d for d in self.domain_list if d not in self.test_domains]
            
        for i in range(self.num_task):
            domain = random.choice(self.domain_list)
            sample_train = [random.choice(self.data[domain]) for i in range(self.k_support)]
            sample_test  = [random.choice(self.data[domain]) for i in range(self.k_query)]
            
            self.support.append(sample_train)     
            self.query.append(sample_test)
            
            
    def create_disjoint_batch(self):
        """
        Arrange data into mutually exclusive query and support batch
        """
        if self.verbose: 
            print("Creating DISJOINT batch...")
        self.__reset_batch__()
        # random.seed(seed)
        for i in range(self.num_task):
            test_domain =  random.choice(self.domain_list)
            train_domains = tuple(set(self.domain_list) - set([test_domain]))
            
            sample_train = [random.choice(self.data[random.choice(train_domains)]) for i in range(self.k_support)]
            sample_test  = [random.choice(self.data[test_domain]) for i in range(self.k_query)] 
            self.support.append(sample_train)     
            self.query.append(sample_test)
    
    
    def create_fixed_batch(self):
        '''
        Select data sequentially from the main set
        '''
        if self.verbose: 
            print("Creating FIXED batch...")
        self.__reset_batch__()
        # assume that there is only 1 domain 
        sample_train = self.data[self.domain_list[0]][:self.k_support]
        sample_test  = self.data[self.domain_list[0]][self.k_support:self.k_support + self.k_query]
        self.support.append(sample_train)     
        self.query.append(sample_test)
        
        
    def create_feature_set(self, samples, text_var, label_vars:list):
        """ 
        Tokenize the samples in each batch of support/query into tensordataset. 
        Note: the order of output is hardcoded below 
        """
        text_ls = []
        domains = [] 
        for sample in samples:
            text_ls.append(sample[text_var]) 
            domains.append(sample['domain'])
        # fill in blanks if empty    
        if len(text_ls) == 0:
            text_ls = [' ']
            label_ls = [0]
        # tokenize
        output = self.tokenizer(text_ls, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=self.truncation)        
        output = collections.OrderedDict(output)
        
#         process labels 
        problem_config = collections.OrderedDict()
        domain = samples[0]['domain'] 
        label_array = []
        for idx, label_var in enumerate(label_vars):
            label_ls = [l[label_var] for l in samples if l[label_var] == l[label_var]]
            if len(label_ls) > 0: 
                label_ls = convert_label_format(label_ls)
                label_ls = torch.tensor(label_ls)
                label_array.append(label_ls)
                output[label_var] = label_ls
            if (self.batch_mode == 'joint' or  self.batch_mode == 'fixed') and  label_var in self.label_config[sample['domain']]: 
                problem_config[label_var]  = self.label_config[sample['domain']][label_var]
                domains = list(set(domains))
                
        if self.to_dataset:
            return hg_Dataset.from_dict(output).with_format('torch'), problem_config, domains
        
        return TensorDataset(output['input_ids'], output[ 'attention_mask'], *label_array), problem_config, domains
        
    def __getitem__(self, index):
        support_set, _, _ = self.create_feature_set(self.support[index],self.text_var, self.label_vars)
        query_set, problem_config, domains = self.create_feature_set(self.query[index], self.text_var, self.label_vars)
        return support_set, query_set, problem_config, domains
    
    
    def __len__(self):
        return self.num_task
    
    
############################## DATA HELPER FUNCTIONS ##############################

def set_seed(seed): 
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    
def statedict_to_cpu(state_dict):
    for k, v in state_dict.items():
          state_dict[k] = v.cpu()
         
    
def create_batch_of_tasks(taskset, is_shuffle = True, batch_size = 4):
    idxs = list(range(0,len(taskset)))
    if is_shuffle:
        random.shuffle(idxs)
    for i in range(0,len(idxs), batch_size):
        yield [taskset[idxs[i]] for i in range(i, min(i + batch_size,len(taskset)))]
     
    
def create_meta_batches(loader:MetaLoader, batch_size: int, is_shuffle=False, split_to_train_test=False): 
    batches = create_batch_of_tasks(loader, is_shuffle=is_shuffle, batch_size=batch_size)
    if split_to_train_test: 
        batch = next(batches)
        collated_train = collate_to_Dataset(batch[0][0])
        collated_test  = collate_to_Dataset(batch[0][1])
        return collated_train, collated_test
    return batches


def create_test_sets(df:pd.DataFrame, k_vals:list, holdout_sets:list, test_sets:list, seed:int, \
                     select_vars:list, tokenize_function=None, k_test:int=0, to_torch=False):
    '''
    Divide general dataset to return all necessary datasets for meta training
    @param select_vars, tokenize_function: arguments to use for the to_torch_dataset function
    '''
    holdout_sets += test_sets 
    holdout_sets = list(set(holdout_sets))
    if not test_sets or test_sets is None or test_sets == ['']:
        print("WARNING: test sets cannot be empty, automatically default to holdout_sets")
        test_sets = holdout_sets
    test_sets = list(set(test_sets))
    
    for s in holdout_sets:
        if s not in df.domain.unique() : print("Domain {} NOT in available domains.CHECK!".format(s))
    
    df_train = df[~df.domain.isin(holdout_sets)]
    df_test_all = df[df.domain.isin(test_sets)]
    
    val_data = dict()
    if k_test == 0:
        df_test = df_test_all.copy() 
        return df_train, df_test, val_data
    else:
        df_test = pd.concat([df_test_all[df_test_all.domain == domain].sample(k_test, random_state=seed) for domain in test_sets])
        
    print("Training Domains:", df_train.domain.unique(), df_train.shape)
    print("Holdout Domains:", holdout_sets)
    print("Testing Domains:", df_test.domain.unique(), df_test.shape)
    
    # Save the validation data sets for each domain and k_val
    for s in test_sets:
        k_dict = dict()
        for k in k_vals: 
            # enforce disjointness between train and test set 
            df_val = df_test_all.loc[~df_test_all.id.isin(df_test.id) & (df_test_all.domain == s)].sample(k, random_state=seed)
            df_val = pd.concat([df_val, df_val]) # duplicate for meta training  
            k_dict[k] = to_torch_Dataset(df_val, select_vars, tokenize_function) if to_torch else df_val 
        val_data[s] = k_dict
    
    return df_train, df_test, val_data
        
        
class TrainingArgs(object):
    '''
        Specifying the key parameters for metalearner
    '''
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
            

# metric = load_metric('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    results = classification_report(y_true=labels, y_pred=predictions,
                                        output_dict=True,
                                        zero_division=0)
    test_f1 = round(results['macro avg']['f1-score'],3)
    test_acc = round(results['accuracy'],3)
    return {'eval_f1':test_f1, 'eval_acc':test_acc}
#     return metric.compute(predictions=predictions, references=labels)


def collate_to_Dataset(tensorset, position_dict={'input_ids':0, 'attention_mask':1, 'labels':2}):
    batch = dict()
    batch['input_ids'] = torch.stack([i[position_dict['input_ids']] for i in tensorset])
    batch['attention_mask'] = torch.stack([i[position_dict['attention_mask']] for i in tensorset])
    batch['labels'] = torch.stack([i[position_dict['labels']] for i in tensorset])

    return hg_Dataset.from_dict(batch)


def to_torch_Dataset(df:pd.DataFrame, select_vars:list, tokenize_function): 
    ds = hg_Dataset.from_pandas(df[select_vars])
    ds = ds.map(tokenize_function, batched=True, remove_columns=['text','__index_level_0__'])
    ds = ds.with_format('torch')
    return ds

############################## TRAINING HELPER FUNCTIONS ##############################
def move_to_device(model:dict, device):
    for key, component in model.items():
        model[key] = component.to(device)
    return model


def quick_eval(model, train_set, test_set, training_args, label_var='labels', require_training=True,
               eval_while_train = False,
               return_result=True, metric_func=None):
    logging.disable(logging.INFO)
    args = deepcopy(training_args)
    results = dict()    
    labels = test_set[label_var]

    if require_training:
        if eval_while_train: 
            trainer = Trainer(model=model, 
                        train_dataset=train_set,
                        eval_dataset=test_set,
                        args=args,
                        compute_metrics=metric_func)
        else:    
            args.evaluation_strategy='no'       
            trainer = Trainer(model=model, 
                        train_dataset=train_set,
                        compute_metrics=metric_func, 
                        args=args)
            
        trainer.train()
    else: 
        trainer = Trainer(model=model)

    if return_result:
        output = trainer.predict(test_set)
        preds = np.argmax(output[0], -1)
        results = classification_report(y_true=labels, y_pred=preds,
                                        output_dict=True,
                                        zero_division=0)

    logging.disable(logging.NOTSET)
    return trainer, results


def meta_train(meta_learner, meta_args, df_train, tokenizer, max_len,
               train_num_task, train_k_support, train_k_query,  
#                test_num_task:int, test_k_supports:list, test_k_query:int, 
               test_num_task:int, test_datasets: dict, 
               eval_while_train, val_metric_function, step_interval,
               val_num_train_epoch, skip_validation=False, return_best_statedict=False,
               disable_tqdm=True, seed=123):
    '''
    Train full loops with meta-learner model
    
    Compatible with basic binary tasks or tasks that share identical singular-classification structure 
    '''
    # set_seed(seed)
    acc_all_train = []
    idx_array = []
    
    validation_f1 =  {domain: {k: [] for k in v } for domain, v in test_datasets.items()}
    validation_acc = deepcopy(validation_f1)
    best_state_dict = deepcopy(validation_f1)
    output_dict = {seed: {'best_statedict': None}}
    
    # Create test loader
    test_args = TrainingArguments(output_dir = './',
                                  save_strategy='no',
                                  evaluation_strategy='epoch',
                                  learning_rate=meta_args.inner_update_lr,
                                  seed=seed,
                                  num_train_epochs=val_num_train_epoch, 
                                  logging_strategy='no',
                                  disable_tqdm=disable_tqdm)
    
    # Meta train 
    global_step = 0
    # set_seed(seed)
    start = time.time()
    for epoch in range(meta_args.meta_epoch):
        train = MetaLoader(df_train, num_task = train_num_task,  max_len=max_len,
                           k_support=train_k_support, k_query=train_k_query,
                           tokenizer = tokenizer, batch_mode='random', verbose=False)
        db = create_meta_batches(train, batch_size=meta_args.outer_batch_size)
        last_step = meta_args.meta_epoch * (ceil(len(train) / meta_args.outer_batch_size))
    
        for step, task_batch in enumerate(db):
            acc = 0
            acc = meta_learner(task_batch, is_training=True, verbose=False)
            acc_all_train.append(acc)
        
            if (not skip_validation and global_step % step_interval == 0) or (global_step == last_step - 1):
                idx_array.append(global_step)
                print('Step:', global_step, '\ttraining Acc:', acc)
                print("-----------------Meta Testing Mode-----------------")
                
                # compute results on each size of set K 
                for domain, test_dict in test_datasets.items(): 
                    print("---------- Evaluating on Domain {} ----------".format(domain.upper()))
                    for k_val, test_dataset in test_dict.items():
                        print("...Validating for k_val {}:".format(k_val))
                        set_seed(seed*5)
                        # create corresponding test set
                        test = MetaLoader(test_dataset, num_task = test_num_task, max_len=max_len,
                              k_support=k_val, k_query=k_val, 
                              tokenizer = tokenizer, batch_mode='fixed', verbose=True)

                        train_ds, test_ds = create_meta_batches(test, 1, split_to_train_test=True)

                        t_model = deepcopy(meta_learner.model)
                        trainer, results = quick_eval(t_model, train_ds, test_ds, test_args,
                                                    metric_func=compute_metrics, eval_while_train=False)
                        test_f1 = round(results['macro avg']['f1-score'],3)
                        test_acc = round(results['accuracy'],3)
                        validation_f1[domain][k_val].append(test_f1)
                        validation_acc[domain][k_val].append(test_acc)

                        # update if better results 
                        if (not validation_f1[domain][k_val]) or (test_f1 >= max(validation_f1[domain][k_val])): 
                            best_state_dict[domain][k_val] = deepcopy(meta_learner.model.state_dict())
                            print("Update best params at step {} for k_val {}".format(global_step, k_val))

                        

                        del test, train_ds, test_ds, t_model, trainer, results, test_f1, test_acc
                        gc.collect()
                        torch.cuda.empty_cache()
                
            global_step += 1
                
    print(acc_all_train)
    print("DURATION:", time.time() - start)
    print("GLOBAL STEP:", global_step, "LAST STEP", last_step)

    # compile output
    output_dict[seed]['train_acc'] = acc_all_train 
    output_dict[seed]['idx_array'] = idx_array
    output_dict[seed]['val_f1'] = validation_f1
    output_dict[seed]['val_acc'] = validation_acc
    if return_best_statedict:
        output_dict[seed]['best_statedict'] = best_state_dict

    return output_dict


def binary_train(model, epochs, batch_size, eval_interval, train_dataset, val_dataset, test_datasets:dict,\
                 test_args, seed, return_best_statedict=True): 
    '''
    Train full loops with BINARY models (e.g.: roberta)
    @param Test_datasets: dict, of form {domain: {k_val: df}} as results of the create_test_dict function
    '''
    # parameters to keep track of training steps 
    shape = len(train_dataset)
    max_steps = ceil(epochs*shape/batch_size)
    train_acc = []
    # steps_to_eval = ceil(number_of_steps/number_of_intervals)
    validation_f1 =  {domain: {k: [] for k in v } for domain, v in test_datasets.items()}
    validation_acc = deepcopy(validation_f1)
    best_state_dict = deepcopy(validation_f1)
    output_dict = {seed: {'best_statedict': None}}
    
    train_sampler = DataLoader(train_dataset, batch_size=batch_size)
    global_step = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    for e in range(epochs):
        for step, batch in enumerate(train_sampler):
            model.train()
            batch = {k:b.to(device) for k,b in batch.items()}
            # return batch 
            outputs = model(input_ids = batch['input_ids'], attention_mask=batch['attention_mask'],\
                            labels=batch['labels'])
        
            train_loss = outputs[0]
            # update gradient 
            train_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # train and eval results on the test set 
            if global_step % eval_interval == 0 or global_step >= max_steps - 1:
                # eval on val set
                if val_dataset  : 
                    set_seed(seed * 5)
                    v_model = deepcopy(model)
                    v_trainer = Trainer(v_model, compute_metrics=compute_metrics,\
                                args = TrainingArguments(disable_tqdm=True, \
                                                         output_dir="./"))
                    v_outputs = v_trainer.evaluate(val_dataset)
                    train_acc.append(v_outputs['eval_acc'])
                    # validation_f1.append(v_outputs['eval_loss'])
                    print("Step: {}, Training Loss: {}, Eval Accuracy: {}".format(global_step, v_outputs['eval_loss'],  v_outputs['eval_acc']))
                    del v_model, v_trainer, v_outputs
                    gc.collect()
                    torch.cuda.empty_cache()

                # eval on each of the test set
                for domain, test_dict in test_datasets.items():
                    print("---------- Evaluating on Domain {} ----------".format(domain.upper()))
                    for k_val, test_dataset in test_dict.items(): 
                        set_seed(seed * 5)
                        t_model = deepcopy(model)
                        t_trainer, t_results = quick_eval(t_model, test_dataset, test_dataset, test_args,
                                                    metric_func=compute_metrics, eval_while_train=False)
                        test_f1 = round(t_results['macro avg']['f1-score'],3)
                        test_acc = round(t_results['accuracy'],3)
                        validation_f1[domain][k_val].append(test_f1)
                        validation_acc[domain][k_val].append(test_acc)

                        # retain datedict with best f1
                        if (not validation_f1[domain][k_val]) or (test_f1 >= max(validation_f1[domain][k_val])): 
                            best_state_dict[domain][k_val] = deepcopy(t_trainer.model.state_dict())
                            print("Update best params at step", global_step)

                        del t_model, t_trainer, t_results, test_f1, test_acc
                        gc.collect()
                        torch.cuda.empty_cache()
            # update steps
            global_step += 1
    
    # tally all reports
    output_dict[seed]['train_acc'] = train_acc
    output_dict[seed]['val_f1'] =  validation_f1
    output_dict[seed]['val_acc'] =  validation_acc
    if return_best_statedict:
        output_dict[seed]['best_statedict'] = best_state_dict

    print("Total number of steps", global_step)
    return output_dict


