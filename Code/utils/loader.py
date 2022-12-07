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
from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_metric, Dataset as hg_Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
from math import ceil

import time
import logging


############################## META LOADER ##############################

# Craete a custom dataset loader that returns a batch of [support-query] set
# Requires parameters to control the data in each batch: 
# number of tasks to sample, size of query, size of support, and argumentns for tokenzier
class MetaLoader(Dataset): 
    def __init__(self, df: pd.DataFrame, num_task, k_support, k_query,tokenizer, max_len=128, truncation=True,
                 domain_var='domain', text_var='text_std', label_var='label_bin', test_domains=[] ,
                 batch_mode = 'random', label_dict={'positive':1, 'negative':0}, verbose=False,seed=123):
        
        self.data = dict()
        for d in df[domain_var].unique():
            self.data[d]  = df.loc[df[domain_var] == d, :].sample(frac=1).to_dict(orient='records')
        self.domain_list = list(self.data.keys())
        
        self.num_task  = num_task
        self.k_support = k_support
        self.k_query   = k_query 
        self.label_dict = label_dict
        self.domain_var = domain_var 
        self.test_domains = test_domains
        self.text_var = text_var
        self.label_var = label_var
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.truncation = truncation
        self.verbose = verbose
        
        print("Tokenizer Config:\nMax Lenght: {}, Truncation: {}".format(max_len, truncation))
        
        if batch_mode == 'fixed': 
            self.create_fixed_batch()
        elif batch_mode == 'disjoint':
            self.create_disjoint_batch()
        else:
            self.create_random_batch()
    
    
    def __reset_batch__(self):
        self.support = [] 
        self.query = []

 
    def create_random_batch(self):
        """
        Arrange data into random query and support test. 
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
        if self.verbose: 
            print("Creating FIXED batch...")
        self.__reset_batch__()
        # assume that there is only 1 domain 
        sample_train = self.data[self.domain_list[0]][:self.k_support]
        sample_test  = self.data[self.domain_list[0]][self.k_support:self.k_support + self.k_query]
        self.support.append(sample_train)     
        self.query.append(sample_test)
        
        
    def create_feature_set(self, samples, text_var, label_var):
        """ 
        Tokenize the samples in each batch of support/query into tensordataset. 
        Note: the order of output is hardcoded below 
        """
        text_ls = []
        label_ls = [l[label_var] for l in samples]
        
        for sample in samples:
            text_ls.append(sample[text_var]) 
            
        # fill in blanks if empty    
        if len(text_ls) == 0:
            text_ls = [' ']
            label_ls = [0]
        # tokenize
        sample = self.tokenizer(text_ls, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=self.truncation)
        return TensorDataset(sample['input_ids'], sample[ 'attention_mask'], torch.tensor(label_ls))
        
        
    def __getitem__(self, index):
        support_set = self.create_feature_set(self.support[index],self.text_var, self.label_var)
        query_set = self.create_feature_set(self.query[index], self.text_var, self.label_var)
        return support_set, query_set
    
    
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
            

metric = load_metric('accuracy')
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def collate_to_Dataset(tensorset, position_dict={'input_ids':0, 'attention_mask':1, 'labels':2}):
    batch = dict()
    batch['input_ids'] = torch.stack([i[position_dict['input_ids']] for i in tensorset])
    batch['attention_mask'] = torch.stack([i[position_dict['attention_mask']] for i in tensorset])
    batch['labels'] = torch.stack([i[position_dict['labels']] for i in tensorset])

    return hg_Dataset.from_dict(batch)


############################## TRAINING HELPER FUNCTIONS ##############################

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


def meta_train(meta_learner, meta_args, df_train, df_test, tokenizer, max_len,
               train_num_task, train_k_support, train_k_query,  
               test_num_task, test_k_support, test_k_query, 
               eval_while_train, val_metric_function, step_interval,
               val_num_train_epoch, skip_validation=False, return_best_statedict=False,
               seed=123):
    # set_seed(seed)
    acc_all_train = []
    acc_all_test = []
    idx_array = []
    validation_results = []
    validation_f1 = []
    validation_acc = []

    output_dict = {seed: {'report': (), 'best_statedict': None}}
    # Create test loader
    test = MetaLoader(df_test, num_task = test_num_task, max_len=max_len,
                      k_support=test_k_support, k_query=test_k_query, 
                      tokenizer = tokenizer, batch_mode='fixed', verbose=True)
    train_ds, test_ds = create_meta_batches(test, 1, split_to_train_test=True)
    test_args = TrainingArguments(output_dir = './',
                                  save_strategy='no',
                                  evaluation_strategy='epoch',
                                  learning_rate=meta_args.inner_update_lr,
                                  seed=seed,
                                  num_train_epochs=val_num_train_epoch, 
                                  logging_strategy='no')
    
    # Meta train 
    global_step = 0
    # set_seed(seed)
    best_state_dict = meta_learner.model.state_dict()
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
                print("-----------------Testing Mode-----------------")
                
                t_model = deepcopy(meta_learner.model)
                trainer, results = quick_eval(t_model, train_ds, test_ds, test_args,
                                            metric_func=compute_metrics, eval_while_train=False)
                test_f1 = round(results['macro avg']['f1-score'],3)
                test_acc = round(results['accuracy'],3)

                # update if better results 
                if (not validation_f1) or (test_f1 >= max(validation_f1)): 
                    best_state_dict = deepcopy(meta_learner.model.state_dict())
                    print("Update best params at step", global_step)
                
                validation_results.append(results)
                validation_f1.append(test_f1)
                validation_acc.append(test_acc)

                del trainer, results, t_model, test_f1
                gc.collect()
                torch.cuda.empty_cache()
                # reset seed to ensure reproducibility
                set_seed(seed + global_step)
                
            global_step += 1

        # Final evaluation
                
    print(acc_all_train)
    print("DURATION:", time.time() - start)
    print("GLOBAL STEP:", global_step, "LAST STEP", last_step)

    # compile output
    output_dict[seed]['report'] = (acc_all_train, validation_results , idx_array, validation_f1, validation_acc)
    if return_best_statedict:
        output_dict[seed]['best_statedict'] = best_state_dict

    return output_dict
