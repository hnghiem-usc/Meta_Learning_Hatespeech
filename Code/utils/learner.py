import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification, BertForSequenceClassification
from copy import deepcopy
import gc
from sklearn.metrics import accuracy_score, f1_score
from math import ceil, floor, prod

model_lookup = {'roberta':RobertaForSequenceClassification, 'bert':BertForSequenceClassification}


#############################  HELPER FUNCTIONS  #############################
def sigmoid(x):
    return np.where(x >= 0, 
                    1 / (1 + np.exp(-x)), 
                    np.exp(x) / (1 + np.exp(x)))
  

def process_prediction(logits_all, labels_all, task_config, p_threshold=0.5, return_preds=False): 
    '''
    Order of task config must match the index of logits and labels 
    
    Use only inside of Meta_learner classes
    '''
    results = {'acc':[], 'f1':[]}
    preds_all = [] 
    for idx, (_, v) in enumerate(task_config.items()):
        problem_type = v['problem_type']
        logits = logits_all[idx].clone().detach().cpu().numpy()
        labels = labels_all[idx].clone().detach().cpu().numpy()
        
        if problem_type == 'multi_label_classification':
            preds = (sigmoid(logits) >= p_threshold).astype(int)
        else:
            preds = np.argmax(logits, axis=1)
        
        preds_all.append(preds)
        results['acc'].append(round(accuracy_score(y_pred = preds, y_true=labels),3))
        results['f1'].append(round(f1_score(y_pred = preds, y_true=labels, average='macro', zero_division=0),3))

        del logits, labels
    
    return (results, preds_all) if return_preds else (results)


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad
        

def get_average_report(*args, num_round=3):
    report = []
    for item in args:
        if isinstance(item[0], list):
            avg = [round(np.mean(i), num_round) for i in item]
        else:
            avg = round(np.mean(item), num_round)
        report.append(avg)
    return tuple(report)

##### FUNCTIONS TO CALCULATE SHAPE OF OUTPUT FOR DIFFERENT LAYERS
def convert_to_tuple(d: int, length=2): 
    return tuple([d for i in range(length)])

def conv2d_output_shape(input_shape: tuple, kernel, padding=(1,1), stride=(1,1), dilation=(1,1), c_out=13):
    n, c_in, h_in , w_in = input_shape
    h_out =  floor((h_in + 2*padding[0] - dilation[0] * (kernel[0] - 1) - 1)/stride[0]) + 1
    w_out =  floor((w_in + 2*padding[1] - dilation[1] * (kernel[1] - 1) - 1)/stride[1]) + 1
    return (n, c_out, h_out, w_out)

def max2d_output_shape(input_shape, kernel, stride=(1,1), dilation=(1,1), padding=(0,0)):
    n, c, h_in, w_in = input_shape
    h_out = floor((h_in + 2*padding[0] - dilation[0] * (kernel[0] - 1) - 1)/stride[0] + 1 )
    w_out = floor((w_in + 2*padding[1] - dilation[1] * (kernel[1] - 1) - 1)/stride[1] + 1 )
    return (n, c, h_out, w_out)

def get_extractor_output_shape(input_shape:tuple, ops:dict = {'conv2d': ((3, 768),1,1,1), 'maxpool2d': (3,1,1,0)}):
    output_shape = input_shape
    shape_function_lookup = {'conv2d': conv2d_output_shape, 'maxpool2d': max2d_output_shape}

    for op, args in ops.items():
        args = [arg if type(arg) is tuple else convert_to_tuple(arg) for arg in args ]
        print(args)
        output_shape = shape_function_lookup[op](output_shape, *args)
    return output_shape
      
    
#############################  SINGLE/BINARY META LEARNER  #############################
class MAML_Learner(nn.Module):
    def __init__(self, args, model_lookup:dict=model_lookup):
        """
        :param args:
        """
        super(MAML_Learner, self).__init__()

        self.num_labels = args.num_labels
        self.outer_batch_size = args.outer_batch_size
        self.inner_batch_size = args.inner_batch_size
        self.outer_update_lr  = args.outer_update_lr
        self.inner_update_lr  = args.inner_update_lr
        self.inner_update_step = args.inner_update_step
        self.inner_update_step_eval = args.inner_update_step_eval
        self.model_class = args.model_class
        self.model_dir = args.model_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model_lookup[self.model_class].from_pretrained(self.model_dir, num_labels = self.num_labels)
        self.outer_optimizer = Adam(self.model.parameters(), lr=self.outer_update_lr) #for meta learner
        self.model.train()
        
        
    def forward(self, batch_tasks, is_training=True, meta_train=True, verbose=False):
        """
        batch_tasks: output of create_batch_tasks
        """
        task_accs = []
        sum_gradients = []
        num_task = len(batch_tasks)
        num_inner_update_step = self.inner_update_step if is_training else self.inner_update_step_eval

        for task_id, task in enumerate(batch_tasks):
            support = task[0]
            query   = task[1]
            
            fast_model = deepcopy(self.model)
            fast_model.to(self.device)
            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=self.inner_batch_size)
            
            inner_optimizer = Adam(fast_model.parameters(), lr=self.inner_update_lr)
            fast_model.train()
            
            if meta_train: 
                # train and update inner model
                for i in range(0,num_inner_update_step):
                    all_loss = []
                    num_iter = 0
                    for inner_step, batch in enumerate(support_dataloader):
                        num_iter += 1
                        batch = tuple(t.to(self.device) for t in batch)
                        input_ids, attention_mask, label_id = batch
                        outputs = fast_model(input_ids, attention_mask, labels = label_id)

                        loss = outputs[0]              
                        loss.backward()
                        inner_optimizer.step()
                        inner_optimizer.zero_grad()

                        all_loss.append(loss.item())

                    if verbose: 
                        if i % 5 == 0:
                            print('----Task',task_id, '----')
                            print("Inner Loss: ", np.mean(all_loss))

            # Process query set also in batch in case of large size
            query_dataloader = DataLoader(query, sampler=None, batch_size=self.outer_batch_size)
            
            query_labels = []
            query_preds  = []
            for _, query_batch in enumerate(query_dataloader): 
                query_batch = tuple(t.to(self.device) for t in query_batch)
                q_input_ids, q_attention_mask,  q_label_id = query_batch
                q_outputs = fast_model(q_input_ids, q_attention_mask, labels = q_label_id)

                # Meta-test on query set 
                if is_training:
                    q_loss = q_outputs[0]
                    q_loss.backward()
                    # this part should be done after all batches are processed?
                    # fast_model.to(torch.device('cpu'))
                    for i, params in enumerate(fast_model.parameters()):
                        if task_id == 0:
                            sum_gradients.append(deepcopy(params.grad).to(torch.device('cpu')))
                        else:
                            sum_gradients[i] += deepcopy(params.grad).to(torch.device('cpu'))
                    
                # collect labels 
                q_logits = F.softmax(q_outputs[1],dim=1)
                pre_label_id = torch.argmax(q_logits,dim=1)
                pre_label_id = pre_label_id.detach().cpu().numpy().tolist()
                q_label_id = q_label_id.detach().cpu().numpy().tolist()
                query_labels += q_label_id
                query_preds += pre_label_id 
                    
            acc = accuracy_score(query_preds, query_labels)
            task_accs.append(acc)
                
            fast_model.to(torch.device('cpu'))
            del fast_model, inner_optimizer
            torch.cuda.empty_cache()
        
        if is_training:
            # Average gradient across tasks
            for i in range(0,len(sum_gradients)):
                sum_gradients[i] = sum_gradients[i] / float(num_task)

            #Assign gradient for original model, then using optimizer to update its weights
            for i, params in enumerate(self.model.parameters()):
                params.grad = sum_gradients[i]

            self.outer_optimizer.step()
            self.outer_optimizer.zero_grad()
            
            del sum_gradients
            gc.collect()
        
        # consider retuning predictions instead for deeper analyses
        return np.mean(task_accs)
    
    
#############################  BASE CLASS FOR MULTI META LEARNER  #############################
class Multi_Meta_Learner(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_name = 'Multi_Meta_Learner'
                
    def forward_fast_model(self): 
        return None
    
    def train_fast_model(self):
        return None
    
    def eval_fast_model(self):
        return None