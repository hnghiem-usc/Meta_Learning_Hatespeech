# Class definitions of multi-task meta learners

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.optim import Adam, AdamW
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import RobertaTokenizer, RobertaModel, BertModel
from transformers import RobertaForSequenceClassification, BertForSequenceClassification
from copy import deepcopy
import gc
from sklearn.metrics import accuracy_score, f1_score

from loader import move_to_device

model_lookup = {'roberta':RobertaModel, 'bert':BertModel}


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


#############################  MULTI_TASK META LEARNER  #############################
class Multi_Meta_Learner(nn.Module):
    def __init__(self):
        super().__init__()
        self.class_name = 'Multi_Meta_Learner'
    
    def cleanup(self): 
        gc.collect()
        torch.cuda.empty_cache()
                
    def forward_fast_model(self): 
        return None
    
    def train_fast_model(self):
        return None
    
    def eval_fast_model(self):
        return None
    

class MAML_Unicorn(Multi_Meta_Learner):
    def __init__(self, args, model_lookup:dict=model_lookup ):
        super().__init__()
        self.class_name = 'MAML_Unicorn'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # num_labels might vary
        for key, attr in args.__dict__.items():
            setattr(self, key, attr)
            
        self.encoder = model_lookup[self.encoder_class].from_pretrained(self.encoder_dir, add_pooling_layer=False)
        self.components = {'encoder':self.encoder}
        classifier_dropout = (
            self.encoder.config.classifier_dropout if self.encoder.config.classifier_dropout is not None else self.encoder.config.hidden_dropout_prob
            )
        for i in range(self.num_classifier): 
            setattr(self, 'fc_' + str(i), torch.nn.Linear(self.encoder.config.hidden_size, self.encoder.config.hidden_size) )
            setattr(self, 'dropout_' + str(i), torch.nn.Dropout(classifier_dropout))
            setattr(self, 'cls_' + str(i),  nn.Linear(self.encoder.config.hidden_size, 1))
            # append all elements into component dict 
            for key in ['fc_', 'dropout_', 'cls_']:
                self.components[key + str(i)] = getattr(self,  key + str(i))
        
        # must manually specify all parameters 
        self.optimizer = AdamW([{'params': self.components[c].parameters()} for c in self.components], lr=self.outer_update_lr)
        self.train()
       
    
    @staticmethod
    def create_fast_model(components:dict, task_config: dict):
        '''
            Duplicate base model with correct conigurations according to task setting
            
            Parameters: 
            task_config: dict, output of Task loader
            components: dict of model's components, should be compatible with self.components
        '''
       
        fast_model = {k: deepcopy(v) for k, v in components.items()}
        fast_losses = dict()
        for cls_index, (label_var, config) in enumerate(task_config.items()):
            fast_model['cls_' + str(cls_index)] = nn.Linear(components['encoder'].config.hidden_size, config['num_classes'])
            # duplicate weights and biases in according to new model 
            fast_model['cls_' + str(cls_index)].weight.data = components['cls_' + str(cls_index)].weight.data.repeat(config['num_classes'], 1)
            fast_model['cls_' + str(cls_index)].bias.data = components['cls_' + str(cls_index)].bias.data.repeat(config['num_classes'])
            
        return fast_model
    
    
    @staticmethod
    def gather_fast_gradients(fast_model):
        '''
        Collect the current gradients of the fast model
        
        For classification layer (cls), sum up gradients to be same shape of base model 
        '''
        temp_gradients = {key: [] for key in fast_model}
        for key, component in fast_model.items():
            for i, param in enumerate(component.parameters()):
                # when cls_1+ not included in current task, add none
                if param.grad is None:
                    grads = torch.zeros_like(param.data)
                else: 
                    grads = deepcopy(param.grad).to(torch.device('cpu'))
                if 'cls' in key: 
                    # sum up gradients and reshape to that of original model's for classifier 
                    if i == 0: # weight
                        grads = grads.sum(dim=0).reshape(1,-1)
                    else: # bias
                        grads = grads.sum().reshape(1)

                temp_gradients[key].append(grads)

        return temp_gradients
    
    
    @staticmethod
    def collect_gradients(base_grads:dict, temp_grads:dict):
        '''
        Sequentially sum the correspoding items of 2 gradients 
            
        Base grads is assumed to contain more components. If not exist, do not add
        '''
        for key, grads in base_grads.items(): 
            if key in temp_grads:
                if grads:
                    base_grads[key] = [torch.add(f,t) for f,t in zip(grads, temp_grads[key])]
                else:
                    base_grads[key] = temp_grads[key]

                    
    @staticmethod
    def forward_fast_model(batch, task_config, model:dict, device ):
        '''
        One pass through the inner/fast model
        '''
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, *label_ids = batch
        
#         print("FAST_FORWARD_MODEL IS ON GPU? ", next(model['encoder'].parameters()).is_cuda ) ##!!!
        outputs = model['encoder'](input_ids = input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]
#         print('outputs shape', outputs[0].shape)
        
        losses = [] 
        logits_all = [] 
        labels_all = [] 
        
        for cls_index,  config in enumerate(task_config.values()):
            logits = sequence_output[:,0,:] # take cls token 
            logits = model['dropout_' + str(cls_index)](logits)
            logits = model['fc_' + str(cls_index)](logits)
            logits = torch.tanh(logits)
            logits = model['dropout_' + str(cls_index)](logits)
            logits = model['cls_' + str(cls_index)](logits)
#             print(config['problem_type'], config['num_classes'],  label_ids[cls_index].view(-1))
            
            # calculate losses according to problem type 
            if config['problem_type'] == 'multi_label_classification': 
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, label_ids[cls_index].to(torch.float32))
            elif config['problem_type'] == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, config['num_classes']), label_ids[cls_index].view(-1) )
                
            losses.append(loss)
            logits_all.append(logits)
            labels_all.append(label_ids[cls_index])
        
        return losses, logits_all, labels_all
    
    
    @staticmethod 
    def train_fast_model(model, optimizer, data_loader, task_config, device, loss_weights:list, 
                        is_training=True, optimize_params=True,
                        p_threshold = 0.5, 
                        base_gradients=None,
                        step_count=None
                       ):
        """
        Train the fast model using all the data in data loader 
        
        Cacualte losses and upgrade/calculate gradients via back-prop
        """
        loss_array = []
        accuracy_array = [[] for i in task_config] 
        f1_array = [[] for i in task_config] 
        
        [c.train() for _, c in model.items()]
        for step, batch in enumerate(data_loader):
            temp_gradients = None
            losses, logits, labels = MAML_Unicorn.forward_fast_model(batch, task_config, model, device)
            loss_array.append(float(sum(losses).item()))
            # calcualte results
            results = process_prediction(logits, labels, task_config, p_threshold, return_preds=False)
            [accuracy_array[i].append(acc) for i,acc in enumerate(results['acc'])] 
            [f1_array[i].append(f1) for i,f1 in enumerate(results['f1'])] 

            
            if is_training: 
                total_losses = sum([l * w for l, w in zip(losses, loss_weights)])
                total_losses.backward()
                # collect gradients 
                if base_gradients:
                    temp_gradients = MAML_Unicorn.gather_fast_gradients(model)
                    MAML_Unicorn.collect_gradients(base_gradients, temp_gradients)
                
                if optimize_params: 
                    optimizer.step()
                optimizer.zero_grad() 
            
        return loss_array, accuracy_array, f1_array
    
    
    @staticmethod
    def eval_fast_model(model, optimizer, data_loader, task_config, device, p_threshold=0.5):
        """
        Evaluate model through the entire dataset contained in (dataloader)
        """
        [c.eval() for _, c in model.items()]
        preds_all = [[] for i in task_config]
        labels_all = [[] for i in task_config]
        for i, (var, config) in enumerate(task_config.items()):
            if config['problem_type'] == 'multi_label_classification':
                preds_all[i] = np.zeros((0, config['num_classes']))
                labels_all[i] = np.zeros((0, config['num_classes']))
        
        for step, batch in enumerate(data_loader):
            losses, logits, labels = MAML_Unicorn.forward_fast_model(batch, task_config, model, device)
            _, preds = process_prediction(logits, labels, task_config, p_threshold, return_preds=True)
            for i, pred in enumerate(preds):
                if len(pred.shape) == 1: 
                    preds_all = [np.hstack([preds_all[i],pred]) for i, pred in enumerate(preds)]
                    labels_all = [np.hstack([labels_all[i],label]) for i, label in enumerate(labels)]
                else: 
                    preds_all = [np.vstack([preds_all[i],pred]) for i, pred in enumerate(preds)]
                    labels_all = [np.vstack([labels_all[i],label]) for i, label in enumerate(labels)]       
            
        del losses, logits, labels, preds    
        gc.collect() 
        torch.cuda.empty_cache() 
        
        return preds_all, labels_all
    
    
    def forward(self, batch_tasks, meta_training=True, is_training=False, verbose=False):
        '''
        One iteration of meta-training
        '''
        # Meta train using SUPPORT tasks
        num_inner_update_step = self.inner_update_step if is_training else self.inner_update_step_eval
        support_losses = []
        task_counter = {i:0 for i in range(self.num_classifier)}
        all_gradients = {k: [] for k in self.components}
        
        for task_id, task in enumerate(batch_tasks):
            support, query, task_config, domains = task
            support_accuracies = [[] for t in task_config]
            support_f1 = deepcopy(support_accuracies)
            query_accuracies = deepcopy(support_accuracies)
            query_f1 = deepcopy(support_accuracies)
            
            for t in range(len(task_config)): #increase corresponding to number of class. heads
                task_counter[t] += 1 
            
            fast_model  = self.create_fast_model(self.components, task_config)
            fast_gradients = {key: [] for key in fast_model}
#             (component.to(self.device) for component in fast_model.values()) # move to device
            fast_model = move_to_device(fast_model, self.device)
            fast_optimizer = AdamW(params=[{'params': fast_model[c].parameters()} for c in fast_model], lr=self.inner_update_lr)
            
            support_dataloader = DataLoader(support, sampler=RandomSampler(support),
                                            batch_size=self.inner_batch_size)

            if meta_training: 
                for i in range(num_inner_update_step):
                    s_losses, s_accuracies, s_f1 = self.train_fast_model(fast_model, fast_optimizer, support_dataloader,
                                                        task_config, self.device, self.loss_weights, 
                                                        is_training=True, optimize_params=True, 
                                                        )
                    [support_accuracies[i].append(round(np.mean(acc),3)) for i, acc in enumerate(s_accuracies)]
                    [support_f1[i].append(round(np.mean(f1),3)) for i, f1 in enumerate(s_f1)]
                    support_losses.append(round(np.mean(s_losses),4))
        
            # shorten if too long
            print("\nDomains:{}".format(domains))
            if verbose: 
                if len(support_losses) > 10:
                    support_losses = [loss for i, loss in enumerate(support_losses) if i % 10 == 0]
                print("Support accuracies: {}\nSupport F1s: {}\nSupport losses:{}".format(support_accuracies,\
                                                                                                        support_f1, support_losses))

            # Meta evaluate using QUERY tasks
            query_dataloader = DataLoader(query, sampler=None, batch_size=self.inner_batch_size)
            # DO NOT UPDATE PARAMETERS
            q_losses, q_accuracies, q_f1 = self.train_fast_model(fast_model, fast_optimizer, query_dataloader,
                                                task_config, self.device, self.loss_weights, 
                                                is_training=True, optimize_params=False,
                                                base_gradients=fast_gradients
                                                )
            [query_accuracies[i].append(np.mean(acc)) for i, acc in enumerate(q_accuracies)]
            [query_f1[i].append(np.mean(f1)) for i, f1 in enumerate(q_f1)]
            self.collect_gradients(all_gradients, fast_gradients)
            
            del fast_model, fast_optimizer, support_dataloader, query_dataloader, \
                s_losses, s_accuracies, s_f1, q_losses, q_accuracies, q_f1
            self.cleanup()
                    
            print("Query accuracies:{}\nQuery F1s: {}".format(query_accuracies, query_f1))
        
        # one gradients are collected for all tasks, update base model's parameters
        if is_training:
            print("Tallying gradients over {} classification tasks".format(max(task_counter.values())))
            for key, component in self.components.items():
                for i, param in enumerate(component.parameters()): 
                    if all_gradients[key]: # update grad ony if not empty
                        if 'cls' in key or 'fc' in key:
#                             param.grad = all_gradients[key][i] / max(1, float(task_counter[int(key[-1])]))
                            param.grad = all_gradients[key][i]

                        else:
#                             param.grad = all_gradients[key][i] / float(len(batch_tasks))
                            param.grad = all_gradients[key][i] 

    
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return query_accuracies