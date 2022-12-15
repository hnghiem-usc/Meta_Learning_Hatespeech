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
from sklearn.metrics import accuracy_score

model_lookup = {'roberta':RobertaForSequenceClassification, 'bert':BertForSequenceClassification}

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