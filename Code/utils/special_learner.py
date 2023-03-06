# DEFITION OF CLASS SPECIAL LEARNER 
import os, sys
os.environ['OPENBLAS_NUM_THREADS'] = '1'
sys.path.append('../../../Code/utils')

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
from math import floor, prod
from torch.autograd import grad

from loader import *
from learner import *

model_lookup = {'roberta':RobertaModel, 'bert':BertModel}

class Special_Learner(Multi_Meta_Learner):
    def __init__(self, args, model_lookup:dict = model_lookup): 
        super().__init__()
        self.class_name = 'Special_Learner'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # num_labels might vary
        for key, attr in args.__dict__.items():
            setattr(self, key, attr)
        self.embedding = model_lookup[self.embedding_class].from_pretrained(self.embedding_dir, add_pooling_layer=False, output_hidden_states=True)
        sequence_length = args.sequence_length
        hidden_size     = self.embedding.config.hidden_size
        self.components = collections.OrderedDict({'embedding': self.embedding})
        
        base_dropout = (self.embedding.config.classifier_dropout if self.embedding.config.classifier_dropout \
                        is not None else self.embedding.config.hidden_dropout_prob )
        # Feature Extractor
        self.conv = nn.Conv2d(in_channels=args.in_channels,
                              out_channels=args.out_channels, 
                              kernel_size=(args.conv_kernel, hidden_size), 
                              padding=args.conv_padding)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=args.pool_kernel, stride=args.pool_stride)
        self.dropout = nn.Dropout(0.1)
        self.extractor = nn.Sequential(collections.OrderedDict([('conv', self.conv), 
                                                                ('relu', self.relu), 
                                                                ('dropout', self.dropout), 
                                                                ('pool', self.pool)
                                                                ]))
        self.components['extractor'] = self.extractor
        
        ops_dict = {'conv2d': ((args.conv_kernel, hidden_size), args.conv_padding), 
                     'maxpool2d':(args.pool_kernel, args.pool_stride)
                    }
        extractor_output_shape = get_extractor_output_shape((0, args.in_channels, sequence_length, hidden_size), ops_dict)
        # Domain Critic
        self.critic = nn.Sequential(nn.Linear(prod(extractor_output_shape[1:]), self.critic_itrm_shape),
                                    nn.ReLU(),
                                    nn.Linear(self.critic_itrm_shape, self.critic_output_shape),
                                    nn.ReLU(),
                                    nn.Linear(self.critic_output_shape, 1))
        self.critic_optim = AdamW(self.critic.parameters(), lr=self.critic_lr) 
        self.components['critic'] = self.critic
        # Classifier
        for i in range(self.num_classifier): 
            cls_dict = collections.OrderedDict()
            cls_dict['fc']     = nn.Linear(prod(extractor_output_shape[1:]), hidden_size) 
            cls_dict['dropout']= nn.Dropout(base_dropout)
            cls_dict['cls']    = nn.Linear(hidden_size, 1)
            setattr(self, 'classifier_' +str(i), nn.Sequential(cls_dict) )
            setattr(self, 'classifier_' +str(i)+'_optimizer', AdamW(getattr(self, 'classifier_' +str(i)).parameters(), lr=self.classifier_lr))
            self.components['classifier_' + str(i)] = getattr(self, 'classifier_' + str(i))
            
        self.include_components = list(set(self.components.keys()) - set(self.exclude_components)) 
        self.lr_dict = {k + '_lr': getattr(self, k.split('_')[0] + '_lr')  for k in self.components if hasattr(self, k.split('_')[0] + '_lr') }
        self.fast_lr_dict = {k: self.fast_classifier_lr if 'classifier' in k else v for k,v in self.lr_dict.items()}
        print("lr_dict:{}\nfast_lr_dict:{}".format(self.lr_dict, self.fast_lr_dict))
    
    @staticmethod
    def create_fast_model(components:dict, task_config, exclude_components:list, optimizer_include:list=None, fast_lr_dict:dict=None):
        """
            Clone the components 
            
            If task_config is provided, then module is a classifier. Classification layers will be 
            duplicated accordingly 
        """
        fast_model = {k: v if k in exclude_components else deepcopy(v) for k, v in components.items()}
        if task_config:
            for cls_index, (label_var, config) in enumerate(task_config.items()):
                orig_cls = fast_model['classifier_' + str(cls_index)].cls
                orig_cls_in_features = orig_cls.in_features
                new_cls_layer = nn.Linear(orig_cls_in_features, config['num_classes'])
                new_cls_weight = orig_cls.weight.data.repeat(config['num_classes'], 1)
                new_cls_data   = orig_cls.bias.repeat(config['num_classes'])
                fast_model['classifier_' + str(cls_index)].cls = new_cls_layer 
                fast_model['classifier_' + str(cls_index)].cls.weight.data = new_cls_weight
                fast_model['classifier_' + str(cls_index)].cls.bias.data   = new_cls_data
                
                del new_cls_layer, new_cls_weight, new_cls_data
        
        # create optimizer
        if optimizer_include: 
            include_components = list(set(components.keys()) - set(exclude_components)) 
            fast_optimizer_include = deepcopy(optimizer_include)
            fast_optimizer_include += include_components
#             print("fast_model_optimizer", fast_optimizer_include)
            fast_optimizer = AdamW(params=[{'params': fast_model[c].parameters(), 
                                             'lr': fast_lr_dict[c+'_lr'] } for c in  fast_model if c in fast_optimizer_include])
            return fast_model, fast_optimizer
        return fast_model 
    
    
    @staticmethod
    def gather_fast_gradients(fast_model:dict):
        '''
        Collect the current gradients of the fast model

        For classification Squential Module, sum up gradients of the CLS layer's weights and biases
        to fit the base model
        '''
        temp_gradients = {key: [] for key in fast_model}
        for key, component in fast_model.items():
            if 'classifier' in key:
                for _, (name, param) in enumerate(component.named_parameters()):
                    if param.grad is None:
                        grads = torch.zeros_like(param.data)
                    else: 
                        grads = deepcopy(param.grad).to(torch.device('cpu'))
                    if name == 'cls.weight': # weight
                        grads = grads.sum(dim=0).reshape(1,-1)
                    elif name == 'cls.bias': # bias
                        grads = grads.sum().reshape(1)
                    temp_gradients[key].append(grads)
            else:
                for i, param in enumerate(component.parameters()):
                    # when cls_1+ not included in current task, add none
                    if param.grad is None:
                        grads = torch.zeros_like(param.data)
                    else: 
                        grads = deepcopy(param.grad).to(torch.device('cpu'))

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

    
    def gradient_penalty(self, critic, h_s, h_t, device):
        """
        Caculate the gradient penalty needed for Wassentein Distance 
        # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
        """
        alpha = torch.rand(h_s.size(0), 1).to(device)
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

        preds = critic(interpolates)
        gradients = grad(preds, interpolates,
                         grad_outputs=torch.ones_like(preds),
                         retain_graph=True, create_graph=True)[0]
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1)**2).mean()
        return gradient_penalty
    
    @staticmethod
    def forward_extractor(batch, embedding, extractor, device, no_grad=False):
        """
        One pass of feature extractor
        
        Need no grad or not? 
        """
        embedding = move_to_device(embedding, device)
        extractor = move_to_device(extractor, device)
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_mask, *labels = batch 
    
        outputs = embedding(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[1] 
        
        x = torch.transpose(torch.cat(tuple([t.unsqueeze(0) for t in hidden_states]), 0), 0, 1)
        del hidden_states
        cleanup()
        
        if no_grad: 
            with torch.no_grad(): x = extractor(x)
        else: x = extractor(x)
#         print("post_extractor", x.shape)
        x = torch.flatten(x, start_dim=1)
        return x, labels

    
    @staticmethod
    def forward_classifier(classifier:dict, inputs, labels, task_config, device):
        losses = [] 
        logits_all = []
        labels_all = []
        classifier = {k:move_to_device(v, device) for k,v in classifier.items()}
        for cls_index,  config in enumerate(task_config.values()):
            logits = classifier['classifier_' + str(cls_index)](inputs)
            # calculate losses according to problem type 
            if config['problem_type'] == 'multi_label_classification': 
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels[cls_index].to(torch.float32))
            elif config['problem_type'] == 'single_label_classification':
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, config['num_classes']), labels[cls_index].view(-1)) 

            losses.append(loss)
            logits_all.append(logits)
            labels_all.append(labels[cls_index])

        return losses, logits_all, labels_all
    
    
    # should pass in fast_feature 
    @staticmethod
    def train_fast_model(model, optimizer, source_loader, target_loader, task_config, device, loss_weights:list, 
                        is_training=True, optimize_params=True, switch_to_eval=False, 
                        p_threshold = 0.5, wd_clf=0.1, 
                        base_gradients=None,
                        step_count=None, 
                        scheduler=None,
                        return_extractor_features=True):

        loss_array = [] 
        accuracy_array = [[] for i in task_config]
        f1_array = [[] for i in task_config]
        source_features = [] 
        wasserstein_distance = 0 
    
        if target_loader: 
            zip_loader = zip(source_loader, target_loader)
        else: 
            zip_loader = source_loader
        model = {k: c.train() for k, c in model.items()}

        if switch_to_eval: 
            model = {k: c.eval() for k, c in model.items()}    
        set_requires_grad(model['critic'], requires_grad=False)

        for step,  zip_batch in enumerate(zip_loader):
            temp_gradients = None
            if target_loader: 
                s_batch, t_batch =  zip_batch
                h_t, _  = Special_Learner.forward_extractor(t_batch, model['embedding'], model['extractor'], device )
            else: 
                s_batch = zip_batch
            h_s, labels = Special_Learner.forward_extractor(s_batch, model['embedding'], model['extractor'], device )
           
            source_features.append(h_s)
            classifier = {k: v for k,v in model.items() if 'classifier' in k}
            losses, logits, labels = Special_Learner.forward_classifier(classifier, h_s, labels, task_config, device)
            loss_array.append(float(sum(losses).item()))

            # calculate WD w/ target set 
            if target_loader: 
                wasserstein_distance = model['critic'](h_s).mean() - model['critic'](h_t).mean()
#                 print("Fast Training Wassertein distance:",  wasserstein_distance)
            #calculate results
            results = process_prediction(logits, labels, task_config, p_threshold, return_preds=False)
            [accuracy_array[i].append(acc) for i,acc in enumerate(results['acc'])] 
            [f1_array[i].append(f1) for i,f1 in enumerate(results['f1'])] 

            #accumulate gradients
            if is_training: 
                total_losses = sum([l * w for l, w in zip(losses, loss_weights)])
                total_losses += wd_clf * wasserstein_distance
                total_losses.backward()
                # collect gradients for Source only 
                if base_gradients:
                    temp_gradients = Special_Learner.gather_fast_gradients(model)
                    Special_Learner.collect_gradients(base_gradients, temp_gradients)

                if optimize_params: 
                    optimizer.step()
                    if scheduler: scheduler.step()
                    optimizer.zero_grad() 
                    
        if return_extractor_features:
            return loss_array, accuracy_array, f1_array, source_features
        return loss_array, accuracy_array, f1_array
    
    
    @staticmethod
    def eval_fast_model(model, optimizer, data_loader, task_config, device, p_threshold=0.5,):
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
            temp_gradients = None
            inputs, labels = Special_Learner.forward_extractor(batch, model['embedding'], model['extractor'], device )
            classifier = {k: v for k,v in model.items() if 'classifier' in k}
            losses, logits, labels = Special_Learner.forward_classifier(classifier, inputs, labels, task_config, device)
            _, preds = process_prediction(logits, labels, task_config, p_threshold, return_preds=True)
            
            for i, pred in enumerate(preds):
                if len(pred.shape) == 1: 
                    preds_all = [np.hstack([preds_all[i],pred]) for i, pred in enumerate(preds)]
                    labels_all = [np.hstack([labels_all[i],label.cpu()]) for i, label in enumerate(labels)]
                else: 
                    preds_all = [np.vstack([preds_all[i],pred]) for i, pred in enumerate(preds)]
                    labels_all = [np.vstack([labels_all[i],label.cpu()]) for i, label in enumerate(labels)]  
            
        del losses, logits, labels, preds    
        cleanup()
        
        return preds_all, labels_all
    
    
    def forward(self, batch_tasks, meta_training=True, is_training=False, verbose=False):
        """
        One iteration of meta-training
        """
        # Meta train using SUPPORT tasks
        num_inner_update_step = self.inner_update_step if is_training else self.inner_update_step_eval
        print(self.include_components)
        
        support_losses = []
        support_accuracies = [[] for t in range(self.num_classifier)]
        support_f1 = deepcopy(support_accuracies)
        query_accuracies = deepcopy(support_accuracies)
        query_f1 = deepcopy(support_accuracies)
        support_gradients = {k: [] for k in self.components if k in self.include_components} 
        query_gradients   = {k: [] for k in self.components if k in self.include_components} 
        task_counter = {i:0 for i in range(self.num_classifier)}
        
        critic_loss = 0
        self.critic = move_to_device(self.critic, self.device)
        for task_id, task in enumerate(batch_tasks): 
            (s_ds, s_config, s_domain), (t_ds, t_config, t_domain), (q_ds, q_config, q_domain) = task
            print("s_domain: {}, target:{}, q_domain: {}".format(s_domain, t_domain, q_domain))
            support_loader = DataLoader(s_ds, sampler=RandomSampler(s_ds), batch_size=self.inner_batch_size)
            target_loader = DataLoader(t_ds, sampler=RandomSampler(t_ds), batch_size=self.inner_batch_size)
            query_loader = DataLoader(q_ds, sampler=RandomSampler(q_ds), batch_size=self.inner_batch_size)
            
            # Train critic on Source and Target
            zip_iterator = zip(loop_iterable(support_loader), loop_iterable(target_loader))
            for e in range(self.critic_epochs): 
                # use source-target terminology
                s_batch, t_batch = next(zip_iterator)
                h_s, _ = self.forward_extractor(s_batch, self.embedding, self.extractor, self.device, no_grad=False)
                h_t, _ = self.forward_extractor(t_batch, self.embedding, self.extractor, self.device, no_grad=False)
                gp  = self.gradient_penalty(self.critic, h_s, h_t, self.device)
                
                critic_s = self.critic(h_s)
                critic_t = self.critic(h_t)
                wasserstein_distance = critic_s.mean() - critic_t.mean()
                critic_criterion = -wasserstein_distance + self.critic_gamma*gp
#                 print("Critic Training, Wassertein Distance: {}, Critic Loss:{}".format(wasserstein_distance, critic_criterion))
                self.critic_optim.zero_grad()
                critic_criterion.backward()
                self.critic_optim.step()
            
                critic_loss += critic_criterion.item()
                
            if verbose:
                print("critic_loss", critic_criterion.item())
                
            del zip_iterator, s_batch, t_batch, h_s, h_t, critic_s, critic_t, wasserstein_distance
            cleanup()
            
            # fast model only duplicates extractor and model 
            s_outputs = []
            s_model, s_optimizer = self.create_fast_model(self.components, s_config, self.exclude_components, self.fast_optimizer_include, self.fast_lr_dict)
            q_model = self.create_fast_model(self.components, q_config, self.exclude_components)
            
            for e in range(num_inner_update_step):
                s_gradients = {k: [] for k in s_model if k not in ['embedding', 'critic']} 
                s_losses, s_accuracies, s_f1, s_ext_features  = self.train_fast_model(s_model, s_optimizer, support_loader, target_loader,
                                                                s_config,self.device, loss_weights=self.loss_weights, base_gradients=s_gradients, 
                                                                wd_clf = self.wd_clf)
                s_loss_mean = get_average_report(s_losses, num_round=4)[0]
                s_acc_mean, s_f1_mean = get_average_report(s_accuracies, s_f1)
                support_losses.append(s_loss_mean)
                [support_accuracies[i].append(acc) for i, acc in enumerate(s_acc_mean)]
                [support_f1[i].append(f1) for i, f1 in enumerate(s_f1_mean)]
                 
                # append data in arrays ! 
                if verbose:
                    print("Average support_losses: {}\t support_acc:{}\t support_F1:{}".format(s_loss_mean, s_acc_mean, s_f1_mean))
                       
            ## next step: get accumulated gradient in the evaluatioon on query set
            q_gradients = {k: [] for k in q_model if k not in 'embedding'} 
            q_losses, q_accuracies, q_f1, q_ext_features = self.train_fast_model(q_model, None, query_loader, None, q_config, self.device,
                                            is_training=True, optimize_params=False, switch_to_eval=True, 
                                            base_gradients = q_gradients, loss_weights=self.loss_weights, wd_clf=self.wd_clf) 
            q_loss_mean = get_average_report(q_losses, num_round=4)[0]
            q_acc_mean, q_f1_mean = get_average_report(q_accuracies, q_f1)
            [query_accuracies[i].append(acc) for i, acc in enumerate(q_acc_mean)]
            [query_f1[i].append(f1) for i, f1 in enumerate(q_f1_mean)]
            
            self.collect_gradients(support_gradients, s_gradients)
            self.collect_gradients(query_gradients, q_gradients)
            
            del s_gradients, s_losses, s_accuracies, s_f1, s_ext_features
            del q_gradients, q_losses, q_accuracies, q_f1, q_ext_features
            cleanup()
           
        # Meta-update
        if is_training:
            print("Tallying gradients over {} classification tasks".format(max(task_counter.values())))
            for key, component in self.components.items():
                if key in support_gradients:
                    print("updating param", key)
                    for i, param in enumerate(component.parameters()):
                        grad = self.alpha_s * support_gradients[key][i] + self.alpha_q * query_gradients[key][i]
                        param.grad = grad / float(len(batch_tasks))
        
            for component in self.include_components:
                getattr(self, component + '_optimizer').step()
                getattr(self, component + '_optimizer').zero_grad()

        del support_gradients, query_gradients
        cleanup() 
        print("Complete forward pass")
        
        return query_accuracies