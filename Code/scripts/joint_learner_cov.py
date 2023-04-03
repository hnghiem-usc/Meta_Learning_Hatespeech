# DEFITION OF CLASS DS_LEARNER 
# VERSION 1: REGULARIZE BY MINIMIZING COVARIANCE OF DS AND DI FEATURES
# INCLUDES DS WASSERSTEIN DISTANCE
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
from special_learner import *
# from ds_learner_cov import DS_Learner_Cov
# from ds_learner_orth import DS_Learner_Orth


model_lookup = {'roberta':RobertaModel, 'bert':BertModel}

class Joint_Learner_Cov(Special_Learner):
    def __init__(self, args, model_lookup:dict = model_lookup, verbose=True): 
        super().__init__(args, model_lookup, verbose=False)
        self.class_name = 'Joint_Learner_Cov'
        # num_labels might vary
        for key, attr in args.__dict__.items():
            setattr(self, key, attr)
        del self.extractor
        self.embedding = model_lookup[self.embedding_class].from_pretrained(self.embedding_dir, add_pooling_layer=False, output_hidden_states=True)
        sequence_length = args.sequence_length
        hidden_size     = self.embedding.config.hidden_size
        self.components = collections.OrderedDict({'embedding': self.embedding})
        
        base_dropout = (self.embedding.config.classifier_dropout if self.embedding.config.classifier_dropout \
                        is not None else self.embedding.config.hidden_dropout_prob )
        # Feature Extractor
        conv = nn.Conv2d(in_channels=args.in_channels,
                              out_channels=args.out_channels, 
                              kernel_size=(args.conv_kernel, hidden_size), 
                              padding=args.conv_padding)
        relu = nn.ReLU()
        pool = nn.MaxPool2d(kernel_size=args.pool_kernel, stride=args.pool_stride)
        dropout = nn.Dropout(0.1)
        self.di_extractor = nn.Sequential(collections.OrderedDict([('conv', conv), 
                                                                ('relu', relu), 
                                                                ('dropout', dropout), 
                                                                ('pool', pool)
                                                                ]))
        self.ds_extractor = deepcopy(self.di_extractor)
        self.components['di_extractor'] = self.di_extractor
        self.components['ds_extractor'] = self.ds_extractor

        ops_dict = {'conv2d': ((args.conv_kernel, hidden_size), args.conv_padding), 
                     'maxpool2d':(args.pool_kernel, args.pool_stride)
                    }
        self.extractor_output_shape = get_extractor_output_shape((0, args.in_channels, sequence_length, hidden_size), ops_dict)
        # Domain Critic
        self.critic = nn.Sequential(nn.Linear(prod(self.extractor_output_shape[1:]), self.critic_itrm_shape),
                                    nn.ReLU(),
                                    nn.Linear(self.critic_itrm_shape, self.critic_output_shape),
                                    nn.ReLU(),
                                    nn.Linear(self.critic_output_shape, 1))
        self.critic_optim = AdamW(self.critic.parameters(), lr=self.critic_lr) 
        self.components['critic'] = self.critic
        
        # Joint Discriminator 
        self.discriminator = nn.Sequential(nn.Linear(2*prod(self.extractor_output_shape[1:]), self.critic_itrm_shape),
                                    nn.ReLU(),
                                    nn.Linear(self.critic_itrm_shape, self.critic_output_shape),
                                    nn.ReLU(),
                                    nn.Linear(self.critic_output_shape, 1))
        self.discriminator_optimizer = AdamW(self.discriminator.parameters(), lr=self.discriminator_lr) 
        self.components['discriminator'] = self.discriminator
        
        # remember learning rates
        self.lr_dict = {}
        self.fast_lr_dict = {}
        for k in self.components:
            if hasattr(self, k + '_lr'):
                self.lr_dict[k + '_lr'] = getattr(self, k + '_lr')
                self.fast_lr_dict[k + '_lr'] = getattr(self, k + '_lr')
            else:
                if 'extractor' in k:
                    self.lr_dict[k + '_lr'] = self.extractor_lr 
                    self.fast_lr_dict[k + '_lr'] = self.lr_dict[k + '_lr'] 
                
        # Classifier
        for i in range(self.num_classifier): 
            cls_dict = collections.OrderedDict()
            cls_dict['fc']     = nn.Linear(2*prod(self.extractor_output_shape[1:]), hidden_size) # concat DI+DS data
            cls_dict['dropout']= nn.Dropout(base_dropout)
            cls_dict['cls']    = nn.Linear(hidden_size, 1)
            cls_name = 'classifier_' +str(i)
            setattr(self, cls_name, nn.Sequential(cls_dict))
            self.components[cls_name] = getattr(self, cls_name)
            self.lr_dict[cls_name + '_lr'] = getattr(self, cls_name + '_lr') if hasattr(self, cls_name + '_lr') else self.classifier_lr
            self.fast_lr_dict[cls_name + '_lr'] = getattr(self, 'fast_' + cls_name + '_lr') if hasattr(self, 'fast_' + cls_name + '_lr') else self.fast_classifier_lr
            setattr(self, cls_name+'_optimizer', AdamW(getattr(self, 'classifier_' +str(i)).parameters(), lr=self.lr_dict[cls_name + '_lr']))
        
        self.include_components = list(set(self.components.keys()) - set(self.exclude_components))
        if verbose:
            print("Model Class:", self.class_name)
            print("lr_dict:{}\nfast_lr_dict:{}".format(self.lr_dict, self.fast_lr_dict))
        
   
    @staticmethod
    def train_fast_model(model, optimizer, source_loader, target_loader, task_config, device, loss_weights:list, 
                            is_training=True, optimize_params=True, switch_to_eval=False, 
                            p_threshold = 0.5, wd_inv_clf=0.1, wd_spc_clf=None,
                            w_discrim=0.5, w_disent=0.1,
                            base_gradients=None,
                            step_count=None, 
                            scheduler=None,
                            return_extractor_features=False):
        loss_array = [] 
        discrim_loss_array = []
        disentangle_loss_array =[]

        accuracy_array = [[] for i in task_config]
        f1_array = [[] for i in task_config]
        di_source_features = [] 
        ds_source_features = [] 
        wasserstein_distance_di = 0
        wasserstein_distance_ds = 0

        if target_loader: 
            zip_loader = zip(source_loader, target_loader)
        else: 
            zip_loader = source_loader
        model = {k: c.train() for k, c in model.items()}

        if switch_to_eval: 
            model = {k: c.eval() for k, c in model.items()}    
        set_requires_grad(model['critic'], requires_grad=False)

        for step,  zip_batch in enumerate(zip_loader):
            # hs_ds, hs_di, ht_ds, ht_di: features from source/target if the domain specific/invariant extractors respectively
            temp_gradients = None
            if target_loader: 
                s_batch, t_batch =  zip_batch
                ht_ds, _  = Special_Learner.forward_extractor(t_batch, model['embedding'], model['ds_extractor'], device )
                ht_di, _  = Special_Learner.forward_extractor(t_batch, model['embedding'], model['di_extractor'], device )
            else: 
                s_batch = zip_batch
            hs_ds, labels      = Special_Learner.forward_extractor(s_batch, model['embedding'], model['ds_extractor'], device )
            hs_di, _           = Special_Learner.forward_extractor(s_batch, model['embedding'], model['di_extractor'], device )
            ds_source_features.append(hs_ds)
            di_source_features.append(hs_di)

            # Disentanglement loss between domain invariant and specific Souce features 
            mean_hs = torch.mean(hs_ds, dim=0)
            mean_hi = torch.mean(hs_di, dim=0)
            hs_norm = hs_ds - mean_hs[None, :]
            hi_norm = hs_di - mean_hi[None, :]
            C = hs_norm[:, :, None] * hi_norm[:, None, :]

            target_cr = torch.zeros(C.shape[0], C.shape[1], C.shape[2]).to(device)
            disentangle_loss = nn.MSELoss()(C, target_cr)
            disentangle_loss_array.append(float(disentangle_loss.item())) 

            # Joint discriminator loss
            source_features = torch.cat([hs_di, hs_ds], dim=1) # invariant-specific in this order
            discrim_loss = 0.
            if target_loader: 
                target_features = torch.cat([ht_di, ht_ds], dim=1) 
                discrim_criterion = nn.BCEWithLogitsLoss()
                # source features: label = 0 , target_features: 1
                source_discrim_loss   = discrim_criterion(source_features, torch.zeros_like(source_features)) 
                target_discrim_loss   = discrim_criterion(target_features, torch.ones_like(target_features))
                discrim_loss = source_discrim_loss + target_discrim_loss
                discrim_loss_array.append(float(discrim_loss.item()))

            # calculate WD w/ target set 
            set_requires_grad(model['critic'], False) #distable critic grad
            if target_loader: 
                wasserstein_distance_di = model['critic'](hs_di).mean() - model['critic'](ht_di).mean()
            # print("Fast Training WD_I: {:6.4f}\t WD_S: {:6.4f}".format( wasserstein_distance_di, wasserstein_distance_ds))

            # Classifier loss 
            classifier = {k: v for k, v in model.items() if 'classifier' in k}
            # change learner class 
            losses, logits, labels = Special_Learner.forward_classifier(classifier, source_features, labels, task_config, device)
            loss_array.append(float(sum(losses).item()))

            #calculate results
            results = process_prediction(logits, labels, task_config, p_threshold, return_preds=False)
            [accuracy_array[i].append(acc) for i,acc in enumerate(results['acc'])] 
            [f1_array[i].append(f1) for i,f1 in enumerate(results['f1'])] 

            #accumulate gradients
            if is_training: 
                total_losses = sum([l * w for l, w in zip(losses, loss_weights)]) # classifier_loss 
                total_losses += (wd_inv_clf * wasserstein_distance_di + w_discrim*discrim_loss + w_disent*disentangle_loss)
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
            return (loss_array, discrim_loss_array, disentangle_loss_array ), accuracy_array, f1_array, (di_source_features, ds_source_features)
        return (loss_array, discrim_loss_array, disentangle_loss_array), accuracy_array, f1_array

    
    @staticmethod
    def eval_fast_model(model, data_loader, task_config, device, p_threshold=0.5,):
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
            input_inv, labels = Special_Learner.forward_extractor(batch, model['embedding'], model['di_extractor'], device, embedding_grad = False, extractor_grad = False )
            input_spc, _     =  Special_Learner.forward_extractor(batch, model['embedding'], model['ds_extractor'], device, embedding_grad = False, extractor_grad = False )
            
            classifier = {k: v for k,v in model.items() if 'classifier' in k}
            inputs = torch.cat([input_inv, input_spc], dim=1) # dspec-dinv order
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
        set_requires_grad(self.critic, True)
        print("CRITIC IS ON GPU? ",next(self.critic.parameters())[-1].is_cuda ) ##!!!

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
                with torch.no_grad(): # disable grad except for critic 
                    h_s, _ = self.forward_extractor(s_batch, self.embedding, self.di_extractor, self.device, embedding_grad=False, extractor_grad=False)
                    h_t, _ = self.forward_extractor(t_batch, self.embedding, self.di_extractor, self.device, embedding_grad=False, extractor_grad=False)
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
                (s_losses, s_discrim_losses, s_disent_losses), s_accuracies, s_f1  = self.train_fast_model(s_model, s_optimizer, support_loader, target_loader,
                                                                s_config,self.device, loss_weights=self.loss_weights, base_gradients=s_gradients, 
                                                                wd_inv_clf = self.wd_inv_clf, w_discrim=self.w_discrim, w_disent=self.w_disent)
                s_loss_mean     = get_average_report(s_losses, num_round=4)[0]
                s_discrim_loss_mean = get_average_report(s_discrim_losses, num_round=4)[0]
                s_disent_loss_mean  = get_average_report(s_disent_losses, num_round=4)[0]
                s_acc_mean, s_f1_mean = get_average_report(s_accuracies, s_f1)
                support_losses.append(s_loss_mean)
                [support_accuracies[i].append(acc) for i, acc in enumerate(s_acc_mean)]
                [support_f1[i].append(f1) for i, f1 in enumerate(s_f1_mean)]
                 
                # append data in arrays ! 
                if verbose:
                    print("Average support_losses: {}\t support_discrim_losses: {}\t support_disent_losses: {}\t support_acc:{}\t support_F1:{}".format(s_loss_mean, s_discrim_loss_mean, s_disent_loss_mean,
                                                                                                                        s_acc_mean, s_f1_mean))
                       
            ## next step: get accumulated gradient in the evaluatioon on query set
            q_gradients = {k: [] for k in q_model if k not in 'embedding'} 
            # show disentanglement loss for query set
            (q_losses, _ , q_dis_losses), q_accuracies, q_f1 = self.train_fast_model(q_model, None, query_loader, None, q_config, self.device,
                                                        is_training=True, optimize_params=False, switch_to_eval=True, 
                                                        base_gradients = q_gradients, loss_weights=self.loss_weights, wd_inv_clf=self.wd_inv_clf,
                                                        w_discrim=self.w_discrim, w_disent=self.w_disent) 
            q_loss_mean = get_average_report(q_losses, num_round=4)[0]
            q_dis_loss_mean = get_average_report(q_dis_losses, num_round=4)[0]
            q_acc_mean, q_f1_mean = get_average_report(q_accuracies, q_f1)
            [query_accuracies[i].append(acc) for i, acc in enumerate(q_acc_mean)]
            [query_f1[i].append(f1) for i, f1 in enumerate(q_f1_mean)]
            
            self.collect_gradients(support_gradients, s_gradients)
            self.collect_gradients(query_gradients, q_gradients)
            
            del s_gradients, s_losses, s_discrim_losses, s_disent_losses, s_accuracies, s_f1
            del q_gradients, q_losses, q_dis_losses, q_accuracies, q_f1
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











