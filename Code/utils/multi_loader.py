# Methods supporting multitask meta learners 

from loader import * 

def meta_quick_eval(model, meta_loader, learning_rate, num_epochs=1, batch_size=10,
                   device=torch.device('cpu'), p_threshold=None, loss_weights=None,
                   report_interval=5, report_only=True ):
    """
    Train, evaluate and collect results in standard formats
    
    Equivalent to quick_eval function for binary trainer
    """
    assert len(meta_loader) == 1, "Loader has more than 1 domain!"
    support, query, task_config, domains = meta_loader[0]
    train_loader = DataLoader(support, sampler=RandomSampler(support), batch_size=batch_size )
    test_loader = DataLoader(query, sampler=None, batch_size=batch_size )
    task_config = meta_loader[0][2]
    
    fast_model = model.create_fast_model(model.components, task_config)
    fast_model = move_to_device(fast_model, self.device)

    fast_optimizer = AdamW([{'params': fast_model[c].parameters()} for c in fast_model], lr=learning_rate)
    
    # Training
    for epoch in range(num_epochs): 
        train_losses, train_acc, train_f1 = model.train_fast_model(fast_model, fast_optimizer, train_loader, task_config, 
                                  is_training=True, optimize_params=True, device=device, 
                                  p_threshold =p_threshold if p_threshold is not None else model.p_threshold,
                                  loss_weights=loss_weights if loss_weights is not None else model.loss_weights
                                  )
        train_losses = [round(loss,4) for i, loss in enumerate(train_losses) if i%report_interval == 0] 
        train_acc = [acc for i, acc in enumerate(train_acc) if i%report_interval == 0] 
        train_f1  = [f1 for i, f1 in enumerate(train_f1) if i%report_interval == 0] 
        print("Epoch {}:\nTrain losses: {}\nTrain Accuracies:{}\nTrain F1:{}".format(epoch, train_losses, train_acc, train_f1))
        
    # Evaluate to collect results
    preds, labels = model.eval_fast_model(fast_model, fast_optimizer, test_loader,
                                          task_config,
                                          device=device,
                                          p_threshold =p_threshold if p_threshold is not None else model.p_threshold)
    
    report = dict()
    for i, (task, config) in enumerate(task_config.items()):
        report['task_' + str(i)] = classification_report(y_true=labels[i], y_pred=preds[i], output_dict=True, zero_division=0)
        if config['problem_type'] == 'multi_label_classification':
            acc = accuracy_score(y_true=labels[i], y_pred=preds[i])
            report['task_' + str(i)]['accuracy'] = acc
    
    if report_only: 
        del preds, labels
        gc.collect() 
        return fast_model, report
    
    return fast_model, preds, labels, report


def multi_meta_train(meta_learner, meta_args, df_train, tokenizer, max_len,
                   train_num_task, train_k_support, train_k_query, label_vars, label_config:dict,
                   test_num_task:int, test_datasets: dict, step_interval:int=5,
                   val_batch_size=10, val_learning_rate = 2e-5, val_num_train_epoch = 3, val_report_interval = 10,
                   skip_validation=False, return_best_statedict=False, train_verbose=False, seed=123):
    """
    Train full loops with meta-learner model
    
    Compatible with domains with multiple classes and/or multiple classification tasks (hate speech + target)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set_seed(seed)
    acc_all_train = []
#     idx_array = []
    validation_f1 =  {domain: {k: [] for k in v } for domain, v in test_datasets.items()}
    validation_acc = deepcopy(validation_f1)
    best_state_dict = deepcopy(validation_f1)
    output_dict = {seed: {'best_statedict': None}}
    
    # Meta train 
    global_step = 0
    # set_seed(seed)
    start = time.time()
    for epoch in range(meta_args.meta_epoch):
        print("\n+++++ Meta Epoch {} +++++".format(epoch))
        train = MetaLoader(df_train, num_task = train_num_task,  max_len=max_len,
                           k_support=train_k_support, k_query=train_k_query, label_config=label_config,
                           tokenizer = tokenizer, batch_mode='joint', verbose=False,
                           label_vars=label_vars, test_domains=None)
        db = create_meta_batches(train, batch_size=meta_args.outer_batch_size)
        last_step = meta_args.meta_epoch * (ceil(len(train) / meta_args.outer_batch_size))
    
        for step, task_batch in enumerate(db):
            acc = meta_learner(task_batch, is_training=True, verbose=train_verbose)
            acc_all_train.append(acc)
            
        
            if (not skip_validation and global_step % step_interval == 0) or (global_step == last_step - 1):
#                 idx_array.append(global_step)
                print('Step:', global_step, '\ttraining Acc:', acc)
                print("-----------------Meta Testing Mode-----------------")
                
                # compute results on each size of set K 
                for domain, test_dict in test_datasets.items(): 
                    print("---------- Evaluating on Domain {} ----------".format(domain.upper()))
                    for k_val, test_dataset in test_dict.items():
                        print("...Testing on k_val {}:".format(k_val))
                        set_seed(seed*5)
                        # create corresponding test set
                        test_metaloader = MetaLoader(test_dataset, num_task=1,
                                                     k_support=k_val, k_query=k_val, tokenizer=tokenizer, 
                                                     batch_mode = 'fixed', label_vars=label_vars, 
                                                     label_config = label_config, max_len=max_len)

                        
                        # - switch uni to save space if need be 
                        fast_model, report = meta_quick_eval(meta_learner, test_metaloader,
                                                             batch_size = val_batch_size,
                                                             learning_rate=val_learning_rate,
                                                             num_epochs = val_num_train_epoch, 
                                                             report_interval= val_report_interval, 
                                                             device=device)
                        # compile results for all tasks 
                        test_f1, test_acc = [], []
                        for task, result in report.items():
                            test_f1.append(round(result['macro avg']['f1-score'],5))
                            test_acc.append(round(result['accuracy'],3))
                        
                      
                        validation_f1[domain][k_val].append(test_f1)
                        validation_acc[domain][k_val].append(test_acc)

                        # update if better results 
                        val_f1_means = [np.mean(v) for v in validation_f1[domain][k_val]]
                        if (not validation_f1[domain][k_val]) or (np.mean(test_f1) >= max(val_f1_means)): 
                            state_dict = deepcopy(meta_learner.state_dict())
                            statedict_to_cpu(state_dict)
                            best_state_dict[domain][k_val] = state_dict
                            print("Update best params at step {} for k_val {}".format(global_step, k_val))
                            
                        
                        del test_metaloader, fast_model, report, test_f1, test_acc
                        gc.collect()
                        torch.cuda.empty_cache()
                
            global_step += 1
                
    print(acc_all_train)
    print("DURATION:", time.time() - start)
    print("GLOBAL STEP:", global_step, "LAST STEP", last_step)

    # compile output
    output_dict[seed]['train_acc'] = acc_all_train 
#     output_dict[seed]['idx_array'] = idx_array
    output_dict[seed]['val_f1'] = validation_f1
    output_dict[seed]['val_acc'] = validation_acc
    if return_best_statedict:
        output_dict[seed]['best_statedict'] = best_state_dict

    return output_dict