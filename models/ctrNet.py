import argparse
import torch
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from models.model import Model
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader,SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel)
import random
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
#设置随机种子  
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)            


class ctrNet(nn.Module):
    def __init__(self,args):
        super(ctrNet, self).__init__()
        #设置GPU和创建模型
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
        args.device = device
        logger.info(" device: %s, n_gpu: %s",device, args.n_gpu)
        model=Model(args)  
        model.to(args.device)   
        self.model=model
        self.args=args
        set_seed(args)
        
    def train(self,train_dataset,dev_dataset=None):
        args=self.args
        #设置dataloader
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
        args.max_steps=args.epoch*len( train_dataloader)
        args.save_steps=len( train_dataloader)//10
        args.warmup_steps=len( train_dataloader)
        args.logging_steps=len( train_dataloader)
        args.num_train_epochs=args.epoch 
        #设置优化器
        optimizer = AdamW(self.model.parameters(), lr=args.lr, eps=1e-8,weight_decay=0.08)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(len(train_dataloader)*args.num_train_epochs*0.2),                                                        num_training_steps=int(len(train_dataloader)*args.num_train_epochs))    
        #多GPU设置
        if args.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model) 
        model=self.model    
        #开始训练
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_dataset))
        logger.info("  Num Epochs = %d", args.num_train_epochs)
        if args.n_gpu!=0:
            logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size//args.n_gpu)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                    args.train_batch_size)
        logger.info("  Total optimization steps = %d", args.max_steps)  

        global_step = 0
        tr_loss, best_age_acc,best_gender_acc,avg_loss,tr_nb = 0.0,0.0, 0.0,0.0,0.0
        model.zero_grad()  
        patience=0
        for idx in range(args.num_train_epochs):     
            tr_num=0
            train_loss=0
            for step, batch in enumerate(train_dataloader):
                #forward和backward
                labels,dense_features,text_features,text_ids,text_masks,text_features_1,text_masks_1=(x.to(args.device) for x in batch)  
                del batch
                model.train()
                loss = model(dense_features,text_features,text_ids,text_masks,text_features_1,text_masks_1,labels)
                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training 
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)         
                tr_loss += loss.item()
                tr_num+=1
                train_loss+=loss.item()
                #输出log
                if avg_loss==0:
                    avg_loss=tr_loss
                avg_loss=round(train_loss/tr_num,5)
                if (step+1) % args.display_steps == 0:
                    logger.info("  epoch {} step {} loss {}".format(idx,step+1,avg_loss))
                #update梯度
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1

                #测试验证结果
                if (step+1) % args.eval_steps == 0 and dev_dataset is not None:
                    #输出验证集性别和年龄的概率
                    age_probs,gender_probs = self.infer(dev_dataset)
                    #输出性别和年龄的loss和acc
                    age_results= self.eval(dev_dataset.df['age'].values,age_probs)
                    gender_results= self.eval(dev_dataset.df['gender'].values,gender_probs)
                    results={}
                    results['eval_age_loss']=age_results['eval_loss']
                    results['eval_gender_loss']=gender_results['eval_loss']
                    results['eval_age_acc']=age_results['eval_acc']
                    results['eval_gender_acc']=gender_results['eval_acc']  
                    #打印结果                  
                    for key, value in results.items():
                        logger.info("  %s = %s", key, round(value,4))                    
                    #保存最好的年龄结果和模型
                    if results['eval_age_acc']>best_age_acc:
                        best_age_acc=results['eval_age_acc']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best age acc:%s",round(best_age_acc,4))
                        logger.info("  "+"*"*20)                          
                        try:
                            os.system("mkdir -p {}".format(args.output_dir))
                        except:
                            pass
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, "pytorch_model_{}.bin".format('age'))
                        torch.save(model_to_save.state_dict(), output_model_file)
                    #保存最好的性别结果和模型
                    if results['eval_gender_acc']>best_gender_acc:
                        best_gender_acc=results['eval_gender_acc']
                        logger.info("  "+"*"*20)  
                        logger.info("  Best gender acc:%s",round(best_gender_acc,4))
                        logger.info("  "+"*"*20)                          
                        try:
                            os.system("mkdir -p {}".format(args.output_dir))
                        except:
                            pass
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(args.output_dir, "pytorch_model_{}.bin".format('gender'))
                        torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("  best_acc = %s",round(best_age_acc+best_gender_acc,4)) 
                        
            #一个epoch结束后，测试验证集结果            
            if dev_dataset is not None:
                #输出验证集性别和年龄的概率
                age_probs,gender_probs = self.infer(dev_dataset)
                #输出性别和年龄的loss和acc
                age_results= self.eval(dev_dataset.df['age'].values,age_probs)
                gender_results= self.eval(dev_dataset.df['gender'].values,gender_probs)
                results={}
                results['eval_age_loss']=age_results['eval_loss']
                results['eval_gender_loss']=gender_results['eval_loss']
                results['eval_age_acc']=age_results['eval_acc']
                results['eval_gender_acc']=gender_results['eval_acc']  
                #打印结果                    
                for key, value in results.items():
                    logger.info("  %s = %s", key, round(value,4))                    
                #保存最好的年龄结果和模型
                if results['eval_age_acc']>best_age_acc:
                    best_age_acc=results['eval_age_acc']
                    logger.info("  "+"*"*20)  
                    logger.info("  Best age acc:%s",round(best_age_acc,4))
                    logger.info("  "+"*"*20)                          
                    try:
                        os.system("mkdir -p {}".format(args.output_dir))
                    except:
                        pass
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, "pytorch_model_{}.bin".format('age'))
                    torch.save(model_to_save.state_dict(), output_model_file)
                #保存最好的性别结果和模型
                if results['eval_gender_acc']>best_gender_acc:
                    best_gender_acc=results['eval_gender_acc']
                    logger.info("  "+"*"*20)  
                    logger.info("  Best gender acc:%s",round(best_gender_acc,4))
                    logger.info("  "+"*"*20)                          
                    try:
                        os.system("mkdir -p {}".format(args.output_dir))
                    except:
                        pass
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    output_model_file = os.path.join(args.output_dir, "pytorch_model_{}.bin".format('gender'))
                    torch.save(model_to_save.state_dict(), output_model_file)
                logger.info("  best_acc = %s",round(best_age_acc+best_gender_acc,4)) 


    def infer(self,eval_dataset):
        #预测年龄和性别的概率分布
        args=self.args
        model=self.model        
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)
        eval_loss = 0.0
        nb_eval_steps = 0
        age_probs=[]
        gender_probs=[]
        model.eval()      
        for batch in eval_dataloader:       
            _,dense_features,text_features,text_ids,text_masks,text_features_1,text_masks_1=(x.to(args.device) for x in batch) 
            with torch.no_grad():
                probs_1,probs_2 = model(dense_features,text_features,text_ids,text_masks,text_features_1,text_masks_1)  
            age_probs.append(probs_1.cpu().numpy()) 
            gender_probs.append(probs_2.cpu().numpy()) 

        age_probs=np.concatenate(age_probs,0)
        gender_probs=np.concatenate(gender_probs,0)
        return age_probs,gender_probs
    
    def eval(self,labels,preds):
        #求出loss和acc
        results={}
        results['eval_acc']=np.mean(labels==np.argmax(preds,-1))
        from sklearn.metrics import log_loss
        results['eval_loss']=log_loss(labels,preds)         
        return results
    
    def reload(self,label):
        #读取在验证集结果最好的模型
        model=self.model
        args=self.args
        args.load_model_path=os.path.join(args.output_dir, "pytorch_model_{}.bin".format(label))
        logger.info("Load model from %s",args.load_model_path)
        model_to_load = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        model_to_load.load_state_dict(torch.load(args.load_model_path))        
        
