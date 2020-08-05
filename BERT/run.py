# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function
from collections import Counter
import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from model import Model
cpu_cont = 4
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
}

class TextDataset(Dataset):
    def __init__(self, args,df,embedding_table):
        self.text_features=[df[x[1]].values for x in args.text_features]
        self.embedding_table=embedding_table
        self.args=args
        self.vocab=[list(x) for x in args.vocab]

    def __len__(self):
        return len(self.text_features[0])

    def __getitem__(self, i):  
        text_features=np.zeros((self.args.block_size,self.args.text_dim))
        text_ids=np.zeros((self.args.block_size,len(self.args.text_features)),dtype=np.int64)
        text_masks=np.zeros(self.args.block_size) 
        text_label=np.zeros((self.args.block_size,len(self.args.text_features)),dtype=np.int64)-100
        begin_dim=0
        #选择20%的token进行掩码，其中80%设为[mask], 10%设为[UNK],10%随机选择
        for idx,x in enumerate(self.args.text_features):
            end_dim=begin_dim+x[2]
            for word_idx,word in enumerate(self.text_features[idx][i].split()[:self.args.block_size]):
                text_masks[word_idx]=1
                if random.random()<self.args.mlm_probability:
                    if word in self.args.vocab[idx]:
                        text_label[word_idx,idx]=self.args.vocab[idx][word]
                    else:
                        text_label[word_idx,idx]=0
                    if random.random()<0.8:
                        text_ids[word_idx,idx]=self.args.vocab_dic['mask']
                    elif random.random()<0.5:
                        text_features[word_idx,begin_dim:end_dim]=self.embedding_table[idx][word]
                        try:
                            text_ids[word_idx,idx]=self.args.vocab_dic[(x[1],word)]
                        except:
                            text_ids[word_idx,idx]=self.args.vocab_dic['unk']
                    else:
                        while True:
                            random_word=random.sample(self.vocab[idx],1)[0]
                            if random_word!=word:
                                break
                        text_features[word_idx,begin_dim:end_dim]=self.embedding_table[idx][random_word] 
                        try:
                            text_ids[word_idx,idx]=self.args.vocab_dic[(x[1],random_word)]
                        except:
                            text_ids[word_idx,idx]=self.args.vocab_dic['unk']
                else:
                    try:
                        text_ids[word_idx,idx]=self.args.vocab_dic[(x[1],word)]
                    except:
                        text_ids[word_idx,idx]=self.args.vocab_dic['unk']
                    text_features[word_idx,begin_dim:end_dim]=self.embedding_table[idx][word]
            begin_dim=end_dim 
        return  torch.tensor(text_features),torch.tensor(text_ids),torch.tensor(text_masks),torch.tensor(text_label)
    
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def train(args, train_dataset,dev_dataset, model):
    #设置dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size,num_workers=4)
    t_total = args.max_steps
    args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1

    #设置优化器
    model.to(args.device)
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()     
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    if os.path.exists(scheduler_last):
        scheduler.load_state_dict(torch.load(scheduler_last, map_location="cpu"))
    if os.path.exists(optimizer_last):
        optimizer.load_state_dict(torch.load(optimizer_last, map_location="cpu"))    
    if args.local_rank == 0:
        torch.distributed.barrier()         
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # 多GPU设置
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

 
    # 训练
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset)* (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = args.start_step
    tr_loss, logging_loss,avg_loss,tr_nb = 0.0, 0.0,0.0,0
    model.zero_grad()
    set_seed(args)
 
    for idx in range(args.start_epoch, int(args.num_train_epochs)): 
        for step, batch in enumerate(train_dataloader):
            inputs,inputs_ids,masks,labels = [x.to(args.device) for x in batch]   
            model.train()
            loss = model(inputs,inputs_ids,masks,labels)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            
   
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()  
                global_step += 1
                output_flag=True
                avg_loss=round(np.exp((tr_loss - logging_loss) /(global_step- tr_nb)),4)
                if global_step %100 == 0:
                    logger.info("  steps: %s  ppl: %s", global_step, round(avg_loss,5))
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    logging_loss = tr_loss
                    tr_nb=global_step

                #验证
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    if args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, dev_dataset)
                        for key, value in results.items():
                            logger.info("  %s = %s", key, round(value,4))                    
                        # Save model checkpoint
                        output_dir = os.path.join(args.output_dir, '{}-{}-{}'.format(checkpoint_prefix, global_step,round(results['perplexity'],4)))

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    #保存模型
                    model_to_save = model.module.encoder if hasattr(model,'module') else model.encoder  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    logger.info("Saving model checkpoint to %s", output_dir)
                    
                    logger.info("Saving linear to %s",os.path.join(args.output_dir, "linear.bin"))  
                    model_to_save_linear = model.module.text_linear if hasattr(model, 'module') else model.text_linear
                    torch.save(model_to_save_linear.state_dict(), os.path.join(output_dir, "linear.bin"))
                    logger.info("Saving embeddings to %s",os.path.join(args.output_dir, "embeddings.bin"))  
                    model_to_save_embeddings = model.module.text_embeddings if hasattr(model, 'module') else model.text_embeddings
                    torch.save(model_to_save_embeddings.state_dict(), os.path.join(output_dir, "embeddings.bin"))
                    
                    
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save.save_pretrained(last_output_dir)
                    logger.info("Saving linear to %s",os.path.join(last_output_dir, "linear.bin"))  
                    model_to_save_linear = model.module.text_linear if hasattr(model, 'module') else model.text_linear
                    torch.save(model_to_save_linear.state_dict(), os.path.join(last_output_dir, "linear.bin"))
                    logger.info("Saving embeddings to %s",os.path.join(last_output_dir, "embeddings.bin"))  
                    model_to_save_embeddings = model.module.text_embeddings if hasattr(model, 'module') else model.text_embeddings
                    torch.save(model_to_save_embeddings.state_dict(), os.path.join(last_output_dir, "embeddings.bin"))
                    logger.info("Saving model to %s",os.path.join(last_output_dir, "model.bin"))  
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), os.path.join(last_output_dir, "model.bin"))


                    idx_file = os.path.join(last_output_dir, 'idx_file.txt')
                    with open(idx_file, 'w', encoding='utf-8') as idxf:
                        idxf.write(str( idx) + '\n')
                    torch.save(optimizer.state_dict(), os.path.join(last_output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(last_output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", last_output_dir)

                    step_file = os.path.join(last_output_dir, 'step_file.txt')
                    with open(step_file, 'w', encoding='utf-8') as stepf:
                        stepf.write(str(global_step) + '\n')
            if args.max_steps > 0 and global_step > args.max_steps:
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            break



def evaluate(args,  model, eval_dataset):
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size,num_workers=4)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()      
    for batch in eval_dataloader:
        inputs,inputs_ids,masks,labels = [x.to(args.device) for x in batch]       
        with torch.no_grad():
            lm_loss = model(inputs,inputs_ids,masks,labels)
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {
        "perplexity": float(perplexity)
    }
    
    return result


        
        
        
def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--dfg_size", default=64, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")    
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument('--log_file', type=str, default='')
    parser.add_argument('--tensorboard_dir', type=str)
    parser.add_argument('--lang', type=str)
    parser.add_argument('--pretrain', type=str, default='')
    args = parser.parse_args()
    pool = None
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # 设置log信息
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)


    # 设置随机种子
    set_seed(args)

    # 判断是否有checkpoint，从而继续预训练
    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]


    base_path="../data"
    text_features=[      
    [base_path+"/sequence_text_user_id_product_id.128d",'sequence_text_user_id_product_id',128,True],
    [base_path+"/sequence_text_user_id_ad_id.128d",'sequence_text_user_id_ad_id',128,True],
    [base_path+"/sequence_text_user_id_creative_id.128d",'sequence_text_user_id_creative_id',128,True],
    [base_path+"/sequence_text_user_id_advertiser_id.128d",'sequence_text_user_id_advertiser_id',128,True],
    [base_path+"/sequence_text_user_id_industry.128d",'sequence_text_user_id_industry',128,True],
    [base_path+"/sequence_text_user_id_product_category.128d",'sequence_text_user_id_product_category',128,True],
    [base_path+"/sequence_text_user_id_time.128d",'sequence_text_user_id_time',128,True],
    [base_path+"/sequence_text_user_id_click_times.128d",'sequence_text_user_id_click_times',128,True],
    ]

    #读取训练数据
    train_df=pd.read_pickle(os.path.join(base_path,'train_user.pkl'))
    test_df=pd.read_pickle(os.path.join(base_path,'test_user.pkl'))
    dev_data=train_df.iloc[-10000:]
    train_data=train_df.iloc[:-10000].append(test_df)  
    
    #创建输入端的词表，每个域最多保留10w个id
    try:
        dic=pickle.load(open(os.path.join(args.output_dir, 'vocab.pkl'),'rb'))
    except:
        dic={}
        dic['pad']=0
        dic['mask']=1
        dic['unk']=2    
        for feature in text_features: 
            conter=Counter()
            for item in train_df[feature[1]].values:
                for word in item.split():
                    try:
                        conter[(feature[1],word)]+=1
                    except:
                        conter[(feature[1],word)]=1 
            most_common=conter.most_common(100000)   
            cont=0
            for x in most_common:
                if x[1]>5:
                    dic[x[0]]=len(dic)
                    cont+=1
                    if cont<10:
                        print(x[0],dic[x[0]])
            print(cont)
     
    #读取或重新创建BERT   
    if args.model_name_or_path is not None:
        config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                              cache_dir=args.cache_dir if args.cache_dir else None)        
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)   
        args.text_dim=config.hidden_size
    else:
        config = RobertaConfig()        
        config.num_hidden_layers=12
        config.hidden_size=512
        config.intermediate_size=config.hidden_size*4
        config.num_attention_heads=16
        config.vocab_size=5    
        model = model_class(config)
        config.vocab_size_v1=len(dic)
        config.vocab_dim_v1=64
        logger.info("%s",config)   
    logger.info("Training/evaluation parameters %s", args)

    #保存输入端词表
    args.vocab_dic=dic
    pickle.dump(dic,open(os.path.join(args.output_dir, 'vocab.pkl'),'wb'))

    #读取word embedding
    import gensim
    embedding_table=[]
    for x in text_features:
        print(x)
        embedding_table.append(pickle.load(open(x[0],'rb')))

    #创建输出端词表，每个域最多保留10w个id
    vocab=[]
    for feature in text_features:
        conter=Counter()
        for item in train_data[feature[1]].values:
            for word in item.split():
                try:
                    conter[word]+=1
                except:
                    conter[word]=1
        most_common=conter.most_common(100000)
        dic={}
        for idx,x in enumerate(most_common):
            dic[x[0]]=idx+1    
        vocab.append(dic)
     
    #设置参数  
    args.vocab_size_v1=config.vocab_size_v1
    args.vocab_dim_v1=config.vocab_dim_v1
    args.vocab=vocab
    args.text_dim=sum([x[2] for x in text_features])
    args.text_features=text_features
    train_dataset=TextDataset(args,train_data,embedding_table)
    dev_dataset=TextDataset(args,dev_data,embedding_table)
    args.vocab_size=[len(x)+1 for x in vocab]
    #创建模型
    model=Model(model,config,args)   
    #如果有checkpoint，读取checkpoint
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        logger.info("Load model from %s",os.path.join(checkpoint_last, "model.bin"))
        model.load_state_dict(torch.load(os.path.join(checkpoint_last, "model.bin"))) 

    #训练      
    train(args, train_dataset,dev_dataset, model)


if __name__ == "__main__":
    main()








