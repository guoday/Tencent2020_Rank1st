import argparse
import torch
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader,SequentialSampler, RandomSampler
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel)
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

    
class Model(nn.Module):
    def __init__(self,args):
        super(Model, self).__init__()
        args.out_size=len(args.dense_features)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.args=args

        #创建BERT模型，并且导入预训练模型
        config = RobertaConfig.from_pretrained(args.pretrained_model_path) 
        config.output_hidden_states=True
        args.hidden_size=config.hidden_size
        args.num_hidden_layers=config.num_hidden_layers
        self.text_layer=RobertaModel.from_pretrained(args.pretrained_model_path,config=config) 
        self.text_linear=nn.Linear(args.text_dim+args.vocab_dim_v1*len(args.text_features), args.hidden_size)
        logger.info("Load linear from %s",os.path.join(args.pretrained_model_path, "linear.bin"))
        self.text_linear.load_state_dict(torch.load(os.path.join(args.pretrained_model_path, "linear.bin")))           
        logger.info("Load embeddings from %s",os.path.join(args.pretrained_model_path, "embeddings.bin"))
        self.text_embeddings=nn.Embedding.from_pretrained(torch.load(os.path.join(args.pretrained_model_path, "embeddings.bin"))['weight'],freeze=True)             
        args.out_size+=args.hidden_size*2
        
        #创建fusion-layer模型，随机初始化
        config = RobertaConfig()        
        config.num_hidden_layers=4
        config.intermediate_size=2048
        config.hidden_size=512
        config.num_attention_heads=16
        config.vocab_size=5
        self.text_layer_1=RobertaModel(config=config)
        self.text_layer_1.apply(self._init_weights)
        self.text_linear_1=nn.Linear(args.text_dim_1+args.hidden_size, 512) 
        self.text_linear_1.apply(self._init_weights)  
        self.norm= nn.BatchNorm1d(args.text_dim_1+args.hidden_size)
        args.out_size+=1024    

        #创建分类器，随机初始化
        self.classifier=ClassificationHead(args)
        self.classifier.apply(self._init_weights)
        
        
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            
    def forward(self,dense_features,text_features,text_ids,text_masks,text_features_1,text_masks_1,labels=None):
        outputs=[]
        #获取浮点数，作为分类器的输入
        outputs.append(dense_features.float())         
        #获取BERT模型的hidden state，并且做max pooling和mean pooling作为分类器的输入
        text_masks=text_masks.float()
        text_embedding=self.text_embeddings(text_ids).view(text_ids.size(0),text_ids.size(1),-1)
        text_features=torch.cat((text_features.float(),text_embedding),-1)
        text_features=torch.relu(self.text_linear(self.dropout(text_features)))
        hidden_states=self.text_layer(inputs_embeds=text_features,attention_mask=text_masks)[0]
        embed_mean=(hidden_states*text_masks.unsqueeze(-1)).sum(1)/text_masks.sum(1).unsqueeze(-1)
        embed_mean=embed_mean.float()
        embed_max=hidden_states+(1-text_masks).unsqueeze(-1)*(-1e10)
        embed_max=embed_max.max(1)[0].float()
        outputs.append(embed_mean)
        outputs.append(embed_max)
        #获取fusion-layer的hidden state，并且做max pooling和mean pooling作为分类器的输入
        text_masks_1=text_masks_1.float()
        text_features_1=torch.cat((text_features_1.float(),hidden_states),-1)
        bs,le,dim=text_features_1.size()
        text_features_1=self.norm(text_features_1.view(-1,dim)).view(bs,le,dim)
        text_features_1=torch.relu(self.text_linear_1(text_features_1))
        hidden_states=self.text_layer_1(inputs_embeds=text_features_1,attention_mask=text_masks_1)[0]
        embed_mean=(hidden_states*text_masks_1.unsqueeze(-1)).sum(1)/text_masks_1.sum(1).unsqueeze(-1)
        embed_mean=embed_mean.float()
        embed_max=hidden_states+(1-text_masks_1).unsqueeze(-1)*(-1e10)
        embed_max=embed_max.max(1)[0].float()
        outputs.append(embed_mean)
        outputs.append(embed_max)            

        #将特征输入分类器，得到20分类的logits
        final_hidden_state=torch.cat(outputs,-1)
        logits=self.classifier(final_hidden_state)
        
        #返回loss或概率结果
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
        else:
            prob=torch.softmax(logits,-1)
            age_probs=prob.view(-1,10,2).sum(2)
            gender_probs=prob.view(-1,10,2).sum(1)
            return age_probs,gender_probs
        

            
class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, args):
        super().__init__()
        self.norm= nn.BatchNorm1d(args.out_size)
        self.dense = nn.Linear(args.out_size, args.linear_layer_size[0])
        self.norm_1= nn.BatchNorm1d(args.linear_layer_size[0])
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.dense_1 = nn.Linear(args.linear_layer_size[0], args.linear_layer_size[1])  
        self.norm_2= nn.BatchNorm1d(args.linear_layer_size[1])
        self.out_proj = nn.Linear(args.linear_layer_size[1], args.num_label)

    def forward(self, features, **kwargs):
        x = self.norm(features)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(self.norm_1(x))
        x = self.dropout(x)
        x = self.dense_1(x)
        x = torch.relu(self.norm_2(x))
        x = self.dropout(x)        
        x = self.out_proj(x)
        return x

