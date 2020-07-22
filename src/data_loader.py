import logging
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import gensim
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

    
class TextDataset(Dataset):
    def __init__(self, args,df):
        self.label=df['label'].values
        self.text_features=df[[x[1] for x in args.text_features]].values
        self.text_features_1=df[[x[1] for x in args.text_features_1]].values
        self.dense_features=df[args.dense_features].values
        self.embeddings_tables=[]
        for x in args.text_features:
            self.embeddings_tables.append(args.embeddings_tables[x[0]] if x[0] is not None else None)
        self.embeddings_tables_1=[]
        for x in args.text_features_1:
            self.embeddings_tables_1.append(args.embeddings_tables_1[x[0]] if x[0] is not None else None)            
        self.args=args
        self.df=df

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):  
        #标签信息
        label=self.label[i]
        #BERT的输入特征
        if len(self.args.text_features)==0:
            text_features=0
            text_masks=0
            text_ids=0
        else:
            text_features=np.zeros((self.args.max_len_text,self.args.text_dim))
            text_masks=np.zeros(self.args.max_len_text)
            text_ids=np.zeros((self.args.max_len_text,len(self.args.text_features)),dtype=np.int64)
            begin_dim=0
            for idx,(embed_table,x) in enumerate(zip(self.embeddings_tables,self.text_features[i])):
                end_dim=begin_dim+self.args.text_features[idx][-1]              
                for w_idx,word in enumerate(x.split()[:self.args.max_len_text]):
                    text_features[w_idx,begin_dim:end_dim]=embed_table[word]
                    text_masks[w_idx]=1
                    try:
                        text_ids[w_idx,idx]=self.args.vocab[self.args.text_features[idx][1],word]
                    except:
                        text_ids[w_idx,idx]=self.args.vocab['unk']
                begin_dim=end_dim
        #decoder的输入特征        
        if len(self.args.text_features_1)==0:
            text_features_1=0
            text_masks_1=0
        else:
            text_features_1=np.zeros((self.args.max_len_text,self.args.text_dim_1))
            text_masks_1=np.zeros(self.args.max_len_text)
            begin_dim=0
            for idx,(embed_table,x) in enumerate(zip(self.embeddings_tables_1,self.text_features_1[i])):
                end_dim=begin_dim+self.args.text_features_1[idx][-1]  
                if embed_table is not None:
                    for w_idx,word in enumerate(x.split()[:self.args.max_len_text]):
                        text_features_1[w_idx,begin_dim:end_dim]=embed_table[word]
                        text_masks_1[w_idx]=1
                else:
                    for w_idx,v in enumerate(x[:self.args.max_len_text]):
                        text_features_1[w_idx,begin_dim:end_dim]=v
                        text_masks_1[w_idx]=1                    
                begin_dim=end_dim
        #浮点数特征                 
        if len(self.args.dense_features)==0:
            dense_features=0
        else:
            dense_features=self.dense_features[i]

        return (
                torch.tensor(label),
                torch.tensor(dense_features),
                torch.tensor(text_features),
                torch.tensor(text_ids),
                torch.tensor(text_masks),
                torch.tensor(text_features_1),
                torch.tensor(text_masks_1),            
               )



