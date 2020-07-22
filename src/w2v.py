import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm
import pickle
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)

def w2v(dfs,f,L=128):
    print("w2v",f)
    sentences=[]
    for df in dfs:
        for line in df[f].values:
            sentences.append(line.split())
    print("Sentence Num {}".format(len(sentences)))
    w2v=Word2Vec(sentences,size=L, window=8,min_count=1,sg=1,workers=32,iter=10)
    print("save w2v to {}".format(os.path.join('data',f+".{}d".format(L))))
    pickle.dump(w2v,open(os.path.join('data',f+".{}d".format(L)),'wb'))  

if __name__ == "__main__":
    train_df=pd.read_pickle('data/train_user.pkl')
    test_df=pd.read_pickle('data/test_user.pkl')
    #训练word2vector，维度为128
    w2v([train_df,test_df],'sequence_text_user_id_ad_id',L=128)
    w2v([train_df,test_df],'sequence_text_user_id_creative_id',L=128)
    w2v([train_df,test_df],'sequence_text_user_id_advertiser_id',L=128)
    w2v([train_df,test_df],'sequence_text_user_id_product_id',L=128)
    w2v([train_df,test_df],'sequence_text_user_id_industry',L=128)
    w2v([train_df,test_df],'sequence_text_user_id_product_category',L=128)
    w2v([train_df,test_df],'sequence_text_user_id_time',L=128)
    w2v([train_df,test_df],'sequence_text_user_id_click_times',L=128)


