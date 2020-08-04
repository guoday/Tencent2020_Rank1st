import pandas as pd
import numpy as np

def merge_files():
    #合并点击记录
    print("merge click files...")
    click_df=pd.read_csv("data/train_preliminary/click_log.csv")
    click_df=click_df.append(pd.read_csv("data/train_semi_final/click_log.csv"))
    click_df=click_df.append(pd.read_csv("data/test/click_log.csv"))
    click_df=click_df.sort_values(by=["time"]).drop_duplicates()   
    
    #合并广告信息
    print("merge ad files...")
    ad_df=pd.read_csv("data/train_preliminary/ad.csv")
    ad_df=ad_df.append(pd.read_csv("data/train_semi_final/ad.csv"))
    ad_df=ad_df.append(pd.read_csv("data/test/ad.csv"))
    ad_df=ad_df.drop_duplicates() 
    
    #合并用户信息
    print("merge user files...")
    train_user=pd.read_csv("data/train_preliminary/user.csv")
    train_user=train_user.append(pd.read_csv("data/train_semi_final/user.csv"))
    train_user=train_user.reset_index(drop=True)
    train_user['age']=train_user['age']-1
    train_user['gender']=train_user['gender']-1
    test_user=pd.read_csv("data/test/click_log.csv").drop_duplicates('user_id')[['user_id']].reset_index(drop=True)
    test_user=test_user.sort_values(by='user_id').reset_index(drop=True)
    test_user['age']=-1
    test_user['gender']=-1

    #合并点击，广告，用户信息
    print("merge all files...")
    click_df=click_df.merge(ad_df,on="creative_id",how='left')
    click_df=click_df.merge(train_user,on="user_id",how='left')
    click_df=click_df.fillna(-1)
    click_df=click_df.replace("\\N",-1)
    for f in click_df:
        click_df[f]=click_df[f].astype(int)
    for i in range(10):
        click_df['age_{}'.format(i)]=(click_df['age']==i).astype(np.int16) 
    for i in range(2):
        click_df['gender_{}'.format(i)]=(click_df['gender']==i).astype(np.int16) 
    
    
    return click_df,train_user,test_user


if __name__ == "__main__":
    click_df,train_user,test_user=merge_files() 
    #保存预处理文件
    print("preprocess done! saving data...")
    click_df.to_pickle("data/click.pkl")
    train_user.to_pickle("data/train_user.pkl")
    test_user.to_pickle("data/test_user.pkl")
