import os
import pandas as pd
import numpy as np
def submit_files(path):
    age_files=[]
    gender_files=[]
    files= os.listdir(path) #得到文件夹下的所有文件名称
    s = []
    for file in files: #遍历文件夹
         if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
            if 'submission_test_gender' in file:
                gender_files.append(os.path.join(path,file))
            elif 'submission_test_age' in file:
                age_files.append(os.path.join(path,file))
    return age_files,gender_files

age_files,gender_files=submit_files("submission")

print("Age Files:")
for f in age_files:
    print(f)
print("Gender Files:")
for f in gender_files:
    print(f)    
age_score=np.mean([float(x.split('_')[-1][:-4]) for x in age_files])
gender_score=np.mean([float(x.split('_')[-1][:-4]) for x in gender_files])
print(len(age_files),len(gender_files))
print(round(age_score,4),round(gender_score,4),round(age_score+gender_score,4))

age_dfs=[pd.read_csv(f)[['user_id']+['age_'+str(i) for i in range(10)]] for f in age_files]
age_df=pd.concat(age_dfs,0)
age_df=pd.DataFrame(age_df.groupby('user_id').mean()).sort_values('user_id').reset_index()
age_df['predicted_age']=np.argmax(age_df[['age_'+str(i) for i in range(10)]].values,-1)+1
print(age_df)

gender_dfs=[pd.read_csv(f)[['user_id']+['gender_'+str(i) for i in range(2)]] for f in gender_files]
gender_df=pd.concat(gender_dfs,0)
gender_df=pd.DataFrame(gender_df.groupby('user_id').mean()).sort_values('user_id').reset_index()
gender_df['predicted_gender']=np.argmax(gender_df[['gender_'+str(i) for i in range(2)]].values,-1)+1
print(gender_df)

df=age_df
df['predicted_gender']=gender_df['predicted_gender']
print(df)

df[['user_id','predicted_age','predicted_gender']].to_csv("submission.csv",index=False)
print(df[['predicted_age','predicted_gender']].mean())

