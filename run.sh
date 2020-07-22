sudo pip install transformers==2.8.0 pandas gensim scikit-learn filelock gdown numpy
pip install transformers==2.8.0 pandas gensim scikit-learn filelock gdown numpy

#数据下载
gdown https://drive.google.com/uc?id=15onAobxlim_uRUNWSMQuK6VxDsmGTtp4
unzip data.zip 
rm data.zip

#数据预处理
python src/preprocess.py

#特征提取
python src/extract_features.py

#下载Word2Vector权重
gdown https://drive.google.com/uc?id=1SUpukAeXR5Ymyf3wH3SRNdQ3Hl2HazQa
unzip w2v.zip 
cp w2v/* data/
rm -r w2v*

#下载BERT-base权重
gdown https://drive.google.com/uc?id=1ToAJwl_oRAeRNyYF_FK0B2APVXlPFTlq
unzip bert-base.zip
mv bert-base BERT/
rm bert-base.zip

#训练模型
mkdir saved_models
mkdir saved_models/log
for((i=0;i<5;i++));  
do  
  python run.py \
      --kfold=5 \
      --index=$i \
      --train_batch_size=256 \
      --eval_steps=5000 \
      --max_len_text=128 \
      --epoch=5 \
      --lr=1e-4 \
      --output_path=saved_models \
      --pretrained_model_path=BERT/bert-base \
      --eval_batch_size=512 2>&1 | tee saved_models/log/$i.txt
done  

#合并结果
python src/merge_submission.py
