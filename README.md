## 赛题介绍-广告受众基础属性预估

比赛将为参赛者提供一组用户在长度为 91 天(3 个月)的时间窗口内的广告点击历史记录作为训练数据集。每条记录中包含了日期 (从 1 到 91)、用户信息 (年龄，性别)，被点击的广告的信息(素材 id、广告 id、产品 id、产品类目 id、广告主 id、广告主行业 id 等)，以及该用户当天点击该广告的次数。测试数据集将会是另一组用户 的广告点击历史记录。提供给参赛者的测试数据集中不会包含这些用户的年龄和性别信息。 本赛题要求参赛者预测测试数据集中出现的用户的年龄和性别。

### 1. 环境配置

- Pytorch
- Linux Ubuntu 16.04, 256G内存，4*p100
- pip install transformers==2.8.0 pandas gensim scikit-learn filelock gdown

### 2. 模型介绍

![avatar](picture/model.png)
![avatar](picture/mlm.png)
![avatar](picture/fusion-layer.png)
![avatar](picture/output.png)

### 3. 低配置资源建议


1)内存不足或者只是想简单跑下完整代码，请只使用初赛数据:

去掉src/prepocess.py的8, 15, 22行

2)如果显存不足，请下载10中的bert-small模型，并调整batch size

### 4. 运行完整过程

可运行以下脚本，运行整个过程并生成结果。或按照3-7节的说明依次运行。

```shell
bash run.sh
```

### 5. 数据下载

通过该[网站](https://drive.google.com/file/d/15onAobxlim_uRUNWSMQuK6VxDsmGTtp4/view?usp=sharing)下载数据集到data目录，或运行下面的命令进行下载

```shell
gdown https://drive.google.com/uc?id=15onAobxlim_uRUNWSMQuK6VxDsmGTtp4
unzip data.zip 
rm data.zip
```

### 6. 数据预处理

合并所有文件，并分为点击记录文件(click.pkl)，用户文件(train_user.pkl/test_user.pkl)

```
python src/preprocess.py
```

### 7. 特征提取

```shell
python src/extract_features.py
```

### 8. 预训练 Word2Vector 与 BERT

这里提供两种方式获得预训练权重: 重新预训练或下载预训练好的权重 

注: Word2Vector和BERT权重必须一致，即要么全部重新预训练，要么全部下载

#### 1) 预训练Word2Vector

预训练word2vector

```shell
python src/w2v.py
```

或下载预训练好的[W2V](https://drive.google.com/file/d/1SUpukAeXR5Ymyf3wH3SRNdQ3Hl2HazQa/view?usp=sharing)

```shell
gdown https://drive.google.com/uc?id=1SUpukAeXR5Ymyf3wH3SRNdQ3Hl2HazQa
unzip w2v.zip 
cp w2v/* data/
rm -r w2v*
```

#### 2) 预训练BERT

预训练BERT (如果GPU是v100，可以安装apex并在参数上加--fp16进行加速)

```shell
cd BERT
mkdir saved_models
python run.py \
    --output_dir saved_models \
    --model_type roberta \
    --config_name roberta-base \
    --mlm \
    --block_size 128 \
    --per_gpu_train_batch_size 64 \
    --per_gpu_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --learning_rate 5e-5 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 1.0 \
    --max_steps 100000 \
    --mlm_probability 0.2 \
    --warmup_steps 10000 \
    --logging_steps 50 \
    --save_steps 10000 \
    --evaluate_during_training \
    --save_total_limit 500 \
    --seed 123456 \
    --tensorboard_dir saved_models/tensorboard_logs    
rm -r saved_models/bert-base    
cp -r saved_models/checkpoint-last saved_models/bert-base
rm saved_models/bert-base/optimizer.pt
cp saved_models/vocab.pkl saved_models/bert-base/vocab.pkl
cd ..
```

或下载预训练好的[BERT-base](https://drive.google.com/file/d/1ToAJwl_oRAeRNyYF_FK0B2APVXlPFTlq/view?usp=sharing)

```shell
gdown https://drive.google.com/uc?id=1ToAJwl_oRAeRNyYF_FK0B2APVXlPFTlq
unzip bert-base.zip
mv bert-base BERT/
rm bert-base.zip
```

### 9. 训练模型

```shell
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
```

合并结果，结果为submission.csv

```shell
python src/merge_submission.py
```

### 10. 不同规模的预训练模型

由于此次比赛融合了不同规模大小的预训练模型，在此也提供不同规模的预训练模型: 

[BERT-small](https://drive.google.com/file/d/1bDneO-YhBs5dx-9qC-WrBf3jUc_QCIYn/view?usp=sharing), [BERT-base](https://drive.google.com/file/d/1ToAJwl_oRAeRNyYF_FK0B2APVXlPFTlq/view?usp=sharing), [BERT-large](https://drive.google.com/file/d/1yQeh3O6E_98srPqTVwAnVbr1v-X0A7R-/view?usp=sharing), [BERT-xl](https://drive.google.com/file/d/1jViHtyljOJxxeOBmxn9tOZg_hmWOj0L2/view?usp=sharing)

其中bert-base效果最好

```shell
#bert-small
gdown https://drive.google.com/uc?id=1bDneO-YhBs5dx-9qC-WrBf3jUc_QCIYn
#bert-base
gdown https://drive.google.com/uc?id=1ToAJwl_oRAeRNyYF_FK0B2APVXlPFTlq
#bert-large
gdown https://drive.google.com/uc?id=1yQeh3O6E_98srPqTVwAnVbr1v-X0A7R-
#bert-xl
gdown https://drive.google.com/uc?id=1jViHtyljOJxxeOBmxn9tOZg_hmWOj0L2
```
