# -*- coding: utf-8 -*-
"""
@Time ： 2020/8/20 下午2:31
@Auth ： shenhao
@Email： shenhao@xiaomi.com
"""
import pandas as pd
import numpy as np

df_test = pd.DataFrame([['a1', 1], ['a2', 4]], columns=['uid', 'score'])
df_tmp = df_test['uid' + ['score']]
print()