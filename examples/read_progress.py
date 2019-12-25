#!/usr/bin/env python
# -*- coding:utf8 -*-
"""
 Author: zhaopenghao
 Create Time: 2019/12/25 下午1:17
"""
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/local/20191225/example_1/run_0/progress.csv")
print(df.columns)
# plt.figure()
df.plot(x='Iteration', y=['ReturnAverage', 'ReturnStd'])
plt.show()