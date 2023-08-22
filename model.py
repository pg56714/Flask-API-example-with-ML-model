# -*- coding: UTF-8 -*-
# 使用 pickle 儲存模型並利用 gzip 壓縮
import pickle
import gzip

# 載入Model
with gzip.open('./model/xgboost-iris.pgz', 'r') as f:
    xgboostModel = pickle.load(f)


def predict(input):
    pred = xgboostModel.predict(input)[0]  # 只抓第一筆
    print(pred)
    return pred
