# 人工データを用いたS4BMの動作実験
import numpy as np
from my_models import S4BM
from my_models import PrMethod

# CSVファイルのパスを指定
file_path = 'data/input/artificial_data_1.csv'
# CSVファイルを読み込んでNumPy配列に変換
data = np.loadtxt(file_path, delimiter=',')
categories=([64, 65, 66, 67], [0, 1, 2, 3], [32, 33, 34, 35], [96, 97, 98, 99])

model = PrMethod(max_iter=10, n_blocks=4, random_state=99)
cluster = model.fit_transform(data, categories)
print("------------------------Result------------------------")
print("model=" + str(model))
print("categories=" + str(categories))
print("file_path=" + str(file_path))
print(cluster)