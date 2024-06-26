# 人工データを用いたS4BMの動作実験
import numpy as np
from my_models import S4BM

# CSVファイルのパスを指定
file_path = 'data/input/artificial_data_1.csv'
# CSVファイルを読み込んでNumPy配列に変換
data = np.loadtxt(file_path, delimiter=',')
categories = [[0, 1, 2, 3, 4],[],[],[]]

model = S4BM(max_iter=10, n_blocks=4, random_state=104)
cluster = model.fit_transform(data, categories)
print("------------------------Result------------------------")
print("model=" + str(model))
print("categories=" + str(categories))
print("file_path=" + str(file_path))
print(cluster)