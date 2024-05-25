# 人工データを用いたS4BMの動作実験
import numpy as np
from my_models import S4BM

# CSVファイルのパスを指定
file_path = 'data/input/artificial_data_for_debug.csv'
# CSVファイルを読み込んでNumPy配列に変換
data = np.loadtxt(file_path, delimiter=',')
categories = [[],[]]

model = S4BM(max_iter=10, n_blocks=2, random_state=102)
cluster = model.fit_transform(data, categories)
print(file_path)
print(cluster)