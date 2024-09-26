# 人工データを用いたS4BMの動作実験
import numpy as np
from my_models import S4BM

# CSVファイルのパスを指定
file_path = 'data/input/artificial_data_debug2a.csv'
# CSVファイルを読み込んでNumPy配列に変換
data = np.loadtxt(file_path, delimiter=',')
categories = [[],[]]

SEED =100
model = S4BM(max_iter=30, n_blocks=2, random_state=SEED, is_show_params_history=True)
cluster = model.fit_transform(data, categories)
print("------------------------Result------------------------")
print("seed=" + str(SEED))
print("categories=" + str(categories))
print("file_path=" + str(file_path))
print(cluster)