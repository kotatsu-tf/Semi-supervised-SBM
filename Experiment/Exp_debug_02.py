# 人工データを用いたS4BMの動作実験
import numpy as np
from my_models import S4BM

# CSVファイルのパスを指定
file_path = 'data/input/artificial_data_debug3c.csv'
# CSVファイルを読み込んでNumPy配列に変換
data = np.loadtxt(file_path, delimiter=',')
categories = [[],[]]
for seed in range(20):
    model = S4BM(max_iter=5, n_blocks=2, random_state=seed)
    cluster = model.fit_transform(data, categories)
    print("------------------------Result------------------------")
    print("seed="+str(seed))
    print("categories=" + str(categories))
    print("file_path=" + str(file_path))
    print(cluster)