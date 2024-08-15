import numpy as np
import itertools
from my_models import S4BM
from my_models import PrMethod

# CSVファイルのパスを指定
file_path = 'data/input/artificial_data_1.csv'

# CSVファイルを読み込んでNumPy配列に変換
data = np.loadtxt(file_path, delimiter=',')

# カテゴリのグループを定義
categories = ([64, 65, 66, 67], [0, 1, 2, 3], [32, 33, 34, 35], [96, 97, 98, 99])

# カテゴリの順列を生成
permutations = list(itertools.permutations(categories))

# 各順列に対してモデルを実行
for rs in range(10):
    for perm in permutations:
        model = S4BM(max_iter=10, n_blocks=4, random_state=rs)
        cluster = model.fit_transform(data, perm)
        print("------------------------Result------------------------")
        print("model=" + str(model))
        print("seed=" + str(rs))
        print("categories=" + str(perm))
        print("file_path=" + str(file_path))
        print(cluster)
