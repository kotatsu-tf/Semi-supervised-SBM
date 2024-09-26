import numpy as np
import itertools
from my_models import S4BM
from my_models import PrMethod

# CSVファイルのパスを指定
file_path = 'data/input/artificial_data_prmethod_debug.csv'

# CSVファイルを読み込んでNumPy配列に変換
data = np.loadtxt(file_path, delimiter=',')

# カテゴリのグループを定義
categories = (
    [64, 65, 66, 67, 68, 69, 70, 71],
    [0, 1, 2, 3, 4, 5, 6, 7],
    [32, 33, 34, 35, 36, 37, 38 ,39],
    [96, 97, 98, 99, 100, 101, 102, 103]
    )
categories=([])

# カテゴリの順列を生成
permutations = list(itertools.permutations(categories))

# 各順列に対してモデルを実行
for rs in range(1):
    for perm in permutations:
        # model = S4BM(max_iter=10, n_blocks=4, random_state=rs)
        model = PrMethod(max_iter=10, num_cluster_k=4, num_cluster_l=4, random_state=rs)
        cluster_k, clusetr_l = model.fit_transform(data, perm, perm)
        print("------------------------Result------------------------")
        print("model=" + str(model))
        print("seed=" + str(rs))
        print("categories=" + str(perm))
        print("file_path=" + str(file_path))
        print(cluster_k)
        print(clusetr_l)
