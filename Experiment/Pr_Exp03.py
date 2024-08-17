import numpy as np
import itertools
from my_models import S4BM
from my_models import PrMethod

# CSVファイルのパスを指定
file_path = 'data/input/artificial_bipartite_data3.csv'
# artificial_bipartite_data 1と同条件で行列5倍の大きさのデータ

# CSVファイルを読み込んでNumPy配列に変換
data = np.loadtxt(file_path, delimiter=',')

# カテゴリのグループを定義
# categories = (
#     [64, 65, 66, 67, 68, 69, 70, 71],
#     [0, 1, 2, 3, 4, 5, 6, 7],
#     [32, 33, 34, 35, 36, 37, 38 ,39],
#     [96, 97, 98, 99, 100, 101, 102, 103]
#     )
categories_row=([],[])
categories_col=([1,2],[],[6,9])

# # カテゴリの順列を生成
# permutations = list(itertools.permutations(categories_col))

# 各順列に対してモデルを実行
for rs in range(1):
    model = PrMethod(max_iter=10, num_cluster_k=2, num_cluster_l=3, random_state=rs)
    cluster_k, clusetr_l = model.fit_transform(data, categories_row, categories_col)
    print("------------------------Result------------------------")
    print("model=" + str(model))
    print("seed=" + str(rs))
    print("categories_row=" + str(categories_row))
    print("categories_col=" + str(categories_col))
    print("file_path=" + str(file_path))
    print(cluster_k)
    print(clusetr_l)
