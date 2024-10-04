import numpy as np
import itertools
from my_models import S4BM
from my_models import PrMethod
from my_module import tools
import os

# CSVファイルのパスを指定
input_path = 'Experiment/Exp01/input/100x150'
# artificial_bipartite_data 1と同条件で行列5倍の大きさのデータ

# CSVファイルを読み込んでNumPy配列に変換
data = np.loadtxt(input_path + str('/adj_matrix.csv'), delimiter=',')
row_true_clusters = np.loadtxt(input_path + str('/domain1_clusters.csv'), delimiter=',')
col_true_clusters = np.loadtxt(input_path + str('/domain2_clusters.csv'), delimiter=',')

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
    print("input_path=" + str(input_path))
    print(cluster_k)
    print(clusetr_l)

row_NMI = tools.cal_NMI(row_true_clusters, cluster_k)
col_NMI = tools.cal_NMI(col_true_clusters, clusetr_l)
print('row_NMI: ' + str(row_NMI))
print('col_NMI: ' + str(col_NMI))

# 出力ディレクトリを定義
output_dir = 'Experiment/Exp01/output'
os.makedirs(output_dir, exist_ok=True)  # ディレクトリがなければ作成

# NMIをCSVに保存
np.savetxt(os.path.join(output_dir, 'row_NMI.csv'), [row_NMI], delimiter=',', fmt='%f', header='row_NMI')
np.savetxt(os.path.join(output_dir, 'col_NMI.csv'), [col_NMI], delimiter=',', fmt='%f', header='col_NMI')

print(f"row_NMI and col_NMI saved to {output_dir}")