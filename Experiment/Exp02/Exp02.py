import numpy as np
import os
import time
from my_models import PrMethod
from my_module import tools

# 入力パラメータ
BASE_ROW_SIZE = 100
BASE_COL_SIZE = 150
N = 5  # データサイズの最大倍率
NUM_CLUSTERS_ROW = 2  # クラスタ数 (行)
NUM_CLUSTERS_COL = 3  # クラスタ数 (列)
THETA = np.array([
    [[0.4, 0.6], [0.8, 0.2], [0.05, 0.95]],
    [[0.4, 0.6], [0.05, 0.95], [0.8, 0.2]]
])
range_model_seed = 10


output_dir = 'Experiment/Exp02/output'
os.makedirs(output_dir, exist_ok=True)

data_sizes = [(BASE_ROW_SIZE * i, BASE_COL_SIZE * i) for i in range(1, N + 1)]

# 関数：指定されたクラスタ数に対してインデックスを取得
def get_indices_for_clusters(cluster_array, num_clusters, num_teachers):
    indices_per_cluster = []
    for value in range(num_clusters):
        indices_per_cluster.append([i for i, x in enumerate(cluster_array) if x == value][:num_teachers])
    return indices_per_cluster

# 各データサイズに対して処理を実行
for (n_row, n_col) in data_sizes:
    file_path = f'Experiment/Exp02/input/{n_row}x{n_col}'
    artificial_data_path = file_path + f'/adj_matrix.csv'
    row_clusters_path = file_path + '/domain1_clusters.csv'
    col_clusters_path = file_path + '/domain2_clusters.csv'

    if not os.path.exists(artificial_data_path) or not os.path.exists(row_clusters_path) or not os.path.exists(col_clusters_path):
        print(f"Files for {n_row}x{n_col} not found, generating new data...")
        os.makedirs(file_path, exist_ok=True)
        tools.generate_bipartite_graph(n_objects_row=n_row, n_objects_col=n_col,
                                       n_clusters_row=NUM_CLUSTERS_ROW, n_clusters_col=NUM_CLUSTERS_COL,
                                       theta=THETA, output_dir=file_path, seed=42)

    try:
        data = np.loadtxt(artificial_data_path, delimiter=',')
        row_true_clusters = np.loadtxt(row_clusters_path, delimiter=',')
        col_true_clusters = np.loadtxt(col_clusters_path, delimiter=',')
    except FileNotFoundError:
        print(f"Error reading files for {n_row}x{n_col}, skipping...")
        continue

    teach_row_num = 0
    # 行と列の教師情報の件数を異なるパターンで実験
    for num_teachers_row, num_teachers_col in [(teach_row_num, 3), (teach_row_num, 5), (teach_row_num, 10), (teach_row_num, 15)]:
        if num_teachers_row == 0:
            # 教師情報なしの場合は空リストを渡す
            categories_row = ([], [])
        else:
            # 行のクラスタ数に基づいて教師情報を設定
            row_indices = get_indices_for_clusters(row_true_clusters, NUM_CLUSTERS_ROW, num_teachers_row)
            categories_row = tuple(row_indices)
        if num_teachers_col == 0:
            categories_col = ([], [], [])
        else:
            # 列のクラスタ数に基づいて教師情報を設定
            col_indices = get_indices_for_clusters(col_true_clusters, NUM_CLUSTERS_COL, num_teachers_col)
            categories_col = tuple(col_indices)

        for rs in range(range_model_seed):
            # モデルの実行時間計測開始
            start_time = time.time()

            # モデルを実行
            model = PrMethod(max_iter=10, num_cluster_k=NUM_CLUSTERS_ROW, num_cluster_l=NUM_CLUSTERS_COL, random_state=rs)
            cluster_k, cluster_l = model.fit_transform(data, categories_row, categories_col)

            # モデルの実行時間計測終了
            end_time = time.time()
            execution_time = round(end_time - start_time, 2)

            # NMIを計算
            row_NMI = tools.cal_NMI(row_true_clusters, cluster_k)
            col_NMI = tools.cal_NMI(col_true_clusters, cluster_l)

            # 結果を表示
            print(f'Data size: {n_row}x{n_col}, Row Teachers: {num_teachers_row}, Col Teachers: {num_teachers_col}')
            print(f'row_NMI: {row_NMI}, col_NMI: {col_NMI}')
            print(f'Execution Time: {execution_time} seconds')

            # NMI結果をCSVに保存
            tools.append_nmi_to_csv(
                os.path.join(output_dir, 'NMI_value.csv'),
                (n_row, n_col),
                row_NMI,
                col_NMI,
                rs,
                num_teachers_row,
                num_teachers_col,
                categories_row,
                categories_col,
                execution_time
            )
            print(f"NMI values for {n_row}x{n_col}, Row Teachers: {num_teachers_row}, Col Teachers: {num_teachers_col} saved.")
