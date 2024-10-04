import numpy as np
import os
from my_models import PrMethod
from my_module import tools


#入力パラメータ
# データサイズの基準
BASE_ROW_SIZE = 100
BASE_COL_SIZE = 150

# 最大倍率 N を指定 (例: 5倍まで)
N = 1

# データ生成に使用するパラメータ
NUM_CLUSTERS_ROW = 2
NUM_CLUSTERS_COL = 3
THETA = np.array([
    [[0.4, 0.6],[0.8, 0.2],[0.05, 0.95]],
    [[0.4, 0.6],[0.05, 0.95],[0.8, 0.2]]
])



# 出力ディレクトリ
output_dir = 'Experiment/Exp02/output'
os.makedirs(output_dir, exist_ok=True)  # 出力ディレクトリがなければ作成

# タプルのリストを内包表記で生成
data_sizes = [(BASE_ROW_SIZE * i, BASE_COL_SIZE * i) for i in range(1, N + 1)]
# data_sizes = [(BASE_ROW_SIZE * N, BASE_COL_SIZE * N)]

# 各データサイズに対して処理を実行
for (n_row, n_col) in data_sizes:
    # ファイルパスを動的に生成
    file_path = f'Experiment/Exp02/input/{n_row}x{n_col}'
    
    # 必要なファイルのパス
    artificial_data_path = file_path + f'/adj_matrix.csv'
    row_clusters_path = file_path + '/domain1_clusters.csv'
    col_clusters_path = file_path + '/domain2_clusters.csv'
    
    # データが存在しなければデータ生成関数を呼び出してデータを作成
    if not os.path.exists(artificial_data_path) or not os.path.exists(row_clusters_path) or not os.path.exists(col_clusters_path):
        print(f"Files for {n_row}x{n_col} not found, generating new data...")
        os.makedirs(file_path, exist_ok=True)  # ディレクトリがなければ作成
        
        # データを生成して保存
        tools.generate_bipartite_graph(n_objects_row=n_row, n_objects_col=n_col,
                                 n_clusters_row=NUM_CLUSTERS_ROW, n_clusters_col=NUM_CLUSTERS_COL,
                                 theta=THETA, output_dir=file_path, seed=42)
    
    # データの読み込み
    try:
        data = np.loadtxt(artificial_data_path, delimiter=',')
        row_true_clusters = np.loadtxt(row_clusters_path, delimiter=',')
        col_true_clusters = np.loadtxt(col_clusters_path, delimiter=',')
    except FileNotFoundError:
        print(f"Error reading files for {n_row}x{n_col}, skipping...")
        continue  # ファイルが読み込めない場合はスキップ
    
    # カテゴリのグループを定義
    categories_row = ([], [])
    categories_col = ([1,2,7],[0,3,4],[6,9,11])

    for rs in range(1):
        # 各データサイズでモデルを実行
        model = PrMethod(max_iter=10, num_cluster_k=2, num_cluster_l=3, random_state=7)
        cluster_k, cluster_l = model.fit_transform(data, categories_row, categories_col)

        # NMIを計算
        row_NMI = tools.cal_NMI(row_true_clusters, cluster_k)
        col_NMI = tools.cal_NMI(col_true_clusters, cluster_l)

        # 結果を表示
        print(f'Data size: {n_row}x{n_col}')
        print(f'row_NMI: {row_NMI}')
        print(f'col_NMI: {col_NMI}')
        print(cluster_k)
        
        # 行と列のNMIをそれぞれのCSVファイルに蓄積
        tools.append_nmi_to_csv(os.path.join(output_dir, 'row_NMI.csv'), (n_row, n_col), row_NMI)
        tools.append_nmi_to_csv(os.path.join(output_dir, 'col_NMI.csv'), (n_row, n_col), col_NMI)

        print(f"row_NMI and col_NMI for {n_row}x{n_col} saved to {output_dir}")
