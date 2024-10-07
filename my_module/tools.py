from sklearn.metrics import normalized_mutual_info_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import numpy as np
import pandas as pd

def generate_bipartite_graph(n_objects_row, n_objects_col, n_clusters_row, n_clusters_col, theta, output_dir,
                             row_cluster_ratios=None, col_cluster_ratios=None, seed=None, probabilistic=False):
    """
    購買履歴を模倣した2部グラフを生成する関数。thetaの比率に従うか確率的に生成するか選択可能。

    Parameters:
    - n_objects_row (int): 行側オブジェクトの数（顧客など）
    - n_objects_col (int): 列側オブジェクトの数（商品など）
    - n_clusters_row (int): 行側クラスタの数
    - n_clusters_col (int): 列側クラスタの数
    - theta (numpy.ndarray): クラスタ間の関係強さのパラメータ, 形状は (n_clusters_row, n_clusters_col, 2) で、
                            それぞれのクラスタペアでの購入する確率、購入しない確率を示す。
    - row_cluster_ratios (list or numpy.ndarray, optional): 行側クラスタごとの比率。指定がない場合は均等に割り当て。
    - col_cluster_ratios (list or numpy.ndarray, optional): 列側クラスタごとの比率。指定がない場合は均等に割り当て。
    - seed (int, optional): 乱数シード
    - probabilistic (bool, optional): Trueの場合は確率的に生成、Falseの場合は必ずthetaの比率に従って生成。

    Returns:
    - adj_matrix (numpy.ndarray): 生成された隣接行列（購買履歴データ）
    - row_clusters (numpy.ndarray): 行側オブジェクトのクラスタ割り当て
    - col_clusters (numpy.ndarray): 列側オブジェクトのクラスタ割り当て
    """
    rng = np.random.default_rng(seed)

    # クラスタ比率の設定（指定がない場合は均等に割り当て）
    if row_cluster_ratios is None:
        row_cluster_ratios = np.ones(n_clusters_row) / n_clusters_row
    if col_cluster_ratios is None:
        col_cluster_ratios = np.ones(n_clusters_col) / n_clusters_col

    # 行側・列側オブジェクトをクラスタに割り当てる
    row_clusters = np.concatenate([
        np.full(int(ratio * n_objects_row), cluster_id)
        for cluster_id, ratio in enumerate(row_cluster_ratios)
    ])
    col_clusters = np.concatenate([
        np.full(int(ratio * n_objects_col), cluster_id)
        for cluster_id, ratio in enumerate(col_cluster_ratios)
    ])

    # 割り当てたクラスタの順序をシャッフル
    rng.shuffle(row_clusters)
    rng.shuffle(col_clusters)

    # 隣接行列を初期化
    adj_matrix = np.zeros((n_objects_row, n_objects_col))

    if probabilistic:
        # 確率的にリンクを生成
        for i in range(n_objects_row):
            for j in range(n_objects_col):
                cluster_row = row_clusters[i]
                cluster_col = col_clusters[j]
                rnd = rng.random()
                if rnd < theta[cluster_row, cluster_col, 0]:  # 購入
                    adj_matrix[i, j] = 1
                else:  # 購入しない
                    adj_matrix[i, j] = 0
    else:
        # thetaの比率に必ず従ってリンクを生成
        for k in range(n_clusters_row):
            for l in range(n_clusters_col):
                row_indices = np.where(row_clusters == k)[0]
                col_indices = np.where(col_clusters == l)[0]

                # リンク数の計算
                n_links = int(np.round(theta[k, l, 0] * len(row_indices) * len(col_indices)))

                # リンクを配置するインデックスをランダムに選択
                link_indices = rng.choice(len(row_indices) * len(col_indices), n_links, replace=False)
                link_positions = np.unravel_index(link_indices, (len(row_indices), len(col_indices)))

                # 隣接行列にリンクを配置
                for i, j in zip(row_indices[link_positions[0]], col_indices[link_positions[1]]):
                    adj_matrix[i, j] = 1

        # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # 隣接行列をCSVとして保存
    adj_matrix_file = os.path.join(output_dir, 'adj_matrix.csv')
    pd.DataFrame(adj_matrix).to_csv(adj_matrix_file, header=False, index=False)
    
    # 行側クラスタ割り当てをCSVとして保存
    row_clusters_file = os.path.join(output_dir, 'domain1_clusters.csv')
    np.savetxt(row_clusters_file, [row_clusters], delimiter=',', fmt='%d', newline='')
    
    # 列側クラスタ割り当てをCSVとして保存
    col_clusters_file = os.path.join(output_dir, 'domain2_clusters.csv')
    np.savetxt(col_clusters_file, [col_clusters], delimiter=',', fmt='%d', newline='')

    print(f"Adjacency matrix saved to {adj_matrix_file}")
    print(f"Row clusters saved to {row_clusters_file}")
    print(f"Col clusters saved to {col_clusters_file}")

def visualize_matrix(matrix, output_dir, filename):
    """
    行列を指定の色で可視化し、指定ディレクトリに保存する関数。
    1: 黒, 0: 白, -1: 赤
    output_dir: 出力ディレクトリ
    filename: 保存するファイル名（例: 'matrix_visualization.png'）
    """
    # カスタムカラーマップの作成
    cmap = mcolors.ListedColormap(['red', 'white', 'black'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(18, 10))  # 図のサイズを指定
    plt.imshow(matrix, cmap=cmap, norm=norm, interpolation='none')

    # グリッド線を引く（薄いグレーで少し細く）
    plt.grid(which='major', color='gray', linestyle='-', linewidth=1)

    # ラベルと目盛りを設定（フォントサイズを大きく）
    plt.xticks(np.arange(matrix.shape[1]), np.arange(1, matrix.shape[1] + 1), rotation=90, fontsize=12)
    plt.yticks(np.arange(matrix.shape[0]), np.arange(1, matrix.shape[0] + 1), fontsize=12)

    # 番号が線の間に来るように調整
    plt.gca().xaxis.set_major_locator(plt.FixedLocator(np.arange(-0.5, matrix.shape[1], 1)))
    plt.gca().yaxis.set_major_locator(plt.FixedLocator(np.arange(-0.5, matrix.shape[0], 1)))

    # 軸の表示を反転（上から下に並べるため）
    plt.gca().invert_yaxis()

    # 枠線を削除
    plt.gca().spines[:].set_visible(False)

    # 余白の調整
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)

    # 図を指定のディレクトリに保存
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, bbox_inches='tight', dpi=300)  # 解像度300dpiで保存

    # 表示を閉じる（メモリ節約のため）
    plt.close()

    print(f"Visualization saved to {output_path}")

def cal_NMI(true_label, pred_label):
    return normalized_mutual_info_score(true_label, pred_label)

import os
import time

def append_nmi_to_csv(file_path, data_size, row_nmi_value, col_nmi_value, seed, categories_row_num, categories_col_num, categories_row, categories_col, execution_time):
    """
    NMIの結果を指定のファイルに追記する関数。
    
    Parameters:
    - file_path (str): 保存するファイルのパス
    - data_size (tuple): データサイズ (n_row, n_col) のタプル
    - row_nmi_value (float): 計算した行方向NMI値
    - col_nmi_value (float): 計算した列方向NMI値
    - seed (int): モデルのseed値
    - categories_row (tuple): 第1ドメインの教師情報
    - categories_col (list): 第2ドメインの教師情報
    - execution_time (float): 実行時間 (秒単位)
    """
    # ファイルが存在しない場合はヘッダーを付けて新規作成
    if not os.path.exists(file_path):
        with open(file_path, 'w') as f:
            f.write('data_size,row_nmi_value,col_nmi_value,seed,categories_row_num,categories_col_num,categories_row,categories_col,execution_time\n')
    
    # categories_rowとcategories_colを文字列に変換
    categories_row_str = str(categories_row)
    categories_col_str = str(categories_col)
    
    # データを追記
    with open(file_path, 'a') as f:
        f.write(f'{data_size[0]}x{data_size[1]},{row_nmi_value},{col_nmi_value},{seed},{categories_row_num},{categories_col_num},"{categories_row_str}","{categories_col_str}",{execution_time:.2f}\n')
