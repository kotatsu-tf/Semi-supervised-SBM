import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma
import sys
import time
import os
from datetime import datetime
import pytz
import csv
import copy


class PrMethod:
    def __init__(self, num_cluster_k=2, num_cluster_l=3, max_iter=2000, random_state=None, is_show_params_history=False):
        self.num_cluster_k = num_cluster_k
        self.num_cluster_l = num_cluster_l
        self.max_iter = max_iter
        self.random_state = random_state
        self.is_show_params_history = is_show_params_history
        self.rng = np.random.RandomState(random_state)


    def fit_transform(self, data, row_categories=None, col_categories=None):
        start_time = time.time()
        # オブジェクト数を行数と列数として取得
        n_objects = (data.shape[0], data.shape[1])
        num_cluster_k = self.num_cluster_k
        num_cluster_l = self.num_cluster_l

        # カテゴリ行列を作成（行と列それぞれ）
        row_category_matrix = self._create_category_matrix(row_categories, n_objects[0])
        col_category_matrix = self._create_category_matrix(col_categories, n_objects[1])

        # 提案手法のアルゴリズムを実行し、パラメータを更新
        assigned_clusters_k, assigned_clusters_l = self._proposed_method(data, num_cluster_k, num_cluster_l, row_category_matrix, col_category_matrix)

        end_time = time.time()
        elapased_time = end_time - start_time
        print(f"実行時間： {elapased_time:.2f} sec")
        return assigned_clusters_k, assigned_clusters_l
    
    def _create_category_matrix(self, category_list, n_nodes):
        """
        カテゴリリストとノードの総数からOne-hotエンコーディングされたカテゴリ行列を作成する。
        categories: 2次元配列（[[13, 15, 16], [34, 78], ...]）
        n_nodes: ノードの総数
        """
        # カテゴリの数を計算
        n_categories = len(category_list)
        
        # カテゴリ行列を初期化
        category_matrix = np.zeros((n_nodes, n_categories))
        
        # カテゴリリストからカテゴリ行列を作成
        for category_id, node_ids in enumerate(category_list):
            for node_id in node_ids:
                if node_id < n_nodes:  # ノードIDがn_nodes未満の場合のみ処理
                    category_matrix[node_id, category_id] = 1
                else:
                    print(f"Error: Node ID {node_id} exceeds the number of nodes {n_nodes}.")
                    sys.exit(1)  # プログラムを終了    
                    
        return category_matrix
    

    def _calculate_link_ratios(self, data):
        total_links = data.size  # 隣接行列内の総要素数
        rp = np.sum(data == 1) / total_links  # 正のリンクの割合
        rn = np.sum(data == -1) / total_links  # 負のリンクの割合
        ro = np.sum(data == 0) / total_links  # リンクなしの割合
        return rp, rn, ro
    
    def _initialize_parameters(self, data, n_objects, num_cluster_1, num_cluster_2):
        """
        パラメータtau, gamma, eta, rho, alphaの初期化
        data: 隣接行列
        n_objects: オブジェクト数 (n_rows, n_cols)
        num_cluster_1: ブロック数（行側）
        num_cluster_2: ブロック数（列側）
        """
        rp, rn, ro = self._calculate_link_ratios(data)
        
        # τ, γ, αの初期化
        tau_1 = self.rng.rand(n_objects[0], num_cluster_1)  # 行側クラスタのτ（タウ）
        tau_2 = self.rng.rand(n_objects[1], num_cluster_2)  # 列側クラスタのτ（タウ）
        
        gamma_1 = self.rng.rand(n_objects[0], num_cluster_1)  # 行側のγ（ガンマ）
        gamma_2 = self.rng.rand(n_objects[1], num_cluster_2)  # 列側のγ（ガンマ）
        
        alpha_1 = self.rng.rand(num_cluster_1, num_cluster_1)  # 行側のα（アルファ）
        alpha_2 = self.rng.rand(num_cluster_2, num_cluster_2)  # 列側のα（アルファ）
        
        # rhoの初期化
        rho_1 = np.ones(num_cluster_1)  # 行側のブロック数に対するρ（ロー）
        rho_2 = np.ones(num_cluster_2)  # 列側のブロック数に対するρ（ロー）

        # etaの初期化
        eta = np.zeros((num_cluster_1, num_cluster_2, 3))  # 3次元配列で初期化 (num_cluster_1, num_cluster_2, 3)
        for k in range(num_cluster_1):
            for l in range(num_cluster_2):
                eta[k][l] = np.array([rp, rn, ro])  # 各eta_klをnp.array([rp, rn, ro])で初期化

        return tau_1, tau_2, gamma_1, gamma_2, alpha_1, alpha_2, rho_1, rho_2, eta


    
    def _proposed_method(self, data, num_cluster_1, num_cluster_2, row_category_matrix, col_category_matrix):
        """
        提案手法に基づくパラメータ推定アルゴリズム。
        data: 隣接行列
        num_cluster_1: 行側ブロック数
        num_cluster_2: 列側ブロック数
        row_category_matrix: 行側カテゴリ行列
        col_category_matrix: 列側カテゴリ行列
        """
        n_objects = (data.shape[0], data.shape[1])

        # パラメータの初期化
        tau_1_init, tau_2_init, gamma_1_init, gamma_2_init, alpha_1_init, alpha_2_init, rho_1_init, rho_2_init, eta_init = self._initialize_parameters(data, n_objects, num_cluster_1, num_cluster_2)
        # _initを外した変数に値を代入
        tau_1 = tau_1_init.copy()
        tau_2 = tau_2_init.copy()
        gamma_1 = gamma_1_init.copy()
        gamma_2 = gamma_2_init.copy()
        alpha_1 = alpha_1_init.copy()
        alpha_2 = alpha_2_init.copy()
        rho_1 = rho_1_init.copy()
        rho_2 = rho_2_init.copy()
        eta = eta_init.copy()


        # パラメータ履歴の保存用リスト
        tau_1_history = []
        tau_2_history = []
        gamma_1_history = []
        gamma_2_history = []
        alpha_1_history = []
        alpha_2_history = []
        rho_1_history = []
        rho_2_history = []
        eta_history = []

        # previous_elbo = self._calculate_elbo(data, tau_1, tau_2, gamma_1, gamma_2, alpha_1, alpha_2, rho_1_init, rho_2_init, eta_init)

        for iteration in range(self.max_iter):
            # 前回のパラメータを保存
            tau_1_prev = tau_1.copy()
            tau_2_prev = tau_2.copy()

            # パラメータの更新（初期化時のrho_1_init, rho_2_init, eta_initを使用）
            tau_1 = self._update_tau_row(alpha_1, gamma_1, eta, rho_1, tau_1_prev, tau_2, data, row_category_matrix)
            tau_2 = self._update_tau_col(alpha_2, gamma_2, eta, rho_2, tau_2_prev, tau_1, data, col_category_matrix)
            gamma_1 = self._update_gamma(gamma_1, tau_1, alpha_1, row_category_matrix)
            gamma_2 = self._update_gamma(gamma_2, tau_2, alpha_2, col_category_matrix)
            rho_1 = self._update_rho(rho_1_init, tau_1)
            rho_2 = self._update_rho(rho_2_init, tau_2)
            eta = self._update_eta(tau_1, tau_2, data, eta_init)
            alpha_1 = self._update_alpha(tau_1, gamma_1, alpha_1)
            alpha_2 = self._update_alpha(tau_2, gamma_2, alpha_2)

            # 各パラメータの値を履歴として保存
            tau_1_history.append(tau_1.copy())
            tau_2_history.append(tau_2.copy())
            gamma_1_history.append(gamma_1.copy())
            gamma_2_history.append(gamma_2.copy())
            alpha_1_history.append(alpha_1.copy())
            alpha_2_history.append(alpha_2.copy())
            rho_1_history.append(rho_1.copy())
            rho_2_history.append(rho_2.copy())
            eta_history.append(eta.copy())

            # 収束条件のチェックを一時的にコメントアウト
            # current_elbo = self._calculate_elbo(data, tau_1, tau_2, gamma_1, gamma_2, alpha_1, alpha_2, rho_1, rho_2, eta)
            # if np.abs(current_elbo - previous_elbo) < 1e-6:
            #     print(f'Converged after {iteration + 1} iterations.')
            #     break
            # previous_elbo = current_elbo

            print(f'Iteration {iteration + 1} completed.')

        Z_1 = np.argmax(tau_1, axis=1)
        Z_2 = np.argmax(tau_2, axis=1)

        if self.is_show_params_history:
            # パラメータ履歴をグラフ化または保存する
            self._make_graph(tau_1_history, tau_2_history, gamma_1_history, gamma_2_history, rho_1_history, rho_2_history, alpha_1_history, alpha_2_history, eta_history)

        return Z_1, Z_2

    
    def _make_graph(self, tau_his, gamma_his, rho_his, alpha_his, eta_his, mu_his):
        data_list = {'tau_his':tau_his, 'gamma_his':gamma_his, 'rho_his':rho_his, 'alpha_his':alpha_his, 'eta_his':eta_his, 'mu_his':mu_his}

        # 現在の時刻を取得してフォルダ名を作成
        jst = pytz.timezone('Asia/Tokyo')
        current_time = datetime.now(jst).strftime('%Y%m%d_%H%M%S')
        save_directory = os.path.join('/workspaces/Semi-supervised-SBM/data/output/images', current_time)
        os.makedirs(save_directory, exist_ok=True)  # ディレクトリが存在しない場合は作成

        for list_name, data in data_list.items():
            data = np.array(data)
            dim = data.ndim
            num_repeats = len(data)
            if dim == 3:
                num_rows, num_cols = data[0].shape
            elif dim == 2:
                num_rows = 1
                num_cols = len(data[0])
            else:
                print('正しく次元数が出力されていません')

            fig_width = max(num_cols * 5, num_repeats / 2)
            fig_height = num_rows * 5

            fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))

            # CSVファイルの準備
            csv_file_path = os.path.join(save_directory, f'{list_name}_values.csv')
            with open(csv_file_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                # ヘッダーを作成
                headers = ['Iteration'] + [f'{i}-{j}' for i in range(num_rows) for j in range(num_cols)]
                csv_writer.writerow(headers)

                # 各イテレーションの値を書き込み
                for t in range(num_repeats):
                    if dim == 3:
                        row = [t] + [data[t][i][j] for i in range(num_rows) for j in range(num_cols)]
                    elif dim == 2:
                        row = [t] + [data[t][j] for j in range(num_cols)]
                    csv_writer.writerow(row)

            # 各要素の値の変化をプロット
            for i in range(num_rows):
                for j in range(num_cols):
                    if dim == 3:
                        values = [data[t][i][j] for t in range(num_repeats)]
                        axes[i, j].plot(range(num_repeats), values, marker='o')
                        axes[i, j].set_title(f'{list_name}, node{i}, cluster{j}')
                        axes[i, j].set_xlabel('Iteration')
                        axes[i, j].set_ylabel('probability')
                        axes[i, j].grid(True)
                        axes[i, j].set_ylim(0, 1) # 縦軸の範囲を0から1に設定
                        axes[i, j].set_xticks(range(num_repeats)) # x軸のメモリを1ごとに設定
                    elif dim == 2:
                        values = [data[t][j] for t in range(num_repeats)]
                        axes[j].plot(range(num_repeats), values, marker='o')
                        axes[j].set_title(f'{list_name}, node{i}, cluster{j}')
                        axes[j].set_xlabel('Iteration')
                        axes[j].set_ylabel('probability')
                        axes[j].grid(True)
                        axes[j].set_ylim(0, 1) # 縦軸の範囲を0から1に設定
                        axes[j].set_xticks(range(num_repeats)) # x軸のメモリを1ごとに設定


            # サブプロット間のレイアウトを調整
            plt.tight_layout()

            # グラフを保存
            file_path = os.path.join(save_directory, f'{list_name}_3d_array_plot.png')
            plt.savefig(file_path)
            plt.close()

            print(f"グラフが {file_path} に保存されました。")

    def _calculate_elbo(self, data, tau, gamma, alpha, rho, mu1, eta1):
        # ダミー
        return np.random.rand()

    def _update_tau_row(self, alpha, gamma, eta, rho, tau_prev, tau_col, data, C):
        """
        行側の各ノードが各ブロックに属する確率τ（タウ）を更新する関数。

        Parameters:
        - alpha (numpy.ndarray): ラベルとブロックの関連性を示すパラメータ、形状は (M, K)
        - gamma (numpy.ndarray): γ確率の行列、形状は (N, M)
        - eta (numpy.ndarray): ブロック間リンクの存在確率、形状は (K, L, 3)
        - rho (numpy.ndarray): 各ブロックの比率、形状は (K)
        - tau_prev (numpy.ndarray): 前回の更新時のτ、形状は (N, K)
        - tau_col (numpy.ndarray): 列側のτ、形状は (N, L)
        - data (numpy.ndarray): ネットワークの隣接行列、形状は (n_objects_row, n_objects_col)。要素は1, -1, 0。
        - C (numpy.ndarray): カテゴリ行列、形状は (N, M)

        Returns:
        - numpy.ndarray: 更新されたτ行列、形状は (N, K)
        """
        N, K = tau_prev.shape
        L = tau_col.shape[1]
        log_tau = np.zeros((N, K))

        log_rho = digamma(rho) - digamma(np.sum(rho))
        beta = np.sum(np.exp(alpha), axis=0)

        for i in range(N):
            for k in range(K):
                log_tau_ik = log_rho[k]
                log_tau_ik += np.sum([
                    gamma[i, c] * (alpha[c, k] - (1.0 / np.dot(beta, tau_prev[i].T)) * beta[k])
                    for c in range(gamma.shape[1])
                ])

                for j in range(tau_col.shape[0]):  # 列方向の処理
                    data_ij = data[i, j]
                    for l in range(eta.shape[1]):
                        if data_ij == 1:
                            log_tau_ik += tau_col[j, l] * (digamma(eta[k, l, 0]) - digamma(np.sum(eta[k, l, :])))
                        elif data_ij == -1:
                            log_tau_ik += tau_col[j, l] * (digamma(eta[k, l, 1]) - digamma(np.sum(eta[k, l, :])))
                        elif data_ij == 0:
                            log_tau_ik += tau_col[j, l] * (digamma(eta[k, l, 2]) - digamma(np.sum(eta[k, l, :])))

                log_tau[i, k] = log_tau_ik

        # 数値安定化のため、各行の最大値を引く
        log_tau -= np.max(log_tau, axis=1, keepdims=True)

        # exp を取る前に数値安定化
        tau = np.exp(log_tau)

        epsilon = 0.01
        for i in range(N):
            if np.sum(C[i]) > 0:
                tau[i] = C[i]
        tau = np.maximum(tau, epsilon)

        # 正規化
        tau /= np.sum(tau, axis=1, keepdims=True)

        return tau

    def _update_tau_col(self, alpha, gamma, eta, rho, tau_prev, tau_row, data, C):
        """
        列側の各ノードが各ブロックに属する確率τ（タウ）を更新する関数。

        Parameters:
        - alpha (numpy.ndarray): ラベルとブロックの関連性を示すパラメータ、形状は (M, L)
        - gamma (numpy.ndarray): γ確率の行列、形状は (N, M)
        - eta (numpy.ndarray): ブロック間リンクの存在確率、形状は (K, L, 3)
        - rho (numpy.ndarray): 各ブロックの比率、形状は (L)
        - tau_prev (numpy.ndarray): 前回の更新時のτ、形状は (N, L)
        - tau_row (numpy.ndarray): 行側のτ、形状は (N, K)
        - data (numpy.ndarray): ネットワークの隣接行列、形状は (n_objects_row, n_objects_col)。要素は1, -1, 0。
        - C (numpy.ndarray): カテゴリ行列、形状は (N, M)

        Returns:
        - numpy.ndarray: 更新されたτ行列、形状は (N, L)
        """
        N, L = tau_prev.shape
        K = tau_row.shape[1]
        log_tau = np.zeros((N, L))

        log_rho = digamma(rho) - digamma(np.sum(rho))
        beta = np.sum(np.exp(alpha), axis=0)

        for j in range(N):
            for l in range(L):
                log_tau_jl = log_rho[l]
                log_tau_jl += np.sum([
                    gamma[j, c] * (alpha[c, l] - (1.0 / np.dot(beta, tau_prev[j].T)) * beta[l])
                    for c in range(gamma.shape[1])
                ])

                for i in range(tau_row.shape[0]):  # 行方向の処理
                    data_ji = data[i, j]
                    for k in range(eta.shape[0]):
                        if data_ji == 1:
                            log_tau_jl += tau_row[i, k] * (digamma(eta[k, l, 0]) - digamma(np.sum(eta[k, l, :])))
                        elif data_ji == -1:
                            log_tau_jl += tau_row[i, k] * (digamma(eta[k, l, 1]) - digamma(np.sum(eta[k, l, :])))
                        elif data_ji == 0:
                            log_tau_jl += tau_row[i, k] * (digamma(eta[k, l, 2]) - digamma(np.sum(eta[k, l, :])))

                log_tau[j, l] = log_tau_jl

        # 数値安定化のため、各行の最大値を引く
        log_tau -= np.max(log_tau, axis=1, keepdims=True)

        # exp を取る前に数値安定化
        tau = np.exp(log_tau)

        epsilon = 0.01
        for j in range(N):
            if np.sum(C[j]) > 0:
                tau[j] = C[j]
        tau = np.maximum(tau, epsilon)

        # 正規化
        tau /= np.sum(tau, axis=1, keepdims=True)

        return tau

    def _update_gamma(self, gamma, tau, alpha, C):
        """
        提案手法に基づいてγ（ガンマ）確率を更新する関数。

        Parameters:
        - gamma (numpy.ndarray): γの初期値、形状は (N, M)。
        - tau (numpy.ndarray): τ確率の行列、形状は (N, K)。ノードが各ブロックに属する確率を示す。
        - alpha (numpy.ndarray): αパラメータの行列で、形状は (M, K)。ラベルとブロックの関連性を示す。
        - C (numpy.ndarray): カテゴリ行列、形状は (N, M)。

        Returns:
        - numpy.ndarray: 更新されたγ行列で、形状は (N, M)。
        """
        N, K = tau.shape
        M = alpha.shape[0]

        for i in range(N):
            for c in range(M):
                numerator = np.exp(np.sum(alpha[c, :] * tau[i, :]) - 1)
                
                denominator = np.sum([
                    np.sum(tau[i, :] * np.exp(alpha[c_prime, :]))
                    for c_prime in range(M)
                ])
                
                gamma[i, c] = numerator / denominator

        # カテゴリ行列Cが与えられている場合、対応するgammaの行を置き換える
        for i in range(N):
            if np.sum(C[i]) > 0:
                gamma[i] = C[i]

        # 確率として正規化
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        return gamma


    def _update_rho(self, rho_init, tau):
        """
        提案手法に基づいてブロックの割合ρを更新する関数。

        Parameters:
        - tau: 各ノードが各ブロックに属する確率を表す配列、形状は(n_objects, num_clusters)
        - rho_init: ρの初期値を表す配列、形状は(num_clusters,)

        Returns:
        - rho: 更新されたブロックの割合を表す配列、形状は(num_clusters,)
        """
        # 各ブロックに対するノードの割り当て確率の合計を計算し、rho_initを加算
        rho = rho_init + np.sum(tau, axis=0)

        # 正規化
        rho /= np.sum(rho)
        
        return rho

    
    def _update_eta(self, tau_1, tau_2, data, eta_init):
        """
        提案手法に基づいてブロック内のリンク確率ηを更新する関数。

        Parameters:
        - tau_1: 行側ノードが各ブロックに属する確率を表す配列、形状は(n_objects_1, num_cluster_1)
        - tau_2: 列側ノードが各ブロックに属する確率を表す配列、形状は(n_objects_2, num_cluster_2)
        - data: ネットワークの隣接行列、形状は(n_objects_1, n_objects_2)。要素は1, -1, 0。
        - eta_init: ηの初期値、形状は(num_cluster_1, num_cluster_2, 3)。

        Returns:
        - eta: 更新されたブロック内のリンク確率、形状は(num_cluster_1, num_cluster_2, 3)。
        """
        num_cluster_1, num_cluster_2, _ = eta_init.shape
        eta = np.zeros_like(eta_init)

        for k in range(num_cluster_1):
            for l in range(num_cluster_2):
                for i in range(tau_1.shape[0]):  # n_objects_1
                    for j in range(tau_2.shape[0]):  # n_objects_2
                        if data[i, j] == 1:
                            eta[k, l, 0] += tau_1[i, k] * tau_2[j, l]  # 正のリンク
                        elif data[i, j] == -1:
                            eta[k, l, 1] += tau_1[i, k] * tau_2[j, l]  # 負のリンク
                        elif data[i, j] == 0:
                            eta[k, l, 2] += tau_1[i, k] * tau_2[j, l]  # リンクなし

        # 初期値 eta_init を加算
        eta += eta_init

        # 正規化
        eta /= np.sum(eta, axis=2, keepdims=True)

        return eta

                            
    def _update_alpha(self, tau, gamma, alpha, learning_rate=0.01, iterations=100, tol=1e-6):
        """
        提案手法に基づいてカテゴリとブロック間の関係パラメータαを更新する関数。

        Parameters:
        - tau: 各ノードが各ブロックに属する確率、形状は(n_objects, num_clusters)。
        - gamma: 各ノードが各カテゴリに属する確率、形状は(n_objects, num_categories)。
        - alpha: カテゴリとブロック間の初期関係パラメータ、形状は(num_categories, num_clusters)。
        - learning_rate: 勾配降下法の学習率。
        - iterations: 勾配降下法の最大反復回数。
        - tol: 収束判定のための許容誤差。デフォルトは1e-6。

        Returns:
        - alpha: 更新されたカテゴリとブロック間の関係パラメータ。
        """
        M, K = alpha.shape
        N = tau.shape[0]
        
        for iteration in range(iterations):
            gradient = np.zeros_like(alpha)  # 勾配を初期化
            alpha_old = alpha.copy()  # 以前のalphaを保存
            
            for c in range(M):
                for k in range(K):
                    numerator = 0
                    denominator = 0
                    
                    for i in range(N):
                        numerator += gamma[i, c] * (tau[i, k] - tau[i, k] * np.exp(alpha[c, k]))
                        
                        # denominator を計算
                        denominator += np.sum([tau[i, k_] * np.exp(alpha[c_prime, k_]) for c_prime in range(M) for k_ in range(K)])
                    
                    gradient[c, k] = numerator / denominator
            
            # 勾配降下法を使用してalphaを更新
            alpha += learning_rate * gradient

            # 収束判定：alphaの変化が許容誤差未満なら終了
            if np.linalg.norm(alpha - alpha_old) < tol:
                print(f"Alpha has converged after {iteration + 1} iterations.")
                break
        
        # αを正規化
        alpha /= np.sum(alpha, axis=1, keepdims=True)

        return alpha