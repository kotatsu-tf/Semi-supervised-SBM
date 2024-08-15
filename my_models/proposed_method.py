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
    def __init__(self, n_blocks=10, max_iter=2000, random_state=None, is_show_params_history=False):
        self.n_blocks = n_blocks
        self.max_iter = max_iter
        self.random_state = random_state
        self.is_show_params_history = is_show_params_history
        self.rng = np.random.RandomState(random_state)

    def fit_transform(self, data, categories=None):
        start_time = time.time()
        # ノードの総数をdataの行数（または列数）から取得
        n_nodes = data.shape[0]
        n_blocks = self.n_blocks
        assigned_clusters = np.zeros((n_nodes, n_blocks))

        # カテゴリ行列を作成
        category_matrix = self._create_category_matrix(categories, n_nodes)

        # S4BLアルゴリズム実行（パラメータ更新）
        assigned_clusters = self._S4BL(data, n_blocks, category_matrix)

        end_time = time.time()
        elapased_time = end_time - start_time
        print(f"実行時間： {elapased_time:.2f} sec")
        return assigned_clusters
    
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
    
    def _initialize_parameters(self,data, N, K, M):
        """
        パラメータtau, gamma, mu, eta, rho, alphaの初期化
        data: 隣接行列
        N: ノード数
        K: ブロック数
        M: カテゴリ数
        """
        rp, rn, ro = self._calculate_link_ratios(data)
        
        tau = self.rng.rand(N, K)  # N×Kの一様分布からのサンプリング
        gamma = self.rng.rand(N, M)  # N×Mの一様分布からのサンプリング
        alpha = self.rng.rand(M, K)  # M×Kの一様分布からのサンプリング

        rho = np.ones(K)  # 各要素が1のK次元ベクトルrho0

        # 1パターン目
        eta1 = np.array([0.6*rp, 0.4*rn, 0.5*ro])
        mu1 = np.array([0.4*rp, 0.6*rn, 0.5*ro])
        
        # 2パターン目
        eta2 = np.array([0.4*rp, 0.6*rn, 0.5*ro])
        mu2 = np.array([0.6*rp, 0.4*rn, 0.5*ro])

        return tau, gamma, alpha, rho, mu1, eta1, mu2, eta2

    
    def _S4BL(self, data, n_blocks, C):
        """
        S4BMのパラメータ推定アルゴリズム。
        data: 隣接行列
        n_blocks: ブロック数
        C: カテゴリ行列
        """
        n_nodes = data.shape[0]


        tau, gamma, alpha, rho0, mu0_1, eta0_1, mu0_2, eta0_2 = self._initialize_parameters(data, n_nodes, n_blocks, C.shape[1])
        
        rho = rho0
        mu1 = mu0_1
        mu2 = mu0_2
        eta1 = eta0_1
        eta2 = eta0_2
        tau_his, gamma_his, alpha_his, rho_his, mu_his, eta_his = [], [], [], [], [], []

        previous_elbo = self._calculate_elbo(data, tau, gamma, alpha, rho, mu1, eta1)
        for iteration in range(self.max_iter):

            tau_his.append(copy.deepcopy(tau))
            gamma_his.append(copy.deepcopy(gamma))
            rho_his.append(copy.deepcopy(rho))
            alpha_his.append(copy.deepcopy(alpha))
            eta_his.append(copy.deepcopy(eta1))
            mu_his.append(copy.deepcopy(mu1))
            
            tau = self._update_tau(alpha, gamma, eta1, mu1, rho, tau, data, C)
            gamma = self._update_gamma(gamma, tau, alpha, C)
            rho = self._update_rho(rho0, tau)            
            alpha = self._update_alpha(tau, gamma, alpha)            
            eta1 = self._update_eta(tau, data, eta0_1)            
            mu1 = self._update_mu(tau, data, mu0_1)


            print(f'Converged after {iteration + 1} iterations.')
        
        if self.is_show_params_history:
            self._make_graph(tau_his, gamma_his, rho_his, alpha_his, eta_his, mu_his)
        Z = np.argmax(tau, axis=1)
        return Z

    
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

    def _update_tau(self, alpha, gamma, eta, mu, rho, tau, data, C):
        """
        各ノードが各ブロックに属する確率τ（タウ）を更新する関数。

        パラメータ:
        - alpha (numpy.ndarray): ラベルとブロックの関連性を示すパラメータ、形状は (M, K)
        - gamma (numpy.ndarray): γ確率の行列、形状は (N, M)
        - eta (numpy.ndarray): リンクが存在する確率、形状は (3,) (正、負、無リンク)
        - mu (numpy.ndarray): 異なるブロック間のリンクが存在する確率、形状は (3,)
        - rho (numpy.ndarray): 各ブロックの比率、形状は (K,)
        - tau (numpy.ndarray): 各ノードが各ブロックに属する確率を表す配列、形状は (N, K)
        - data (numpy.ndarray): 隣接行列、形状は (N, N)

        戻り値:
        - numpy.ndarray: 更新されたτ行列、形状は (N, K)
        """
        N, K = tau.shape
        log_tau = np.zeros((N, K))

        log_rho = digamma(rho) - digamma(np.sum(rho))
        beta = np.sum(np.exp(alpha), axis=0)
        
        for i in range(N):
            for k in range(K):
                log_tau_ik = log_rho[k]

                for c in range(gamma.shape[1]):
                    log_tau_ik += gamma[i, c] * (alpha[c, k] - (1.0 / np.dot(beta, tau[i].T)) * beta[k])

                for j in range(N):
                    if i != j:
                        if data[i, j] == 1:
                            log_tau_ik += tau[j, k] * (digamma(eta[0]) - digamma(np.sum(eta)))
                        elif data[i, j] == -1:
                            log_tau_ik += tau[j, k] * (digamma(eta[1]) - digamma(np.sum(eta)))
                        else:
                            log_tau_ik += tau[j, k] * (digamma(eta[2]) - digamma(np.sum(eta)))

                        for l in range(K):
                            if k != l:
                                if data[i, j] == 1:
                                    log_tau_ik += tau[j, l] * (digamma(mu[0]) - digamma(np.sum(mu)))
                                elif data[i, j] == -1:
                                    log_tau_ik += tau[j, l] * (digamma(mu[1]) - digamma(np.sum(mu)))
                                else:
                                    log_tau_ik += tau[j, l] * (digamma(mu[2]) - digamma(np.sum(mu)))

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


    
    def _update_gamma(self, gamma, tau, alpha, C):
        """
        各ノードとラベルに対するγ（ガンマ）確率を更新する関数。

        パラメータ:
        - alpha (numpy.ndarray): αパラメータの行列で、形状は (M, K)。ラベルとブロックの関連性を示す。
        - tau (numpy.ndarray): τ確率の行列で、形状は (N, K)。ノードが各ブロックに属する確率を示す。

        戻り値:
        - numpy.ndarray: 更新されたγ行列で、形状は (N, M)。
        """
        # 各ノードとラベルに対して指数部分を計算
        N, K = tau.shape             
        M = alpha.shape[0]
        

        for i in range(N):
            for c in range(M):
                top = 0
                for k_ in range(K):
                    top += alpha[c, k_] * (1.0 / tau[i, k_])
                top = np.exp(top)

                bottom = 0
                for c_ in range(M):
                    for k_ in range(K):
                        bottom += tau[i,k_] * np.exp(alpha[c_, k_])
                gamma[i, c] = top / bottom

        for i in range(N):
            if np.sum(C[i]) > 0:
                gamma[i] = C[i]

        # 確率として正規化
        gamma /= np.sum(gamma, axis=1, keepdims=True)
        
        return gamma


    
    def _update_rho(self, rho0, tau):
        """
        式(12)に基づいてブロックの割合ρを更新する。

        Parameters:
        - tau: 各ノードが各ブロックに属する確率を表す配列、形状は(n_nodes, n_blocks)
        - rho0: ρの初期値を表す配列、形状は(n_blocks,)

        Returns:
        - rho: 更新されたブロックの割合を表す配列、形状は(n_blocks,)
        """
        # 各ブロックに対するノードの割り当て確率の合計を計算し、rho0を加算
        rho = rho0 + np.sum(tau, axis=0) 

        # 正規化
        rho /= np.sum(rho)
        
        return rho
    
    def _update_eta(self, tau, data, eta0):
        """
        式(13)に基づいてブロック内のリンク確率ηを更新する。

        Parameters:
        - tau: 各ノードが各ブロックに属する確率を表す配列、形状は(n_nodes, n_blocks)
        - data: ネットワークの隣接行列、形状は(n_nodes, n_nodes)。要素は1, -1, 0。
        - eta0: ηの初期値、形状は(3,)。正のリンク、負のリンク、リンクなしの確率。

        Returns:
        - eta: 更新されたブロック内のリンク確率、形状は(3,)。
        """
        N, K = tau.shape
        eta = np.zeros_like(eta0)
        for i in range(N-1):
            for j in range(i+1, N):
                for k in range(K):
                    if data[i, j] == 1:
                        eta[0] += tau[i,k]*tau[j,k]
                    elif data[i, j] == -1:
                        eta[1] += tau[i,k]*tau[j,k]
                    else:
                        eta[2] += tau[i,k]*tau[j,k] 
        eta += eta0
        # 正規化
        eta /= np.sum(eta)

        return eta
                        
    
    def _update_mu(self, tau, data, mu0):
        """
        式(14)に基づいてブロック間のリンク確率μを更新する。

        Parameters:
        - tau: 各ノードが各ブロックに属する確率を表す配列、形状は(n_nodes, n_blocks)
        - data: ネットワークの隣接行列、形状は(n_nodes, n_nodes)。要素は1, -1, 0。
        - mu0: μの初期値、形状は(3,)。正のリンク、負のリンク、リンクなしの確率。

        Returns:
        - mu: 更新されたブロック間のリンク確率、形状は(3,)。
        """

        N, K = tau.shape
        mu = np.zeros_like(mu0)
        for i in range(N - 1):
            for j in range(i+1, N):
                for q in range(K):
                    for l in range(K):
                        if q != l:
                            if data[i, j] == 1:
                                mu[0] += tau[i,q]*tau[j,l]
                            elif data[i, j] == -1:
                                mu[1] += tau[i,q]*tau[j,l]
                            else:
                                mu[2] += tau[i,q]*tau[j,l]
        mu += mu0
        # 正規化
        mu /= np.sum(mu)
        return mu
    
    def _update_alpha(self, tau, gamma, alpha, learning_rate=0.01, iterations=100):
        """
        カテゴリとブロック間の関係パラメータαを式(15)に基づいて更新する関数。

        パラメータ:
        - tau: 各ノードが各ブロックに属する確率、形状は(n_nodes, n_blocks)。
        - gamma: 各ノードが各カテゴリに属する確率、形状は(n_nodes, n_categories)。
        - alpha: カテゴリとブロック間の初期関係パラメータ、形状は(n_categories, n_blocks)。
        - learning_rate: 勾配降下法の学習率。
        - iterations: 勾配降下法の反復回数。

        戻り値:
        - alpha: 更新されたカテゴリとブロック間の関係パラメータ。
        """
        M, K = alpha.shape
        N = tau.shape[0]
        
        for _ in range(iterations):
            gradient = np.zeros_like(alpha)  # 勾配を初期化
            
            # alphaの勾配を計算
            for c in range(M):
                for k in range(K):
                    for i in range(N):
                        bottom = 0
                        for c_ in range(M):
                            for k_ in range(K):
                                bottom += tau[i, k_] * np.exp(alpha[c_, k_])
                        gradient[c, k] += gamma[i, c] * ((tau[i, k] - (tau[i, k] * np.exp(alpha[c, k]))) / bottom)
            
            # 勾配降下法を使用してalphaを更新
            alpha += learning_rate * gradient
        
        alpha /= np.sum(alpha)
        return alpha