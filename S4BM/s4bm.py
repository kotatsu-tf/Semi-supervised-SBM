import numpy as np
from scipy.special import digamma, psi

class S4BM:
    def __init__(self, n_blocks=10, max_iter=2000, random_state=None):
        self.n_blocks = n_blocks
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, data, categories):
        # ノードの総数をdataの行数（または列数）から取得
        n_nodes = data.shape[0]
        n_blocks = self.n_blocks
        assigned_clusters = np.zeros((n_nodes, n_blocks))
                
        # カテゴリ行列を作成
        category_matrix = self._create_category_matrix(categories, n_nodes)

        # S4BLアルゴリズム実行（パラメータ更新）
        assigned_clusters = self.S4BL(data, n_blocks, category_matrix)
    

    
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
        
        tau = np.random.rand(N, K)  # N×Kの一様分布からのサンプリング
        gamma = np.random.rand(N, M)  # N×Mの一様分布からのサンプリング
        alpha = np.random.rand(M, K)  # M×Kの一様分布からのサンプリング

        rho = np.ones(K)  # 各要素が1のK次元ベクトルrho0

        # 1パターン目
        eta1 = np.array([0.6*rp + 0.4*rn + 0.5*ro])
        mu1 = np.array([0.4*rp + 0.6*rn + 0.5*ro])
        
        # 2パターン目
        eta2 = np.array([0.4*rp + 0.6*rn + 0.5*ro])
        mu2 = np.array([0.6*rp + 0.4*rn + 0.5*ro])

        return tau, gamma, alpha, rho, mu1, eta1, mu2, eta2

    
    def _S4BL(self, data, K, C):
        """
        S4BMのパラメータ推定アルゴリズム。
        data: 隣接行列
        K: ブロック数
        C: カテゴリ行列
        """

        tau, gamma, alpha, rho, mu1, eta1, mu2, eta2 = self._initialize_parameters(data, data.size, K, C)
        


    def _update_tau(self, alpha, gamma, eta, mu, rho, data):
        """
        各ノードが各ブロックに属する確率τ（タウ）を更新する関数。

        パラメータ:
        - alpha (numpy.ndarray): ラベルとブロックの関連性を示すパラメータ、形状は (M, K)
        - beta (numpy.ndarray): β値の計算結果、形状は (K,)
        - gamma (numpy.ndarray): γ確率の行列、形状は (N, M)
        - eta (numpy.ndarray): リンクが存在する確率、形状は (3,) (正、負、無リンク)
        - mu (numpy.ndarray): 異なるブロック間のリンクが存在する確率、形状は (3,)
        - rho (numpy.ndarray): 各ブロックの比率、形状は (K,)
        - data (numpy.ndarray): 隣接行列、形状は (N, N)

        戻り値:
        - numpy.ndarray: 更新されたτ行列、形状は (N, K)
        """
        N, K = gamma.shape
        tau = np.zeros((N, K))
        beta = np.sum(np.exp(alpha), axis=0)
        
        # 全ノードに対して更新を行う
        for i in range(N):
            for k in range(K):
                sum_gamma_alpha = np.dot(gamma[i], alpha[:, k])
                beta_term = 1 / np.dot(beta, tau[i]) * beta[k]
                tau_ik = psi(rho[k]) - psi(np.sum(rho)) + sum_gamma_alpha - beta_term
                
                for j in range(N):
                    if i != j:
                        tau_ik += np.sum(tau[j] * (digamma(eta) - digamma(np.sum(eta))) * data[i, j])
                        tau_ik += np.sum(tau[j] * (digamma(mu) - digamma(np.sum(mu))) * (1 - data[i, j]))
                
                tau[i, k] = np.exp(tau_ik)
        
        # 正規化
        tau /= np.sum(tau, axis=1, keepdims=True)
        
        return tau
    
    def _update_gamma(self, tau, alpha):
        """
        各ノードとラベルに対するγ（ガンマ）確率を更新する関数。

        パラメータ:
        - alpha (numpy.ndarray): αパラメータの行列で、形状は (M, K)。ラベルとブロックの関連性を示す。
        - tau (numpy.ndarray): τ確率の行列で、形状は (N, K)。ノードが各ブロックに属する確率を示す。

        戻り値:
        - numpy.ndarray: 更新されたγ行列で、形状は (N, M)。
        """
        # 各ノードとラベルに対して指数部分を計算
        exp_sum = np.exp(np.dot(tau, alpha.T))  # 形状 (N, M)
        
        # 確率として正規化
        gamma = exp_sum / np.sum(exp_sum, axis=1, keepdims=True)
        
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
        rho = np.sum(tau, axis=0) + rho0
        
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
        n_blocks = tau.shape[1]
        eta = np.zeros_like(eta0)
        for h in [-1, 0, 1]:  # リンクの種類: 負のリンク, リンクなし, 正のリンク
            link_type = h + 1  # インデックス調整
            for k in range(n_blocks):
                for l in range(n_blocks):
                    if k != l:
                        continue
                    mask = data == h
                    eta[link_type] += np.sum(mask * np.dot(tau[:, k:k+1], tau[:, l:l+1].T))
        eta += eta0
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
        n_blocks = tau.shape[1]
        mu = np.zeros_like(mu0)
        for h in [-1, 0, 1]:  # リンクの種類: 負のリンク, リンクなし, 正のリンク
            link_type = h + 1  # インデックス調整
            for k in range(n_blocks):
                for l in range(n_blocks):
                    if k == l:
                        continue
                    mask = data == h
                    mu[link_type] += np.sum(mask * np.dot(tau[:, k:k+1], tau[:, l:l+1].T))
        mu += mu0
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
        n_categories, n_blocks = alpha.shape
        
        for iteration in range(iterations):
            gradient = np.zeros_like(alpha)  # 勾配を初期化
            
            # alphaの勾配を計算
            for c in range(n_categories):
                for k in range(n_blocks):
                    for i in range(tau.shape[0]):
                        gradient[c, k] += gamma[i, c] * (tau[i, k] - np.exp(np.dot(alpha[c, :], tau[i, :])) / np.sum(np.exp(np.dot(alpha, tau[i, :]))))
            
            # 勾配降下法を使用してalphaを更新
            alpha += learning_rate * gradient
        
        return alpha