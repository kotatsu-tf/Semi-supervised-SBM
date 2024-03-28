import numpy as np

class S4BM:
    def __init__(self, n_blocks, n_iter=2000, random_state=None):
        self.n_blocks = n_blocks
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, data, categories):
        # ノードの総数をdataの行数（または列数）から取得
        n_nodes = data.shape[0]
        
        # カテゴリが提供されている場合のみ最大カテゴリ数を計算
        n_categories = max(categories.values()) + 1 if categories else 0
        
        # カテゴリ行列を作成
        category_matrix = self._create_category_matrix(categories, n_categories, n_nodes)

    def _create_category_matrix(self, categories, n_categories, n_nodes):
        """
        カテゴリ辞書とノードの総数からOne-hotエンコーディングされたカテゴリ行列を作成する。
        categories: {node_id: category} 形式の辞書
        n_categories: カテゴリの総数
        n_nodes: ノードの総数
        """
        category_matrix = np.zeros((n_nodes, n_categories))

        for node_id, category in categories.items():
            # カテゴリが指定されたノードに対してのみ、One-hotエンコーディングを適用
            category_matrix[node_id, category] = 1

        return category_matrix
    

    def calculate_link_ratios(self, data):
        total_links = data.size  # 隣接行列内の総要素数
        rp = np.sum(data == 1) / total_links  # 正のリンクの割合
        rn = np.sum(data == -1) / total_links  # 負のリンクの割合
        ro = np.sum(data == 0) / total_links  # リンクなしの割合
        return rp, rn, ro
    
    def initialize_parameters(self,data, N, K, M):
        """
        パラメータtau, gamma, mu, eta, rho, alphaの初期化
        data: 隣接行列
        N: ノード数
        K: ブロック数
        M: カテゴリ数
        """
        rp, rn, ro = self.calculate_link_ratios(data)
        
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

    
    def S4BL(self, data, K, C):
        """
        S4BMのパラメータ推定アルゴリズム。
        data: 隣接行列
        K: ブロック数
        C: カテゴリ行列
        """

