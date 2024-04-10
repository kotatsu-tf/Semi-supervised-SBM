import numpy as np

class S4BM:
    def __init__(self, n_blocks, n_iter=2000, random_state=None):
        self.n_blocks = n_blocks
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, data, categories):
        # ノードの総数をdataの行数（または列数）から取得
        n_nodes = data.shape[0]
                
        # カテゴリ行列を作成
        category_matrix = self._create_category_matrix(categories, n_nodes)

    
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

    
    def S4BL(self, data, K, C):
        """
        S4BMのパラメータ推定アルゴリズム。
        data: 隣接行列
        K: ブロック数
        C: カテゴリ行列
        """

        tau, gamma, alpha, rho, mu1, eta1, mu2, eta2 = self._initialize_parameters(data, data.size, K, C)
