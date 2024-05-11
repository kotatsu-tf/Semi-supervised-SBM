import unittest
import numpy as np
from my_models import S4BM

class TestS4BM(unittest.TestCase):
    def setUp(self):
        self.model = S4BM(n_blocks=3, max_iter=100, random_state=42)

    def test_initialization(self):
        """クラスの初期化が正しく行われるかテスト"""
        self.assertEqual(self.model.n_blocks, 3)
        self.assertEqual(self.model.max_iter, 100)
        self.assertIsNotNone(self.model.random_state)

    def test_create_category_matrix(self):
        """カテゴリ行列の生成が正しいかテスト"""
        categories = [[0, 1], [2]]
        n_nodes = 3
        expected_matrix = np.array([[1, 0], [1, 0], [0, 1]])
        category_matrix = self.model._create_category_matrix(categories, n_nodes)
        np.testing.assert_array_equal(category_matrix, expected_matrix)

    def test_initialize_parameters(self):
        """パラメータの初期化が適切かテスト"""
        n_nodes = 4
        n_blocks = 3
        n_categories = 2
        data = np.random.randint(-1, 2, size=(n_nodes, n_nodes))
        tau, gamma, alpha, rho, mu1, eta1, mu2, eta2 = self.model._initialize_parameters(data, n_nodes, n_blocks, n_categories)
        self.assertEqual(tau.shape, (n_nodes, n_blocks))
        self.assertEqual(gamma.shape, (n_nodes, n_categories))
        self.assertEqual(alpha.shape, (n_categories, n_blocks))
        self.assertTrue(np.all(rho > 0))

if __name__ == '__main__':
    unittest.main()
