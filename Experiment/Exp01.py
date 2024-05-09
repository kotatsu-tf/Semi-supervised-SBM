# 人工データを用いたS4BMの動作実験
import numpy as np
from my_models import S4BM

print(sys.path)
model = S4BM()
print(model)
# # CSVファイルのパスを指定
# file_path = 'data/input/artificial_data_1.csv'
# # CSVファイルを読み込んでNumPy配列に変換
# data = np.loadtxt(file_path, delimiter=',')
# print(data)

# model = S4BM.S4BM()
# print(model)