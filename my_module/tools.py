from sklearn.metrics import normalized_mutual_info_score

# 2つの変数の和を計算する自作関数
def my_sum(x,y):
    z = x + y
    return z

# 2つの変数の差を計算する自作関数
def my_dif(x,y):
    z = x - y
    return z

def cal_NMI(true_label, pred_label):
    return normalized_mutual_info_score(true_label, pred_label)