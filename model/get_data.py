from model.config import *


# 将签名者的数据打乱的数据打乱
# path_person_all[i] 里面依次存放的是 真签名 随机伪造签名 刻意伪造签名
nums_people = len(os.listdir(path_g))
path_people_all = [[] for i in range(nums_people)]

for index, i in enumerate(os.listdir(path_g)):
    dir_path = os.path.join(path_g, i)
    path_people_all[index].append(dir_path)

for index, i in enumerate(os.listdir(path_f)):
    dir_path = os.path.join(path_f, i)
    path_people_all[index].append(dir_path)

for index, i in enumerate(os.listdir(path_h)):
    dir_path = os.path.join(path_h, i)
    path_people_all[index].append(dir_path)

# 将原始的数据集打乱
random.shuffle(path_people_all)

# sig_people_all[i][j][k]
# 表示第i种类别的第j个人的第k个签名， i = 0,1,2 分别表示 真签名， 随机伪造签名， 刻意伪造签名
sig_people_all = [[], [], []]

for i in range(3):
    sig_people_all[i] = [[] for i in range(nums_people)]
for i, path in enumerate(path_people_all):
    sig_people_all[0][i] = [os.path.join(path[0], x) for x in os.listdir(path[0])]
    sig_people_all[1][i] = [os.path.join(path[1], x) for x in os.listdir(path[1])]
    sig_people_all[2][i] = [os.path.join(path[2], x) for x in os.listdir(path[2])]


# 生成了g-g对，pair_g_g[i] 表示第i个人的g-g对，36张真签名，生成 (1 + 2 + ... + 35) * 35 / 2 = 630 张
# 生成了g-f对，pair_g_g[i] 表示第i个人的g-f对，36张真签名，18张随机伪造签名， 36 * 18 = 648 张
# 生成了g-h对，pair_g_g[i] 表示第i个人的g-h对，36张真签名，18张刻意伪造签名， 36 * 18 = 648 张
pair_g_g = [[] for i in range(nums_people)]
pair_g_f = [[] for i in range(nums_people)]
pair_g_h = [[] for i in range(nums_people)]

for i, path, in enumerate(sig_people_all[0]):
    pair_g_g[i] = list(list(itertools.combinations(path, 2)))

for p in range(nums_people):
    for g in sig_people_all[0][p]:
        for f in sig_people_all[1][p]:
            pair_g_f[p].append((g, f))

for p in range(nums_people):
    for g in sig_people_all[0][p]:
        for h in sig_people_all[2][p]:
            pair_g_h[p].append((g, h))


def split_pair(pairs, p1, p2):
    """
    :param pairs: 数据集
    :param p1: 训练集的分割点
    :param p2: 验证集的分割点
    :return: 分割好的训练集，验证集，测试集
    """
    tra, val, tes = [], [], []
    for i, pair in enumerate(pairs):
        if i < p1:
            tra.extend(pair)
        elif i >= p2:
            tes.extend(pair)
        else:
            val.extend(pair)
    return tra, val, tes


tra_g_g_pair, val_g_g_pair, tes_g_g_pair = split_pair(pair_g_g, split1, split2)
tra_g_f_pair, val_g_f_pair, tes_g_f_pair = split_pair(pair_g_f, split1, split2)
tra_g_h_pair, val_g_h_pair, tes_g_h_pair = split_pair(pair_g_h, split1, split2)

# 展示训练数据集
print("训练集，验证集，测试集中的真签名-真签名的数据对分别有：")
print(len(tra_g_g_pair), len(val_g_g_pair), len(tes_g_g_pair))
print("训练集，验证集，测试集中的真签名-随机伪造数据对分别有：")
print(len(tra_g_f_pair), len(val_g_f_pair), len(tes_g_f_pair))
print("训练集，验证集，测试集中的真签名-刻意伪造数据对分别有：")
print(len(tra_g_h_pair), len(val_g_h_pair), len(tes_g_h_pair))
