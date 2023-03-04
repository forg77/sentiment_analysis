import pandas as pd

# 中立语料不好处理 这里就不另外更新了
# 1 angry 8344 + NA + ND + NN + NK + NL
# 2 happy 5379 + PA + PE
# 3 surprise 2086 + PC
# 4 sad 4990 + NB + NJ +NH + PF
# 5 fear 1220 + NI + NC + NG
list1 = ["NA", "ND", "NN", "NK", "NL"]
list2 = ["PA", "PE"]
list3 = ["PC"]
list4 = ["NB", "NJ", "NH", "PF"]
list5 = ["NI", "NC", "NG"]
a = pd.read_csv("senti.csv")
a_vol = a['词语'].values
a_pow = a['强度'].values
a_cls = a['情感分类'].values


def classify(key1):
    for item in list1:
        if key1 == item:
            return 1
    for item in list2:
        if key1 == item:
            return 2
    for item in list3:
        if key1 == item:
            return 3
    for item in list4:
        if key1 == item:
            return 4
    for item in list5:
        if key1 == item:
            return 5


def solve(x):
    if x > 9:
        return False

    z = len(a_vol)
    a_more_than_x = []
    for i in range(z):
        if a_pow[i] >= x:
            clsk = classify(a_cls[i])
            if clsk == None:
                continue
            a_more_than_x.append((a_vol[i], a_pow[i], clsk))
    return a_more_than_x
