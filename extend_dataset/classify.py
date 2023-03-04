import jieba
import get_word_list as gwl
SHUFFLE = 0
a = gwl.solve(SHUFFLE)

dict_a = {}
for item in a:
    dict_a[item[0]] = (item[1],item[2])

# for items in
# stop word may be added someday
def cls(str1):
    str1_l = jieba.cut(str1,cut_all=True)
    ans = [0] * 6
    for item in str1_l:
        # print(item)
        check = dict_a.get(item,0)
        # print(check)
        if check == 0:
            continue
        else:
            ans[check[1]] += check[0] - (SHUFFLE - 1)
    maxi = 0
    ansp = []
    for i in range(1,6):
        if ans[i] > maxi:
            maxi = ans[i]
            ansp = []
            ansp.append(i)
        elif ans[i] == maxi:
            ansp.append(i)
    if len(ansp) > 2:
        ansp = []
    return ansp
