# 23.2.11 frog77
# 向量的拼接方法：直接使用 sbert embedding模型用 https://huggingface.co/sentence-transformers/distiluse-base-multilingual-cased-v2
#              结巴分词 用dutir情感词典取得对应7类21种情感倾向的2*21维向量 其中，0.5 0.5代表中性，1 0 代表褒义，0 1代表贬义，1 1代表兼有褒贬两性。
from sentence_transformers import SentenceTransformer
import jieba
import numpy as np
import pymysql
import time
result_list = ['PA', 'PE', 'PD', 'PH', 'PG', 'PB', 'PK', 'NA', 'NB',
               'NJ', 'NH', 'PF', 'NI', 'NC', 'NG', 'NE', 'ND', 'NN',
               'NK', 'NL', 'PC']
model = SentenceTransformer("sentence-transformers/distiluse-base-multilingual-cased-v2")
ebds = np.zeros((42,), dtype=np.float32)
db = pymysql.connect(host='localhost',
                     user='root',
                     password='1q2w3e4r',
                     database='senti')
cursor = db.cursor()


def vector(sentences):
    embedding = model.encode(sentences)
    words = jieba.lcut_for_search(sentences)
    ebds1 = ebds
    ebds1 = np.concatenate((ebds1, embedding))
    for items in words:
        ebds1 = senti(ebds1, items)
    return ebds1


def senti(ebdsi, word):
    # print(word)
    sql = "SELECT * FROM sheet1 WHERE 词语 = '%s'" % word
    cursor.execute(sql)
    result = cursor.fetchone()
    # print(result)
    if result is not None:
        status = result[1]
        strength = result[2]
        attitude = result[3]
        aux_status = result[4]
        aux_strength = result[5]
        aux_attitude = result[6]
        cnt = 0
        for items in result_list:
            if items == status:
                if attitude == '0':
                    ebdsi[cnt * 2] += 0.5 * float(strength)
                    ebdsi[cnt * 2 + 1] += 0.5 * float(strength)
                if attitude == '1':
                    ebdsi[cnt * 2] += 1.0 * float(strength)
                if attitude == '2':
                    ebdsi[cnt * 2 + 1] += 1.0 * float(strength)
                if attitude == '3':
                    ebdsi[cnt * 2] += 1.0 * float(strength)
                    ebdsi[cnt * 2 + 1] += 1.0 * float(strength)
                break
            cnt += 1
        cnt = 0
        if aux_status is not None:
            for items in result_list:
                if items == aux_status:
                    if aux_attitude == '0':
                        ebdsi[cnt * 2] += 0.5 * float(aux_strength)
                        ebdsi[cnt * 2 + 1] += 0.5 * float(aux_strength)
                    if aux_attitude == '1':
                        ebdsi[cnt * 2] += 1.0 * float(aux_strength)
                    if aux_attitude == '2':
                        ebdsi[cnt * 2 + 1] += 1.0 * float(aux_strength)
                    if aux_attitude == '3':
                        ebdsi[cnt * 2] += 1.0 * float(aux_strength)
                        ebdsi[cnt * 2 + 1] += 1.0 * float(aux_strength)
                    break
                cnt += 1
    return ebdsi


if __name__ == '__main__':
    # ebd = model.encode("明月几时有，把酒问青天，不知天上宫阙，今夕是何年")
    # print("transformer test done", ebd)

    cursor1 = db.cursor()
    # sql = "SELECT * FROM sheet1 WHERE 词语 = '一琴一鹤'"
    # cursor1.execute(sql)
    # result = cursor1.fetchone()
    # print(result)
    # ('一琴一鹤', 'PH', '7', '1', 'PD', '5', '1')

    # sql1 = "SELECT * FROM sheet1 WHERE 词语 = '114514'"
    # cursor1.execute(sql1)
    # result = cursor1.fetchone()
    # if result is None:
    #     print(1)
    # print("sql test done")
    #
    t1 = time.time()
    v1 = vector("冈斯这部片子不知道修复了多少次，精神可嘉啊，也是史诗级的神片")
    # print(v1)
    t2 = time.time()
    print(t2-t1) # 0.5551683902740479
    t1 = time.time()
    str1 = open('longtxt.txt', 'r', encoding='utf-8')
    str1 = str1.read()
    v2 = vector(str1)
    # print(v2)
    t2 = time.time()
    print(t2-t1) #0.6400654315948486 可见文字长度对处理事件影响较小 平均每次相应需要 0.5-0.6 s

