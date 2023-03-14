import classify
import requests
from spider import bv2av as b2a
from spider import comment
from spider import danmuku
import json
import time
import random
import pandas as pd
# 选取比较近期的弹幕和评论 2020-10-20 ~ 2023-03-04
# nowav = b2a.dec("BV15g4y1E7Xz")
# print(nowav)
# 822955175
# 2023-03-04 08:30:00
# pastbv = b2a.enc(800000005)
# print(pastbv)
# 2020-10-20 01:47:04
# 找时间实现一个代理池 随机挂一个跑


UPPER_BOUND = 822955175
LOWER_BOUND = 800000005
WRITEPATH = r'.\new_data.csv'
dlist = []
clist = []

for i in range(LOWER_BOUND,UPPER_BOUND+1):

    time.sleep(random.randint(1,3)) # 防封
    ansc = comment.cspider(i)
    bvid = b2a.enc(i)
    Ntstr = 'https://api.bilibili.com/x/player/pagelist?bvid='
    Ntstr = Ntstr + bvid
    resp = requests.get(Ntstr)
    resp = resp.text
    resp = json.loads(resp)
    CHECK = resp['code']
    if CHECK != 0 :
        continue
    CIDstr = resp['data'][0]['cid']
    ansd = danmuku.dspider(CIDstr)
    for item in ansd:
        clss = classify.cls(item)
        if len(clss) != 0:
            dlist.append([item,clss[0]])
    for item in ansc:
        clss = classify.cls(item)
        if len(clss) != 0:
            clist.append([item,clss[0]])

    if i % 5 == 0:
        anslist_content = dlist + clist

        dlist = []
        clist = []

        # print(anslist_content)
        frame = pd.DataFrame(anslist_content,columns=['text','label'])
        frame.to_csv(WRITEPATH,mode='a',index=False,header=False)
        print("5 done")