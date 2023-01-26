# 2023.1.26 frog77
# -*- coding:utf-8 -*-
from danmuku import dspider
from comment import cspider
import requests
import json
from bv2av import dec
while True:
    print("输入BV号")
    Ntstr = 'https://api.bilibili.com/x/player/pagelist?bvid='
    BVstr = input()
    AVstr = dec(BVstr)
    print(AVstr)
    Ntstr = Ntstr + BVstr
    resp = requests.get(Ntstr)
    resp = resp.text
    resp = json.loads(resp)
    CHECK = resp['code']
    if CHECK != 0:
        print("检查网络连接，或BV号失效")
        continue

    CIDstr = resp['data'][0]['cid']
    danmuku_list = dspider(CIDstr) #获取弹幕
    comment_list = cspider(AVstr) #获取评论
