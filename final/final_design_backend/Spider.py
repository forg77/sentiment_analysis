# 2023.1.26 frog77
# -*- coding:utf-8 -*-
import json

import requests
from bv2av import dec
from comment import cspider
from danmuku import dspider


def func(BVstr):
    Ntstr = 'https://api.bilibili.com/x/player/pagelist?bvid='
    AVstr = dec(BVstr)
    print(AVstr)
    Ntstr = Ntstr + BVstr
    resp = requests.get(Ntstr)
    resp = resp.text
    resp = json.loads(resp)
    CHECK = resp['code']
    if CHECK != 0:
        return "检查网络连接，或BV号失效"


    CIDstr = resp['data'][0]['cid']
    danmuku_list = dspider(CIDstr) #获取弹幕
    comment_list = cspider(AVstr) #获取评论
    print("spider done")
    return danmuku_list,comment_list