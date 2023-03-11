# 23.1.26 frog77
import requests
import json
def cspider(oid):
    is_end = False
    page = 1
    ans = []
    while True:
        Ntstr = 'https://api.bilibili.com/x/v2/reply/main?next=' #至is_end == False
        part2 = '&type=1&oid='
        oid = str(oid)
        mode = '&mode=3'
        Ntstr = Ntstr + str(page) + part2 + oid + mode
        resp = requests.get(Ntstr)
        resp = resp.text
        resp = json.loads(resp)
        message = resp['message']
        if message != '0':
            return ans
        data = resp['data']
        # print(page, data['cursor']['is_end'])
        is_end = bool(data['cursor']['is_end'])
        if is_end:
            break
        page += 1
        # print(data['replies'][0])
        # replies = data['replies'][0]['replies']
        #
        # for key in replies[0].keys():
        #     print(key, replies[0][key])#子评论
        ansp = []
        for items in data['replies']:
            ansp.append(items['content']['message'])
        ans += ansp
    return ans
