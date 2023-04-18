# 23.1.26 frog77

import requests
from bs4 import BeautifulSoup
import re
import json
def dspider(CIDstr):
    Ntstr = 'https://comment.bilibili.com/'
    tail = '.xml'
    Ntstr = Ntstr + str(CIDstr) + tail
    resp = requests.get(Ntstr)
    resp.encoding = 'utf-8'
    soup = BeautifulSoup(resp.text, 'xml')
    d = soup.find_all('d')
    resp1 = []
    for item in d:
        item = str(item)
        item = re.search('">(.)*</', item)
        resp1.append(item.group()[2:-2])
    return resp1