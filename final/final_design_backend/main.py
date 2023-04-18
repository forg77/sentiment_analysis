from flask import Flask, request,jsonify, url_for
from flask_cors import CORS
import Spider
import classifier
import cloud_generator
import matplotlib.pyplot as plt
app = Flask(__name__)
CORS(app)
sentiments = ["neutral", "angry", "happy", "surprise", "sad", "fear"]

@app.route('/api/search',methods=['GET'])
def work():
    # print(request.args.get('param1'))
    bvstr = request.args.get('param1')
    # bvstr = "BV1Ze4y1c7AH"
    dlist , clist = Spider.func(bvstr)
    dpred , cpred = classifier.work(dlist,clist)
    ans = {}
    lend = len(dlist)
    lenc = len(clist)
    d_cnt = [0] * 6
    c_cnt = [0] * 6
    ans["lend"] = lend
    ans["lenc"] = lenc
    cntd = 0
    cntc = 0
    for item in dpred:
        d_cnt[item] += 1
        cntd += 1
    for item in cpred:
        c_cnt[item] += 1
        cntc += 1


    ret = cloud_generator.work(bvstr,dlist+clist)
    if cntc != 0:
        plt.pie(c_cnt,labels=sentiments)
        plt.title(bvstr + " comment sentiment percentage")
        plt.savefig('/home/ubuntu/final_design_frontend/public/img/'+bvstr+'_comment.png')
        ans["path_per_c"] = '/img/'+bvstr+'_comment.png'
        plt.clf()
    if cntd != 0:
        plt.pie(d_cnt,labels=sentiments)
        plt.title(bvstr + " danmuku sentiment percentage")
        plt.savefig('/home/ubuntu/final_design_frontend/public/img/'+bvstr+'_danmuku.png')
        ans["path_per_d"] = '/img/'+bvstr+'_danmuku.png'
        plt.clf()
    ans["path"] = ret
    return jsonify(ans)


if __name__ == '__main__':
    app.run(port=12345)