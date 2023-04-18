from flask import Flask, jsonify
from flask_cors import CORS
import Spider
import classifier
import cloud_generator
# app = Flask(__name__)
# CORS(app)
sentiments = ["neutral", "angry", "happy", "surprise", "sad", "fear"]
# @app.route('/search')
def work():
    # print(request.args.get('param1'))
    # bvstr = request.args.get('param1')
    bvstr = "BV1Ze4y1c7AH"
    dlist , clist = Spider.func(bvstr)
    dpred , cpred = classifier.work(dlist,clist)
    ans = {}
    lend = len(dlist)
    lenc = len(clist)
    for i in range(lend):
        ans[dlist[i]] = sentiments[dpred[i]]
    for i in range(lenc):
        ans[clist[i]] = sentiments[cpred[i]]
    ret = cloud_generator.work(bvstr,dlist+clist)
    for i in range(1,6):
        ans["path"+str(i)] = ret[i]

    return jsonify(ans)

if __name__ == '__main__':
    # app.run(port=8081)
    bvstr = "BV1Ze4y1c7AH"
    dlist, clist = Spider.func(bvstr)
    dpred, cpred = classifier.work(dlist, clist)
    ans = {}
    lend = len(dlist)
    lenc = len(clist)
    for i in range(lend):
        ans[dlist[i]] = sentiments[dpred[i]]
    for i in range(lenc):
        ans[clist[i]] = sentiments[cpred[i]]
    ret = cloud_generator.work(bvstr, dlist + clist)
    for i in range(1, 6):
        ans["path" + str(i)] = ret[i]

