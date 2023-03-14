import spider
import classifier
import time
import matplotlib.pyplot as plt
label = ['neutral','angry','happy','surprise','sad','fear']
if __name__ == '__main__':
    while True:
        print(1)
        bvstr = input()
        dlist , clist = spider.func(bvstr)
        lend = len(dlist)
        lenc = len(clist)

        # print(2)
        t0 = time.time()
        d_pred , c_pred = classifier.work(dlist,clist)

        t1 = time.time()
        print(t1-t0)
        d_dict = [0]*6
        c_dict = [0]*6
        d_ans = [[],[],[],[],[],[]]
        c_ans = [[],[],[],[],[],[]]
        for item in d_pred:
            d_dict[item] += 1
        for i in range(lend):
            d_ans[d_pred[i]].append(dlist[i])
        for item in c_pred:
            c_dict[item] += 1
        for i in range(lenc):
            c_ans[c_pred[i]].append(clist[i])
        plt.pie(c_dict,labels=label)
        plt.title(bvstr + " comment sentiment percentage")
        plt.savefig('pics//'+bvstr+'_comment.png')
        plt.clf()
        plt.pie(d_dict,labels=label)
        plt.title(bvstr + " danmuku sentiment percentage")
        plt.savefig('pics//'+bvstr+'_danmuku.png')
        plt.clf()
        for i in range(6):
            print("##############__danmuku",i,"danmuku__################",label[i])
            print(d_ans[i])
        for i in range(6):
            print("##############__comment",i,"comment__################",label[i])
            print(c_ans[i])
        print("done")