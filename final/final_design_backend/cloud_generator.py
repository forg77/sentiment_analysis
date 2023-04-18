from wordcloud import WordCloud
import pandas as pd
import jieba
sentiments = ["neutral", "angry", "happy", "surprise", "sad", "fear"]
colors = {'angry':'orange','happy':'green','surprise':'red','sad':'purple','fear':'brown'}
a = pd.read_csv("new_dict.csv")
a_vol = a['vocab'].values
a_cls = a['label'].values
lena = len(a_vol)
dictz = {}
for i in range(lena):
    dictz[a_vol[i]] = int(a_cls[i])
def work(bv,worklist):
    ans = []
    dict_r = {}
    FILENAME = bv + 'analysis_'
    for item in worklist:
        str_l = jieba.cut(item,cut_all=True)
        for word in str_l:
            if dictz.get(word,-1) != -1:
                ans.append(word)
                dict_r[word] = colors[sentiments[dictz[word]]]
    # for i in range(1,6):
    #     v = WordCloud(background_color="white",font_path="/usr/share/fonts/truetype/arphic/ukai.ttc").generate(" ".join(ans[i]))
    #
    #     v.to_file(FILENAME+str(i)+'.png')
    #     ret[i] = FILENAME+str(i)+'.png'
    wc = WordCloud(background_color="white",font_path="/usr/share/fonts/truetype/arphic/ukai.ttc",width=1400,height=200)
    wc.generate(" ".join(ans))
    wc.recolor(color_func=lambda word, font_size, position, orientation, font_path, random_state=None: dict_r.get(word, 'black'))
    wc.to_file('/home/ubuntu/final_design_frontend/public/img/'+FILENAME+'.png')
    ret = '/img/'+FILENAME+'.png'
    # ret[0] = ''
    return ret
