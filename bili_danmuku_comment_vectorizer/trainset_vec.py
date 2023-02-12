# 2023.2.12 frog77
import vec
import pandas as pd
WORKBOOKPATH = r"C:\Users\1323231\Desktop\毕业设计\评测数据集\train\usual_train.xlsx"
WRITEPATH = r"C:\Users\1323231\Desktop\毕业设计\评测数据集\train\vec_train.xlsx"
data = pd.read_excel(WORKBOOKPATH, sheet_name="Sheet1")
writer = pd.ExcelWriter(WRITEPATH)
to_vec = data[["文本"]]
# print(to_vec)
lenw = len(to_vec)
vec_list = []
print("done", lenw)
to_vec = to_vec.copy()
for i in range(lenw):
    str1 = str(to_vec["文本"][i])
    to_vec["文本"][i] = vec.vector(str1)
    print(i)
data[["文本"]] = to_vec
data.to_excel(writer, sheet_name="Sheet1")
writer.save()

