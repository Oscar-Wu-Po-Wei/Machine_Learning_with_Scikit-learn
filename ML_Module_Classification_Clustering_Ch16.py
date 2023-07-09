# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 20:37:15 2023

@author: oscar.wu
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split

# 決策樹
data = pd.read_csv("titanic.csv")
print(data.shape)
# 敘述統計
print(data.describe()) # Age count只有756而非1313，代表有遺漏值。
# 進一步檢查遺漏值
print(data.info())
data["SexCode"] = np.where(data["Sex"]=="male", 1, 0)

# 轉換欄位值成為數值
label_encoder = preprocessing.LabelEncoder()
encoded_class = label_encoder.fit_transform(data["PClass"])

data["SexCode"] = np.where(data["Sex"]=="male", 1, 0)

X = pd.DataFrame([data["SexCode"], encoded_class]).T
X.columns = ["SexCode", "PClass"]
y = data["Survived"]

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.25, random_state=1)
# 測試資料集25%，訓練資料集75%。
dtree = tree.DecisionTreeClassifier() # Scikit-learn套件之決策樹分類器物件。
dtree.fit(XTrain, yTrain) # fit()函數使用訓練資料集訓練模型。
print(dtree.score(XTest, yTest)) # 使用測試資料集測試準確度。

preds = dtree.predict_proba(X=XTest)
# print(preds.shape)
print(pd.crosstab(preds[:,0], columns=[XTest["PClass"], XTest["SexCode"]]))
# 傳回值是一個n行k列的矩陣，在第i行第j列為模型預測第i個樣本為j的機率。

# K Nearest Neighbor Algorithm (KNN)：分類預測→物以類聚(K個最近鄰居)。
# 使用K個最接近目標資料的資料來對目標資料進行歸類預測。
# Step 1.：計算目標資料與資料集中其他所有資料之距離。
# Step 2.：排序找出K個最近距離的資料。
# Step 3.：新資料分類為K個最近距離的資料的多數屬性之分類。
import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
# print(X)
X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
# X具有四個變數(多元)：四個屬性值。
target = pd.DataFrame(iris.target, columns=["target"])
# print(target)
y = target["target"] #取出數值內容不含欄位名稱。
# print(y)

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.33, random_state=1)

k = 3

knn = neighbors.KNeighborsClassifier(n_neighbors=k)

knn.fit(X, y)

print("準確率：", knn.score(XTest, yTest))

print(knn.predict(XTest))
print(yTest.values)

# 選擇K值
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target = pd.DataFrame(iris.target, columns=["target"])
y = target["target"]

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.33, random_state=1)

# 不同的K值會影響分類的準確度，可以使用迴圈執行多次不同的K值分類，找出最佳K值。
Ks = np.arange(1, round(0.2*len(XTrain)+1)) #一般K值上限不超過訓練資料集的20%。
accuracies = []
for k in Ks:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn.fit(X, y)
    accuracy = knn.score(XTest, yTest)
    accuracies.append(accuracy)
    
plt.plot(Ks, accuracies)
plt.show()

# K-Fold Cross Validation 交叉驗證：使用完整的資料集進行模型訓練。
# 功能：K值最佳化。
# 解決持久性驗證無法使用完整資料集進行模型訓練，交叉驗證可以充分使用所有資料集進行訓練。
# 將資料集分割2個或更多分隔區，將每個分隔區逐一作為測試資料集，其他所有分隔區則逐次做為訓練資料集。
# K-fold方法，將目標資料集隨機分割成相同大小的K個分隔區(或稱為「折」，Fold)，
# Step 1. 使用第1個分隔區作為測試資料集，其他K-1個分隔區作為訓練資料集；
# Step 2. 使用第2個分隔區作為測試資料集，其他K-1個分隔區作為訓練資料集；
# Step 3-K. 重複執行K次，組合建立最終的訓練模型，充分使用所有資料集進行訓練。
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import neighbors
from sklearn.model_selection import cross_val_score

iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
target = pd.DataFrame(iris.target, columns=["target"])
y = target["target"]

# 不同的K值會影響分類的準確度，可以使用迴圈執行多次不同的K值分類，找出最佳K值。
Ks = np.arange(1, round(0.2*len(X)+1)) #一般K值上限不超過訓練資料集的20%。
accuracies = []
for k in Ks:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, y, scoring="accuracy", cv=10) #代替knn.fit()
    accuracies.append(scores.mean())
    
plt.plot(Ks, accuracies)
plt.show()

# K-means 演算法 (K-means Clustering，K平均數分群)
# 分類與分群的比較：
# 分類：在已知資料集分類的情況下，替新東西進行分類。
# 分群：在根本不知道資料集分類的情況下，直接使用資料的特徵進行分類，也屬於一種分類，只不過我們並不知道結果的各個群組是哪一類東西(只知道特徵分類)。
# 屬於一種非監督式學習，不需要答案的標籤(標準答案)，只需要訓練資料集(X)即可。
# K-means 演算法原理：先找出K個群組的重心(Centroid)，資料集就以距離最近重心來分成群組，重新計算群組的新重心後，再度分群一次，重複操作以完成分群。
# Step 1. 依據資料集大小，決定適當的K個重心。
# Step 2. 計算資料集與重心的距離(公式與KNN計算距離的公式相同)，然後以距離最近各個重心的資料集來分成數個群組。
# Step 3. 重新計算群組資料集各個特徵的算術平均數做為新的重心(移動K個重心)。
# Step 4. 再次計算資料集與新重心的距離，然後以距離最近各個重心的資料集來分成數個群組。
# Step 5. 重複Step 1.~Step 4.，直到重心與群組不再改變為止。

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
df = pd.DataFrame({
    "feature A": [51, 46, 51, 45, 51, 50, 33,
              38, 37, 33, 33, 21, 23, 24],
    "feature B": [10.2, 8.8, 8.1, 7.7, 9.8, 7.2, 4.8,
              4.6, 3.5, 3.3, 4.3, 2.0, 1.0, 2.0]
    })

k = 3

kmeans = cluster.KMeans(n_clusters=k, random_state=12)
kmeans.fit(df)
print(kmeans.labels_)

colmap = np.array(["r", "g", "y"])
plt.scatter(df["feature A"], df["feature B"], color=colmap[kmeans.labels_])
plt.show()

# 使用K-means將鳶尾花分群
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import cluster

iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
X.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
y = iris.target

k = 3

kmeans = cluster.KMeans(n_clusters=k, random_state=12)
kmeans.fit(X) #K-means屬於非監督式學習，不需要答案的標籤(標準答案y)，只需要訓練資料集(X)即可。
print(kmeans.labels_)
print(y)
# 兩者排列順序不相同，顯示分群標籤錯誤。

# 與標準答案比對(比對兩個Lists)，計算準確度。
# ans=[]
# for i in kmeans.labels_:
#     for j in y:
#         if i==j:
#             ans.append(1)
#         elif i!=j:
#             ans.append(0)
# print(ans.count(1)/len(ans))

colmap = np.array(["r", "g", "y"])
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.subplots_adjust(hspace = 0.5)
plt.scatter(X["petal_length"], X["petal_width"], color=colmap[y])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Real Classification")

plt.subplot(1, 2, 2)
plt.scatter(X["petal_length"], X["petal_width"], color=colmap[kmeans.labels_])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-means Classification")
plt.show()

# 因為K-means演算法並沒有標籤，因此分類結果的標籤並不正確。
# 修正分群標籤錯誤，重新繪製散佈圖。
kmeans = cluster.KMeans(n_clusters=k, random_state=12)
kmeans.fit(X)
print("K-means Classification")
print(kmeans.labels_)
# 修正標籤錯誤
pred_y = np.choose(kmeans.labels_, [2, 0, 1]).astype(np.int64)
print("K-means Fix Classification")
print(pred_y)
print("Real Calssification")
print(y)

colmap = np.array(["r", "g", "y"])
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.subplots_adjust(hspace = 0.5)
plt.scatter(X["petal_length"], X["petal_width"], color=colmap[y])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("Real Classification")

plt.subplot(1, 2, 2)
plt.scatter(X["petal_length"], X["petal_width"], color=colmap[pred_y])
plt.xlabel("Petal Length")
plt.ylabel("Petal Width")
plt.title("K-means Classification")
plt.show()

# K-means模型績效衡量
import sklearn.metrics as sm

kmeans = cluster.KMeans(n_clusters=k, random_state=12)
kmeans.fit(X)
pred_y = np.choose(kmeans.labels_, [2, 0 ,1]).astype(np.int64)
# 績效矩陣
print(sm.accuracy_score(y, pred_y)) #accuracy_score(真實分類值, 模型分類值)
# 混淆矩陣
print(sm.confusion_matrix(y, pred_y))