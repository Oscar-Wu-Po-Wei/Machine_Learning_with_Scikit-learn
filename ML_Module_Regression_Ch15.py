# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 10:07:43 2023

@author: oscar.wu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 註：Scikit-learn只能使用數值資料。

# [線性回歸：連續數值評估與預測]
# 簡單線性迴歸(Simple Linear Regression)：一因一果。
# y = ax+b
# y:(反)應變數(Response)
# x:解釋變數/自變數(Explanatory)
# a:迴歸係數
# b:截距
# 目的:利用X預測Y。

# 簡單回歸(範例1)

X_value = np.array([29, 28, 34, 31,
                    25, 29, 32, 31,
                    24, 33, 25, 31,
                    26, 30]) #建立NumPy陣列
Y_value = np.array([7.7, 6.2, 9.3, 8.4,
                    5.9, 6.4, 8.9, 7.5,
                    5.8, 9.1, 5.1, 7.3,
                    6.5, 8.4]) #建立NumPy陣列

X = pd.DataFrame(X_value, columns=['X_Value']) #X解釋變數DataFrame物件
Target = pd.DataFrame(Y_value, columns=['Y_Value'])
Y = Target['Y_Value'] #Y反應變數是DataFrame物件Target的Y_Value欄位

lm = LinearRegression() #建立LinearRegression物件
lm.fit(X, Y)            #使用fit()函數訓練模型
print('回歸分析:', lm.coef_)
print('截距:', lm.intercept_)
# 預測X_value為26、30的Y_value為多少？
New_X = pd.DataFrame(np.array([26, 30, 38])) #填入幾個，輸出對應個數預測值。
Predicted_Y = lm.predict(New_X)
print(Predicted_Y)

plt.scatter(X_value, Y_value)
Regression_Y = lm.predict(X)
plt.plot(X_value, Regression_Y, color='blue')
plt.plot(New_X, Predicted_Y, color='red', marker='o', markersize=10)
plt.show()

# 簡單回歸(範例2)

X_value = np.array([147.9, 163.5, 159.8,
                    155.1, 163.3, 158.7,
                    172.0, 161.2, 153.9, 161.6]) #建立NumPy陣列
Y_value = np.array([41.7, 60.2, 47.0,
                    53.2, 48.3, 55.2,
                    58.5, 49.0, 46.7, 52.5]) #建立NumPy陣列

X = pd.DataFrame(X_value, columns=['X_Value']) #X解釋變數DataFrame物件
Target = pd.DataFrame(Y_value, columns=['Y_Value'])
Y = Target['Y_Value'] #Y反應變數是DataFrame物件Target的Y_Value欄位

lm = LinearRegression() #建立LinearRegression物件
lm.fit(X, Y)            #使用fit()函數訓練模型
print('回歸分析:', lm.coef_)
print('截距:', lm.intercept_)
# 預測X_value為150, 160, 170的Y_value為多少？
New_X = pd.DataFrame(np.array([150, 160, 170])) #填入幾個，輸出對應個數預測值。
Predicted_Y = lm.predict(New_X)
print(Predicted_Y)

plt.scatter(X_value, Y_value)
Regression_Y = lm.predict(X)
plt.plot(X_value, Regression_Y, color='blue')
plt.plot(New_X, Predicted_Y, color='red', marker='o', markersize=10)
plt.show()

# 複回歸(範例1)
waist_heights = np.array([[67, 160], [68, 165], [70, 167],
                          [65, 170], [80, 165], [85, 167],
                          [78, 178], [79, 182], [95, 175],
                          [89, 172]])
weights = np.array([50, 60, 65, 65,
                    70, 75, 80, 85,
                    90, 81])
X = pd.DataFrame(waist_heights, columns=["Waist", "Height"])
target = pd.DataFrame(weights, columns=["Weight"])
y = target["Weight"]

lm = LinearRegression()
lm.fit(X, y)
print("迴歸係數:", lm.coef_)
print("截距:", lm.intercept_)

new_waist_heights = pd.DataFrame(np.array([[66,164], [82, 172]]))

predicted_weights = lm.predict(new_waist_heights)
print(predicted_weights)

# 複回歸(範例2)
Area_Distance = np.array([[10, 80], [8, 0], [8, 200], 
        [5, 200], [7, 300], [8, 230], 
        [7, 40], [9, 0], [6, 330], [9, 180]])
Revenue = np.array([46.9, 36.6, 37.1, 
            20.8, 24.6, 29.7, 
            36.6, 43.6, 19.8, 36.4])

X = pd.DataFrame(Area_Distance, columns=['Area', 'Distance'])
target = pd.DataFrame(Revenue, columns=['Revenue'])
y = target['Revenue']

lm = LinearRegression()
lm.fit(X, y)
print('迴歸係數', lm.coef_)
print('截距', lm.intercept_)

# [利用波士頓資料集預測房價]
from sklearn import datasets

boston = datasets.load_boston()
print(type(boston))
print(boston.keys()) #顯示鍵值清單
print(boston.data) #data房屋特徵資料(共13項)
print(boston.data.shape)
print(boston.target)
print(boston.feature_names)
print(boston.DESCR)
print(boston.filename)
print(boston.data_module)

X = pd.DataFrame(boston.data, columns=boston.feature_names) #data房屋特徵資料(共13項)，解釋變數X1,X2,...,X13。
print(X.head())
target = pd.DataFrame(boston.target, columns=["MEDV"]) #target房價資料
print(target.head())
y = target['MEDV']

# 1.訓練線性複回歸預測模型
lm = LinearRegression()
lm.fit(X, y)
print("迴歸係數:", lm.coef_) #13個解釋變數，共13個係數。
print("截距:", lm.intercept_)

coef = pd.DataFrame(boston.feature_names, columns=["features"])
coef["estimatedCoefficients"] = lm.coef_
print(coef) #發現RM之特徵係數最大，顯示RM與房價高度相關。(平均房間數量)

plt.scatter(X.RM, y)
plt.xlabel("Average number of rooms per dwelling (RM)")
plt.ylabel("Housing Price (MEDV)")
plt.title("Relationship between RM and Price")
plt.show()

predicted_price = lm.predict(X) #predict函數預測房價，參數為訓練資料。
print(predicted_price[0:5])

plt.scatter(y, predicted_price) #需要同樣單位才可繪製。
plt.xlabel("Price")
plt.ylabel("Predicted Price")
plt.title("Price vs. Predicted Price")
plt.show() #Price約為50時，有些錯誤預測，可使用殘差圖進行異常值(Outlier)偵測。

# 2.訓練和測試資料集
# 使用train_test_split()函數以指定比例隨機分割資料集，進行持久性驗證(Holdout Validation)。
# 持久性驗證：使用訓練資料集訓練模型，使用測試資料集驗證模型，兩者不會互相交叉利用。(比較：交叉驗證)
from sklearn.model_selection import train_test_split
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.33, random_state=5)
# test_size為測試資料集的分割比例，0.33表示測試資料集佔33%(三成)，訓練資料集佔67%(七成)，random_state指定亂數種子。

# 訓練模型
lm = LinearRegression()
lm.fit(XTrain, yTrain)

# 預測模型
pred_test = lm.predict(XTest)

plt.scatter(yTest, pred_test)
plt.xlabel("Price")
plt.ylabel("Predicted_Price")
plt.title("Price vs. Predicted_Price")
plt.show()

# 評估預測模型之績效
pred_train = lm.predict(XTrain)
pred_test = lm.predict(XTest)

MSE_train = np.mean((yTrain-pred_train)**2)
MSE_test = np.mean((yTest-pred_test)**2)

# 值愈小，模型愈好。
print("訓練資料的MSE: ", MSE_train)
print("測試資料的MSE: ", MSE_test)

# 值愈大，模型愈好。
print("訓練資料的R-squared: ", lm.score(XTrain, yTrain))
print("測試資料的R-squared: ", lm.score(XTest, yTest))

MSE = np.mean((y-predicted_price)**2)
print("MSE: ", MSE)
print("R-squared: ", lm.score(X, y))

# 殘差值 = 觀察值(Observed)-預測值(Predicted) → 找出異常值。
residue = y - predicted_price
plt.scatter(pred_train, pred_train-yTrain, c='b', s=40, alpha=0.5, label="Training Data")
plt.scatter(pred_test, pred_test-yTest, c='r', s=40, label="Test Data")
plt.hlines(y=0, xmin=0, xmax=50)
plt.title("Residual Plot")
plt.ylabel("Residual Value")
plt.legend()
plt.show()

# [Logistic Regression：分類問題]
# Logistic函數 = Sigmoid函數
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model

t = np.arange(-6, 6, 0.1)
S = 1/(1+np.e**(-t))

plt.plot(t, S)
plt.title("sigmoid function")
plt.show()

# [範例：鐵達尼號生存預測]
data = pd.read_csv('titanic.csv')
print(data.info())
age_median = np.nanmedian(data["Age"]) #nanmedian計算中位數。
print("年齡中位數", age_median)
new_age = np.where(data["Age"].isnull(), age_median, data["Age"])
# np.where(判斷式, 是的運算處理, 否的運算處理)
data["Age"] = new_age

# 將分類字串編碼成數值資料。
# 1.LabelEncoder物件中fit_transform()函數
label_encoder = preprocessing.LabelEncoder() #LabelEncoder物件
encoded_class = label_encoder.fit_transform(data["PClass"])

encoded_sex = label_encoder.fit_transform(data["Sex"])
data["SexCode"] = encoded_sex
# 2.np.where()
# data["SexCode"] = np.where(data["Sex"]=="male", 1, 0)
# 3.map()函數
# sex_mapping = {"male":1, "female":0}
# data["SexCode"] = data["Sex"].map(sex_mapping)
# 訓練資料集
X = pd.DataFrame([encoded_class, data["SexCode"], data["Age"]]).T
y = data["Survived"]

#訓練Logistic迴歸預測模型
logistic = linear_model.LogisticRegression()
logistic.fit(X, y)
print("迴歸係數:", logistic.coef_)
print("截距:", logistic.intercept_)

#Logistic迴歸預測模型準確度
logistic = linear_model.LogisticRegression()
logistic.fit(X, y)

preds = logistic.predict(X)
print(pd.crosstab(preds, data["Survived"]))
crosstab = pd.crosstab(preds, data["Survived"])
print((crosstab.iloc[0,0]+crosstab.iloc[1,1])/
      (crosstab.iloc[0,0]+crosstab.iloc[0,1]+crosstab.iloc[1,0]+crosstab.iloc[1,1]))
crosstab.to_excel("titanic_logistic_accuracy.xlsx")
print(logistic.score(X, y))