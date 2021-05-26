import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import KFold, cross_val_score as CVS, train_test_split as TTS, GridSearchCV
from xgboost import XGBClassifier as XGBC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost
from sklearn.metrics import accuracy_score

# 数据预处理模块
'''data_path = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"

dataframe = pd.read_csv(r"WA_Fn-UseC_-Telco-Customer-Churn.csv", encoding="UTF-8-sig")

# 数据总览
# print(dataframe.info())
# print(dataframe.head())

# 标签类编码
label = np.array(dataframe.iloc[:, -1])
le = LabelEncoder()
le = le.fit(label)
label_encode = le.transform(label)

# 找列名
# print(dataframe.columns.values.tolist())

# 调整列顺序，方便特征编码处理
dataframe = dataframe[['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                       'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaymentMethod', 'Churn']]

# TotalCharges列缺失值处理
l_monthly = dataframe["MonthlyCharges"].tolist()
l_total = dataframe["TotalCharges"].tolist()
for i in range(len(l_total)):
    if l_total[i] == ' ':
        l_total[i] = l_monthly[i]
l_new = [float(i) for i in l_total]
dataframe["TotalCharges"] = pd.DataFrame(l_new)

# 对Yes/No型特征进行0-1编码
dataframe["gender"] = OrdinalEncoder().fit_transform(np.array(dataframe["gender"]).reshape(-1, 1))
dataframe["SeniorCitizen"] = OrdinalEncoder().fit_transform(np.array(dataframe["SeniorCitizen"]).reshape(-1, 1))
dataframe["Partner"] = OrdinalEncoder().fit_transform(np.array(dataframe["Partner"]).reshape(-1, 1))
dataframe["Dependents"] = OrdinalEncoder().fit_transform(np.array(dataframe["Dependents"]).reshape(-1, 1))
dataframe["PhoneService"] = OrdinalEncoder().fit_transform(np.array(dataframe["PhoneService"]).reshape(-1, 1))
dataframe["PaperlessBilling"] = OrdinalEncoder().fit_transform(np.array(dataframe["PaperlessBilling"]).reshape(-1, 1))

# 对三分类特征进行独热编码
dataframe_temp = dataframe.iloc[:, 10:20]
enc = OneHotEncoder(categories="auto").fit(np.array(dataframe_temp))
result = enc.transform(np.array(dataframe_temp)).toarray()

# 找独热编码后变量的名字
# print(enc.get_feature_names())

# 拼接dataframe，得到最终表格
dataframe_drop = dataframe.drop(['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                                 'PaymentMethod', 'Churn'], axis=1)
new_dataframe = pd.concat([dataframe_drop, pd.DataFrame(result)], axis=1)
new_dataframe_churn = pd.concat([new_dataframe, pd.DataFrame(np.array(label_encode).reshape(-1, 1))], axis=1)
new_dataframe_churn.columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                               'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'x0_No',
                               'x0_No phone service', 'x0_Yes', 'x1_DSL', 'x1_Fiber optic', 'x1_No', 'x2_No',
                               'x2_No internet service', 'x2_Yes', 'x3_No', 'x3_No internet service', 'x3_Yes', 'x4_No',
                               'x4_No internet service', 'x4_Yes', 'x5_No', 'x5_No internet service', 'x5_Yes', 'x6_No',
                               'x6_No internet service', 'x6_Yes', 'x7_No', 'x7_No internet service', 'x7_Yes',
                               'x8_Month-to-month', 'x8_One year', 'x8_Two year', 'x9_Bank transfer (automatic)',
                               'x9_Credit card (automatic)', 'x9_Electronic check', 'x9_Mailed check', 'Churn']

# 流失用户数量：1869；一共7043个样本
# print(new_dataframe_churn["Churn"].sum())

# 输出编码后文件
pd.DataFrame.to_csv(new_dataframe_churn, path_or_buf="1.csv")'''

# 数据预处理2
'''data_path = r"WA_Fn-UseC_-Telco-Customer-Churn.csv"

dataframe = pd.read_csv(r"WA_Fn-UseC_-Telco-Customer-Churn.csv", encoding="UTF-8-sig")

# 数据总览
# print(dataframe.info())
# print(dataframe.head())

# 标签类编码
label = np.array(dataframe.iloc[:, -1])
le = LabelEncoder()
le = le.fit(label)
label_encode = le.transform(label)

# 找列名
# print(dataframe.columns.values.tolist())

# 调整列顺序，方便特征编码处理
dataframe = dataframe[['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService',
                       'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'MultipleLines', 'InternetService',
                       'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                       'StreamingMovies', 'Contract', 'PaymentMethod', 'Churn']]

# TotalCharges列缺失值处理
l_monthly = dataframe["MonthlyCharges"].tolist()
l_total = dataframe["TotalCharges"].tolist()
for i in range(len(l_total)):
    if l_total[i] == ' ':
        l_total[i] = l_monthly[i]
l_new = [float(i) for i in l_total]
dataframe["TotalCharges"] = pd.DataFrame(l_new)

# 对Yes/No型特征进行0-1编码
dataframe["gender"] = OrdinalEncoder().fit_transform(np.array(dataframe["gender"]).reshape(-1, 1))
dataframe["SeniorCitizen"] = OrdinalEncoder().fit_transform(np.array(dataframe["SeniorCitizen"]).reshape(-1, 1))
dataframe["Partner"] = OrdinalEncoder().fit_transform(np.array(dataframe["Partner"]).reshape(-1, 1))
dataframe["Dependents"] = OrdinalEncoder().fit_transform(np.array(dataframe["Dependents"]).reshape(-1, 1))
dataframe["PhoneService"] = OrdinalEncoder().fit_transform(np.array(dataframe["PhoneService"]).reshape(-1, 1))
dataframe["PaperlessBilling"] = OrdinalEncoder().fit_transform(np.array(dataframe["PaperlessBilling"]).reshape(-1, 1))
dataframe["MultipleLines"] = OrdinalEncoder().fit_transform(np.array(dataframe["MultipleLines"]).reshape(-1, 1))
dataframe["InternetService"] = OrdinalEncoder().fit_transform(np.array(dataframe["InternetService"]).reshape(-1, 1))
dataframe["OnlineSecurity"] = OrdinalEncoder().fit_transform(np.array(dataframe["OnlineSecurity"]).reshape(-1, 1))
dataframe["OnlineBackup"] = OrdinalEncoder().fit_transform(np.array(dataframe["OnlineBackup"]).reshape(-1, 1))
dataframe["DeviceProtection"] = OrdinalEncoder().fit_transform(np.array(dataframe["DeviceProtection"]).reshape(-1, 1))
dataframe["TechSupport"] = OrdinalEncoder().fit_transform(np.array(dataframe["TechSupport"]).reshape(-1, 1))
dataframe["StreamingTV"] = OrdinalEncoder().fit_transform(np.array(dataframe["StreamingTV"]).reshape(-1, 1))
dataframe["StreamingMovies"] = OrdinalEncoder().fit_transform(np.array(dataframe["StreamingMovies"]).reshape(-1, 1))
dataframe["Contract"] = OrdinalEncoder().fit_transform(np.array(dataframe["Contract"]).reshape(-1, 1))
dataframe["PaymentMethod"] = OrdinalEncoder().fit_transform(np.array(dataframe["PaymentMethod"]).reshape(-1, 1))

# 对三分类特征进行独热编码
# dataframe_temp = dataframe.iloc[:, 10:20]
# enc = OneHotEncoder(categories="auto").fit(np.array(dataframe_temp))
# result = enc.transform(np.array(dataframe_temp)).toarray()

# 找独热编码后变量的名字
# print(enc.get_feature_names())

# 拼接dataframe，得到最终表格
# dataframe_drop = dataframe.drop(['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
#                                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
#                                  'PaymentMethod', 'Churn'], axis=1)
# new_dataframe = pd.concat([dataframe_drop, pd.DataFrame(result)], axis=1)
# new_dataframe_churn = pd.concat([new_dataframe, pd.DataFrame(np.array(label_encode).reshape(-1, 1))], axis=1)
# new_dataframe_churn.columns = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
#                                'PhoneService', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges', 'x0_No',
#                                'x0_No phone service', 'x0_Yes', 'x1_DSL', 'x1_Fiber optic', 'x1_No', 'x2_No',
#                                'x2_No internet service', 'x2_Yes', 'x3_No', 'x3_No internet service', 'x3_Yes', 'x4_No',
#                                'x4_No internet service', 'x4_Yes', 'x5_No', 'x5_No internet service', 'x5_Yes', 'x6_No',
#                                'x6_No internet service', 'x6_Yes', 'x7_No', 'x7_No internet service', 'x7_Yes',
#                                'x8_Month-to-month', 'x8_One year', 'x8_Two year', 'x9_Bank transfer (automatic)',
#                                'x9_Credit card (automatic)', 'x9_Electronic check', 'x9_Mailed check', 'Churn']

# 流失用户数量：1869；一共7043个样本
# print(new_dataframe_churn["Churn"].sum())

# 输出编码后文件
# pd.DataFrame.to_csv(dataframe, path_or_buf="2.csv")'''

time_start = time.time()

# 读取预处理后文件
new_dataframe_churn = pd.read_csv(r"1.csv", encoding="UTF-8-sig")

# 取特征集
X = np.array(new_dataframe_churn.iloc[:, 2:-1])
Y = np.array(new_dataframe_churn.iloc[:, -1])

# 划分训练集测试集
X_train, X_test, Y_train, Y_test = TTS(X, Y, train_size=0.3, random_state=0)

# XGBoost预测
rec = XGBC( booster="gbtree"                # 弱评估器：梯度提升树
            , n_estimators=18               # default:100
            , max_depth=3                   # default:6
            , subsample=0.9                 # 每次建树时对样本的有放回抽样比例,default=1
            , learning_rate=0.3             # 学习率eta.default:1
            # , colsample_bytree=1          # 每次生成一棵树时对特征的随机抽样比例
            # , colsample_bylevel=1         # 每次生成一层树时对特征的随机抽样比例
            # , gamma=0                     # 叶节点分支阈值，后剪枝参数
            # , reg_alpha=0                 # L1正则化强度
            # , reg_lambda=1                # L2正则化强度
            # , max_delta_step=0            # 树的权重估计中允许的单次最大增量
            # , min_child_weight=1          # 叶节点最小权重
            ).fit(X_train, Y_train)

score = CVS(rec, X_train, Y_train, cv=10)

print("测试集上预测准确率为：{:.2f}%".format(rec.score(X_test, Y_test)*100))
print("训练集上十则交叉验证预测准确率为：{:.2f}%".format(score.mean()*100))

# XGBoost预测2
'''rec = XGBC( booster="gbtree"              # 弱评估器：梯度提升树
            , n_estimators=39
            , max_depth=1
            , subsample=0.2               # 每次建树时对样本的有放回抽样比例
            , learning_rate=0.2           # 学习率
            # , colsample_bytree=1        # 每次生成一棵树时对特征的随机抽样比例
            # , colsample_bylevel=1       # 每次生成一层树时对特征的随机抽样比例
            # , gamma=0                   # 叶节点分支阈值
            # , reg_alpha=0               # L1正则化强度
            # , reg_lambda=1              # L2正则化强度
            # , scale_pos_weight=3        # 类别不平衡处理
            # , max_delta_step=0          # 树的权重估计中允许的单次最大增量
            ).fit(X_train, Y_train)

score = CVS(rec, X_train, Y_train, cv=10)

print("测试集上预测准确率为：{:.2f}%".format(rec.score(X_test, Y_test)*100))
print("训练集上十则交叉验证预测准确率为：{:.2f}%".format(score.mean()*100))'''

# 决策树预测
'''clf = DecisionTreeClassifier(
                                criterion="gini"
                                , max_depth=3
                                , min_impurity_decrease=0
                                , min_samples_leaf=1
                                , random_state=0
                                ).fit(X_train, Y_train)

score = CVS(clf, X_train, Y_train, cv=10).mean()

print("测试集上预测准确率为：{:.2f}%".format(clf.score(X_test, Y_test)*100))
print("训练集上十则交叉验证预测准确率为：{:.2f}%".format(score.mean()*100))'''

# 随机森林预测
'''rfc = RandomForestClassifier(
                                criterion="gini"
                                , n_estimators=28
                                , max_depth=8
                                , min_samples_leaf=9
                                , random_state=0
                                ).fit(X_train, Y_train)

score = CVS(rfc, X_train, Y_train, cv=10).mean()

print("测试集上预测准确率为：{:.2f}%".format(rfc.score(X_test, Y_test)*100))
print("训练集上十则交叉验证预测准确率为：{:.2f}%".format(score.mean()*100))'''

# 超参数学习曲线
'''test = []
np_range = np.arange(1, 31, 1)
print(np_range)
for i in np_range:
    rec1 = XGBC(
                n_estimators=i
                )
    score = CVS(rec1, X_train, Y_train, cv=10)
    test.append(score.mean())
print("最优准确率为：{:.2f}%".format(max(test)*100))
print("此时建立树的数目为：{}".format(test.index(max(test))+1))
plt.plot(np_range, test, color="red", label="n_estimators")
plt.xticks(np_range)
plt.tick_params(axis='both', labelsize=7)
plt.legend()
plt.show()'''

# XGBoost网格搜索
'''parameters = {
                "n_estimators": np.arange(1, 31, 1)
                , "max_depth": np.arange(1, 10, 1)
                , "subsample": np.arange(0.1, 1.1, 0.1)
                , "learning_rate": np.arange(0.1, 1.1, 0.1)
                }

xgb = XGBC()
GS = GridSearchCV(xgb, parameters, cv=10)
GS = GS.fit(X_train, Y_train)
print("网格搜索下的最高精确度为：{:.2f}%".format(GS.best_score_*100))
print("网格搜索得到的最优参数取值为：{}".format(GS.best_params_))'''

# 决策树网格搜索
'''parameters = {
            "criterion": ("gini", "entropy")
            , "max_depth": np.arange(1, 11, 1)
            , "min_samples_leaf": np.arange(1, 21, 1)
            , "min_impurity_decrease": np.arange(0, 1, 0.1)
            }

clf1 = DecisionTreeClassifier(random_state=0)
GS = GridSearchCV(clf1, parameters, cv=10)
GS = GS.fit(X_train, Y_train)
print("网格搜索下的最高精确度为：{:.2f}%".format(GS.best_score_*100))
print("网格搜索得到的最优参数取值为：{}".format(GS.best_params_))'''

# 随机森林网格搜索
'''params = {
            "criterion": ("gini", "entropy")
            , "max_depth": np.arange(1, 11, 1)
            , "n_estimators": np.arange(1, 50, 1)
            , "min_samples_leaf": np.arange(1, 21, 1)
            }

rfc = RandomForestClassifier(random_state=0)
GS = GridSearchCV(rfc, params, cv=10)
GS.fit(X_train, Y_train)
print("网格搜索下的最高精确度为：{:.2f}%".format(GS.best_score_*100))
print("网格搜索得到的最优参数取值为：{}".format(GS.best_params_))'''

time_finish = time.time()
print("程序运行时间为：{:.4f}秒".format(time_finish-time_start))
