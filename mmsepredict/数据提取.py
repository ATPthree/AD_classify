import os
import pandas as pd
import numpy as np

path = 'D:/AD_DETECT/mmsepredict/metadata_allvisits_T.xlsx'
# 使用read_excel来读取.xlsx文件
data = pd.read_excel(path)
# 移除第4列（索引为3）值为空的行
data = data.dropna(subset=[data.columns[3]])
# print(data.head())
y=data.iloc[:,3:4].values
X=data.iloc[:,11:12].values
# print(X.head(),y.head())
index = np.random.permutation(len(X))
train_size=int(len(X)*0.8)
train_index=index[:train_size]
test_index= index[train_size:]

X_train,X_test=X[train_index],X[test_index]
y_train,y_test=y[train_index],y[test_index]

# 创建训练集DataFrame并保存为CSV
train_df = pd.DataFrame({
    'X': X_train.flatten(),
    'y': y_train.flatten()
})
# 强制覆盖已有文件
if os.path.exists('data1.csv'):
    os.remove('data1.csv')
train_df.to_csv('data1.csv', index=False)

# 创建测试集DataFrame并保存为CSV
test_df = pd.DataFrame({
    'X': X_test.flatten(),
    'y': y_test.flatten()
})
# 强制覆盖已有文件
if os.path.exists('data2.csv'):
    os.remove('data2.csv')
test_df.to_csv('data2.csv', index=False)

print("训练集和测试集已保存为CSV文件")






