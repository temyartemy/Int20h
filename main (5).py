import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

customer_data1 = pd.read_csv('train.csv')
customer_data2 = pd.read_csv("test.csv")
columns1 = customer_data1.columns.values.tolist()
columns2 = customer_data2.columns.values.tolist()

dataset = customer_data1.drop(['Id', 'Week'], axis=1)
for i in columns1[2:51]:
    dataset[i].fillna(0, inplace=True)


for i in columns2[2:51]:
    customer_data2[i].fillna(0, inplace=True)
dataset1 = customer_data2.drop(['Id', 'Week'], axis=1)

data = customer_data2[['Id']]

feature_set = dataset.drop(['target'], axis=1)
result_set = dataset['target']

X_train, X_test, y_train, y_test = train_test_split(feature_set, result_set, test_size=0.2, random_state=42)
classifier = RandomForestClassifier(n_estimators=500, random_state=0)
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)
predictions1 = classifier.predict(dataset1)
data1 = pd.DataFrame(predictions1, columns=['Predicted'])

data2 = pd.concat([data, data1], axis=1, join='inner')
data3_predicted = []
data3_Id = []

i = 0
while i < data2.shape[0]:
    if data2.loc[i]['Predicted'].item() + data2.loc[i + 1]['Predicted'].item() + data2.loc[i + 2]['Predicted'].item() + data2.loc[i + 3]['Predicted'].item() > 0:
        data3_predicted.append(1)
        data3_Id.append(data2.at[i, 'Id'])
    else:
        data3_predicted.append(0)
        data3_Id.append(data2.at[i, 'Id'])
    i = i + 4

data3_Id_df = pd.DataFrame(data3_Id, columns = ['Id'])
data3_predicted_df =pd.DataFrame(data3_predicted, columns = ['Predicted'])
data3 = pd.concat([data3_Id_df, data3_predicted_df], axis=1, join='inner')


data3.to_csv('Submission.csv', index=False)