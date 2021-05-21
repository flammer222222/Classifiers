import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
##########################################
import numpy as np
from sklearn import svm

svm_model = svm.SVC(kernel='rbf', gamma=0.1, C=2)
KNN_model = KNeighborsClassifier(n_neighbors=2)
tree_model = DecisionTreeClassifier()

attributes = []
csv_test_data_path = "list1.csv"
csv_predict_data_path = "list2.csv"

test_data_tables = pd.read_csv(csv_test_data_path, sep=';', encoding="utf_8")
result = np.array(test_data_tables['result'])
test_data_tables = test_data_tables.loc[:, test_data_tables.columns != 'result']
test_data_tables = test_data_tables.astype('float')
attributes = np.array(test_data_tables)


KNN_model.fit(attributes, result)
svm_model.fit(attributes, result)
tree_model.fit(attributes, result)


predict_data_tables = pd.read_csv(csv_predict_data_path, sep=';', encoding="utf_8")
result_predict_input = np.array(predict_data_tables['result'])
predict_data_tables = predict_data_tables.loc[:, predict_data_tables.columns != 'result']
predict_data_tables = predict_data_tables.astype('float')
attributes_predict = np.array(predict_data_tables)

result_predict1 = KNN_model.predict(attributes_predict)
print(result_predict1)
# print(result_predict_input)
print(accuracy_score(result_predict1, result_predict_input))
result_predict2 = svm_model.predict(attributes_predict)
print(result_predict2)
# print(result_predict_input)
print(accuracy_score(result_predict2, result_predict_input))

result_predict3 = tree_model.predict(attributes_predict)
print(result_predict3)
print(result_predict_input)
print(accuracy_score(result_predict3, result_predict_input))
res = []
for i in range(len(result_predict1)):
    if result_predict1[i] + result_predict2[i] + result_predict3[i] >= 2:
        res.append(1)
    else:
        res.append(0)
print(accuracy_score(res, result_predict_input))

import joblib
joblib.dump(svm_model, 'SVM_classifire.pkl')
joblib.dump(KNN_model, 'KNN_classifire.pkl')
joblib.dump(tree_model, 'tree_classifire.pkl')

clf = joblib.load('SVM_classifire.pkl')
result_predict3 = clf.predict(attributes_predict)
print(result_predict3)
print(accuracy_score(result_predict3, result_predict_input))
print(attributes_predict)