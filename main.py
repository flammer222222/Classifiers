import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import svm

# настройка моделей классификаторов
svm_model = svm.SVC(kernel='rbf', gamma=0.1, C=2)
KNN_model = KNeighborsClassifier(n_neighbors=2)
tree_model = DecisionTreeClassifier()

# пути к данным обучения и тестирования
csv_test_data_path = "doc/list1.csv"
csv_predict_data_path = "doc/list2.csv"

test_data_tables = pd.read_csv(csv_test_data_path, sep=';', encoding="utf_8")
result = np.array(test_data_tables['result'])
test_data_tables = test_data_tables.loc[:, test_data_tables.columns != 'result']
test_data_tables = test_data_tables.astype('float')
attributes = np.array(test_data_tables)

# расчет весов
KNN_model.fit(attributes, result)
svm_model.fit(attributes, result)
tree_model.fit(attributes, result)

# подгтовка тестовых данных
predict_data_tables = pd.read_csv(csv_predict_data_path, sep=';', encoding="utf_8")
result_predict_input = np.array(predict_data_tables['result'])
predict_data_tables = predict_data_tables.loc[:, predict_data_tables.columns != 'result']
predict_data_tables = predict_data_tables.astype('float')
attributes_predict = np.array(predict_data_tables)

# проверка на тестовых данных и расчет точности для каждого классификатора
result_predict_KNN = KNN_model.predict(attributes_predict)
print(result_predict_KNN)
print(accuracy_score(result_predict_KNN, result_predict_input))

result_predict_svm = svm_model.predict(attributes_predict)
print(result_predict_svm)
print(accuracy_score(result_predict_svm, result_predict_input))

result_predict_tree = tree_model.predict(attributes_predict)
print(result_predict_tree)
print(accuracy_score(result_predict_tree, result_predict_input))

# расчет среднего арефметического
res = []
for i in range(len(result_predict_KNN)):
    if result_predict_KNN[i] + result_predict_svm[i] + result_predict_tree[i] >= 2:
        res.append(1)
    else:
        res.append(0)
print(accuracy_score(res, result_predict_input))

# сохранение моделей клаччификаторов для дальнейшего пользования
joblib.dump(svm_model, 'SVM_classifire.pkl')
joblib.dump(KNN_model, 'KNN_classifire.pkl')
joblib.dump(tree_model, 'tree_classifire.pkl')


