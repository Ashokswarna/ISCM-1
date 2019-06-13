# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:05:59 2019

@author: ashok.swarna
"""
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

df = pd.read_excel('C:\\Users\\ashok.swarna\\Documents\\Book2.xlsx')
y = df['Embarked']

x = df
x=x.drop(['Embarked', 'Name', 'Ticket', 'Sex', 'Cabin', 'Age'], axis = 1, inplace = True)
x = x.rename(columns = {'Pclass': 'Pass_class'})

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
scaler = MinMaxScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(256,64),max_iter=100)

parameter_space = {
    'hidden_layer_sizes': [(256,64),(12,8)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}

from sklearn.model_selection import GridSearchCV

clf = GridSearchCV(mlp, parameter_space, n_jobs=-1, cv=3)
clf.fit(X_train, y_train)

# Best parameter set
print('Best parameters found:\n', clf.best_params_)

# All results
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
    
y_true, y_pred = y_test , clf.predict(X_test)

from sklearn.metrics import classification_report
print('Results on the test set:')
print(classification_report(y_true, y_pred))
