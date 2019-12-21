# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

data = pd.read_csv("mental health in tech survey.csv")


## Company Size and Work Interference as Predictors of Treatment

new5 = pd.DataFrame(columns = ['employee', 'interfere', 'treatment'])

leave_values = ['Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult', 'Very easy']
employee_values = ['6-25', 'More than 1000', '26-100', '100-500', '1-5', '500-1000']

for i in range(len(data.iloc[:, 9])):
    if data.iloc[i, 9] == '1-5':
        if data.iloc[i, 8] == 'Never':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(1,5), 'interfere': random.uniform(1.00,2.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(1,5), 'interfere': random.uniform(1.00,2.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Rarely':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(1,5), 'interfere': random.uniform(2.50,5.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(1,5), 'interfere': random.uniform(2.50,5.00), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Sometimes':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(1,5), 'interfere': random.uniform(5.00,7.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(1,5), 'interfere': random.uniform(5.00,7.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Often':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(1,5), 'interfere': random.uniform(7.50,10.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(1,5), 'interfere': random.uniform(7.50,10.00), 'treatment': 'Yes'}, ignore_index=True)
    if data.iloc[i, 9] == '6-25':
        if data.iloc[i, 8] == 'Never':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(6,25), 'interfere': random.uniform(1.00,2.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(6,25), 'interfere': random.uniform(1.00,2.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Rarely':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(6,25), 'interfere': random.uniform(2.50,5.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(6,25), 'interfere': random.uniform(2.50,5.00), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Sometimes':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(6,25), 'interfere': random.uniform(5.00,7.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(6,25), 'interfere': random.uniform(5.00,7.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Often':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(6,25), 'interfere': random.uniform(7.50,10.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(6,25), 'interfere': random.uniform(7.50,10.00), 'treatment': 'Yes'}, ignore_index=True)
    if data.iloc[i, 9] == '26-100':
        if data.iloc[i, 8] == 'Never':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(26,100), 'interfere': random.uniform(1.00,2.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(26,100), 'interfere': random.uniform(1.00,2.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Rarely':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(26,100), 'interfere': random.uniform(2.50,5.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(26,100), 'interfere': random.uniform(2.50,5.00), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Sometimes':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(26,100), 'interfere': random.uniform(5.00,7.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(26,100), 'interfere': random.uniform(5.00,7.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Often':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(26,100), 'interfere': random.uniform(7.50,10.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(26,100), 'interfere': random.uniform(7.50,10.00), 'treatment': 'Yes'}, ignore_index=True)
    if data.iloc[i, 9] == '100-500':
        if data.iloc[i, 8] == 'Never':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(100,500), 'interfere': random.uniform(1.00,2.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(100,500), 'interfere': random.uniform(1.00,2.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Rarely':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(100,500), 'interfere': random.uniform(2.50,5.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(100,500), 'interfere': random.uniform(2.50,5.00), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Sometimes':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(100,500), 'interfere': random.uniform(5.00,7.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(100,500), 'interfere': random.uniform(5.00,7.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Often':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(100,500), 'interfere': random.uniform(7.50,10.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(100,500), 'interfere': random.uniform(7.50,10.00), 'treatment': 'Yes'}, ignore_index=True)
    if data.iloc[i, 9] == '500-1000':
        if data.iloc[i, 8] == 'Never':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(500,1000), 'interfere': random.uniform(1.00,2.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(500,1000), 'interfere': random.uniform(1.00,2.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Rarely':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(500,1000), 'interfere': random.uniform(2.50,5.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(500,1000), 'interfere': random.uniform(2.50,5.00), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Sometimes':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(500,1000), 'interfere': random.uniform(5.00,7.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(500,1000), 'interfere': random.uniform(5.00,7.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Often':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(500,1000), 'interfere': random.uniform(7.50,10.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(500,1000), 'interfere': random.uniform(7.50,10.00), 'treatment': 'Yes'}, ignore_index=True)
    if data.iloc[i, 9] == 'More than 1000':
        if data.iloc[i, 8] == 'Never':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(1000,10000), 'interfere': random.uniform(1.00,2.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(1000,10000), 'interfere': random.uniform(1.00,2.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Rarely':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(1000,10000), 'interfere': random.uniform(2.50,5.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(1000,10000), 'interfere': random.uniform(2.50,5.00), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Sometimes':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(1000,10000), 'interfere': random.uniform(5.00,7.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(1000,10000), 'interfere': random.uniform(5.00,7.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == 'Often':
            if data.iloc[i, 7] == 'No':
                new5 = new5.append({'employee': random.uniform(1000,10000), 'interfere': random.uniform(7.50,10.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new5 = new5.append({'employee': random.uniform(1000,10000), 'interfere': random.uniform(7.50,10.00), 'treatment': 'Yes'}, ignore_index=True)



X = new5.iloc[:, [0,1]].values
y = new5.iloc[:, 2].values


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

scaler=StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)



knn = KNeighborsClassifier(n_neighbors=25)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("The accuracy of company size and work interference:",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



error=[]
for i in range(1,50):
    kk = KNeighborsClassifier(n_neighbors=i)
    kk.fit(X_train, y_train)
    pred_i = kk.predict(X_test)
    error.append(np.mean(pred_i != y_test))


plt.figure(figsize=(12, 6))
plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')




scatter_x = new5['employee'].values
scatter_y = new5['interfere'].values
group = new5['treatment'].values

cdict = {'No': 'red', 'Yes': 'cyan'}


fig, ax = plt.subplots()
#ax.set_yticklabels([])
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], marker='x', label = g, s = 100)
ax.legend()
plt.title('Actual')
plt.xlabel('No of Employees')
plt.ylabel('Level of Work Interference')
plt.show()


X_test = scaler.inverse_transform(X_test)
X_train = scaler.inverse_transform(X_train)

xblah=[]
for i in range(len(X_test)):
    xblah.append(X_test[i][0])
xblah = np.asarray(xblah)

yblah=[]
for i in range(len(X_test)):
    yblah.append(X_test[i][1])
yblah = np.asarray(yblah)


scatter_x = xblah
scatter_y = yblah
group = y_pred

fig, ax = plt.subplots()

for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], marker='x', label = g, s = 100)
ax.legend()
plt.title('Prediction on Test Set')
plt.xlabel('No of Employees')
plt.ylabel('Level of Work Interference')
plt.show()






## Age and Interference as Predictors of Treatment

new1 = pd.DataFrame(columns = ['Age', 'interfere', 'treatment'])

for i in range(len(data.iloc[:, 1])):
    if data.iloc[i, 1] >= 18 and data.iloc[i, 1] <= 80:
        tt.append(data.iloc[i,1])
        if data.iloc[i, 8] == "Never":
            if data.iloc[i, 7] == 'No':
                new1 = new1.append({'Age': data.iloc[i,1], 'interfere': random.uniform(1.00,2.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new1 = new1.append({'Age': data.iloc[i,1], 'interfere': random.uniform(1.00,2.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == "Rarely":
            if data.iloc[i, 7] == 'No':
                new1 = new1.append({'Age': data.iloc[i,1], 'interfere': random.uniform(2.50,5.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new1 = new1.append({'Age': data.iloc[i,1], 'interfere': random.uniform(2.50,5.00), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == "Sometimes":
            if data.iloc[i, 7] == 'No':
                new1 = new1.append({'Age': data.iloc[i,1], 'interfere': random.uniform(5.00,7.50), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new1 = new1.append({'Age': data.iloc[i,1], 'interfere': random.uniform(5.00,7.50), 'treatment': 'Yes'}, ignore_index=True)
        if data.iloc[i, 8] == "Often":
            if data.iloc[i, 7] == 'No':
                new1 = new1.append({'Age': data.iloc[i,1], 'interfere': random.uniform(7.50,10.00), 'treatment': 'No'}, ignore_index=True)
            elif data.iloc[i, 7] == 'Yes':
                new1 = new1.append({'Age': data.iloc[i,1], 'interfere': random.uniform(7.50,10.00), 'treatment': 'Yes'}, ignore_index=True)

X = new1.iloc[:, [0,1]].values
y = new1.iloc[:, 2].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

scaler=StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

print("The accuracy of age and work interference::",metrics.accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



scatter_x = new1['Age'].values
scatter_y = new1['interfere'].values
group = new1['treatment'].values

cdict = {'No': 'red', 'Yes': 'cyan'}


fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], marker='x', label = g, s = 100)
ax.legend()
plt.title('Actual')
plt.xlabel('Age')
plt.ylabel('Level of Work Interference')
plt.show()

X_test = scaler.inverse_transform(X_test)
X_train = scaler.inverse_transform(X_train)

xblah=[]
for i in range(len(X_test)):
    xblah.append(X_test[i][0])
xblah = np.asarray(xblah)

yblah=[]
for i in range(len(X_test)):
    yblah.append(X_test[i][1])
yblah = np.asarray(yblah)


scatter_x = xblah
scatter_y = yblah
group = y_pred

fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(scatter_x[ix], scatter_y[ix], c = cdict[g], marker='x', label = g, s = 100)
ax.legend()
plt.title('Prediction on Test Set')
plt.xlabel('Age')
plt.ylabel('Level of Work Interference')
plt.show()
######################################
