from chefboost import Chefboost as chef
import pandas as pd
from shapes import *

df = pd.read_csv(r'C:\Users\minim\PycharmProjects\decisiontree\traindata.csv')


config = {'algorithm': 'C4.5'}
model = chef.fit(df.copy(), config)

pred = pd.read_csv(r'C:\Users\minim\PycharmProjects\decisiontree\testdata.csv')

predcopy = pred.copy()
accuracyCopy = pred.copy()
for index, instance in pred.iterrows():
    prediction = chef.predict(model, instance)
    predcopy.iat[index, 2] = prediction
    actual = instance['Decision']

    if actual == prediction:
        #print(accuracyCopy.iloc[index]['Decision'])
        accuracyCopy.iat[index, 2] = 'Correct'
    else:
        accuracyCopy.iat[index, 2] = 'False'
        #print("*", end='')
    #print(actual, ' - ', prediction)



remappedTest = np.zeros((1000, 1000))
remappedTrain = np.zeros((1000, 1000))
remappedAcc = np.zeros((1000, 1000))
X_test = df.values
X_train = predcopy.values
Acc_Val = accuracyCopy.values
print(X_test[0][2])
print(Acc_Val[0][2])
for x in range(len(X_test)):
    if X_test[x][2] == 'pos':
        remappedTest[int(X_test[x][0])][int(X_test[x][1])] = 1
    else:
        remappedTest[int(X_test[x][0])][int(X_test[x][1])] = 2
for x in range(len(X_train)):
    if X_train[x][2] == 'pos':
        remappedTrain[int(X_train[x][0])][int(X_train[x][1])] = 1
    else:
        remappedTrain[int(X_train[x][0])][int(X_train[x][1])] = 2
for x in range(len(Acc_Val)):
    if Acc_Val[x][2] == 'Correct':
        remappedAcc[int(Acc_Val[x][0])][int(Acc_Val[x][1])] = 1
    else:
        remappedAcc[int(Acc_Val[x][0])][int(Acc_Val[x][1])] = 2


plt.matshow(remappedTrain)
plt.matshow(remappedTest)
plt.matshow(remappedAcc)
plt.show()