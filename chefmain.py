from chefboost import Chefboost as chef
import pandas as pd
from shapes import *

df = pd.read_csv(r'C:\Users\minim\PycharmProjects\decisiontree\traindata.csv')
'''
config (dictionary):
			
			config = {
				'algorithm' (string): ID3, 'C4.5, CART, CHAID or Regression
				'enableParallelism' (boolean): False
				
				'enableGBM' (boolean): True,
				'epochs' (int): 7,
				'learning_rate' (int): 1,
				
				'enableRandomForest' (boolean): True,
				'num_of_trees' (int): 5,
				
				'enableAdaboost' (boolean): True,
				'num_of_weak_classifier' (int): 4
			}
'''

config = {'algorithm': 'C4.5', 'epochs': 7, 'num_of_trees': 3, 'enableAdaboost': False, 'enableRandomForest' : False,
          'enableGBM' : False}
model = chef.fit(df.copy(), config)

pred = pd.read_csv(r'C:\Users\minim\PycharmProjects\decisiontree\testdata.csv')

predcopy = pred.copy()
accuracyCopy = pred.copy()
for index, instance in pred.iterrows():
    prediction = chef.predict(model, instance)
    predcopy.iat[index, len(predcopy.columns)-1] = prediction
    actual = instance['Decision']
    if actual == prediction:
        #print(accuracyCopy.iloc[index]['Decision'])
        accuracyCopy.iat[index, len(accuracyCopy.columns)-1] = 'Correct'
    else:
        print(actual, ' - ', prediction)
        accuracyCopy.iat[index, len(accuracyCopy.columns)-1] = 'False'
        #print("*", end='')
    #print(actual, ' - ', prediction)



remappedTest = np.zeros((1000, 1000))
remappedTrain = np.zeros((1000, 1000))
remappedPrediction = np.zeros((1000, 1000))
remappedAcc = np.zeros((1000, 1000))

X_test = pred.values
X_train = df.values
X_pred = predcopy.values
Acc_Val = accuracyCopy.values
#print(X_test[0][2])
#print(Acc_Val[0][2])
#print(X_test[x][0],X_test[x][1] )
for x in range(len(X_test)):
    if X_test[x][len(X_test[0])-1] == 'pos':
        #print(X_test[x][0], X_test[x][1])
        remappedTest[int(X_test[x][0])][int(X_test[x][1])] = 2
    else:
        remappedTest[int(X_test[x][0])][int(X_test[x][1])] = 1
for x in range(len(X_train)):
    if X_train[x][len(X_train[0])-1] == 'pos':
        remappedTrain[int(X_train[x][0])][int(X_train[x][1])] = 2
    else:
        remappedTrain[int(X_train[x][0])][int(X_train[x][1])] = 1
for x in range(len(X_pred)):
    if X_pred[x][len(X_pred[0])-1] == 'pos':
        remappedPrediction[int(X_pred[x][0])][int(X_pred[x][1])] = 2
    else:
        remappedPrediction[int(X_pred[x][0])][int(X_pred[x][1])] = 1
for x in range(len(Acc_Val)):
    if Acc_Val[x][len(Acc_Val[0])-1] == 'Correct':
        remappedAcc[int(Acc_Val[x][0])][int(Acc_Val[x][1])] = 2
    else:
        remappedAcc[int(Acc_Val[x][0])][int(Acc_Val[x][1])] = 1


plt.matshow(remappedTrain, cmap='Greys')
plt.matshow(remappedTest, cmap='Greys')
plt.matshow(remappedPrediction, cmap='Greys')
plt.matshow(remappedAcc, cmap='Greys')
plt.show()