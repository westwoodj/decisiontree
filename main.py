#!/usr/bin/env python
import pdb
#https://github.com/barisesmer/C4.5

'''
Create your own synthetic data with two numeric attributes with values from the unit interval, [0,1].
Determine a certain geometric figure (square, circle, ring, or preferably something more complicated);
all examples inside this figure are labeled as positive, all examples outside this figure are labeled as negative.
Using a subset of these examples as a training set, induce a decision tree either C4.5 from Ross Quinlan’s website or
J48 from WEKA. Using a similar method as in Problem 2 (visualization of backprop behavior in 2-dimensional domains),
show the decision surface implemented by the induced decision tree—all examples labeled as positive are black and all 
examples labeled as negative are gray. Show how the shape of the positive region changes with different extents of pruning.
Experiment with 2 or 3 different geometric shapes. Perhaps also consider introducing some classlabel noise. 
Show how the behavior of decision trees can differ in different domains.
'''


from C4point5 import *
from shapes import *
from sklearn.model_selection import train_test_split
import csv
from itertools import chain
a = Mtx(1000)
a.setShape("square")
data = a.returnData()
#print(data[0][0])
y = []
for i in range(len(data)):
    for j in range(len(data[0])):
        y.append(data[i][j][2])
newdata = data.reshape((data.shape[0] * data.shape[1]))
#newdata = list(chain.from_iterable(data))
#print(y[0])
#print(len(newdata), len(y))
#data.reshape(1000, 1000 , 3)
#y = y.reshape(1000,1000,1)
#print(y.shape)
X_train, X_test, y_train, y_test = train_test_split(newdata, y, test_size=0.25, random_state=42)
#remappedTest = np.zeros((1000, 1000))
#remappedTrain = np.zeros((1000, 1000))
#print(X_train)
#print(X_train.shape)
#print(X_test[x][2])
'''
for x in range(len(X_test)):
    print(X_test[x][2])
    if X_test[x][2] == 'pos':
        remappedTest[int(X_test[x][0])][int(X_test[x][1])] = 1
    else:
        remappedTest[int(X_test[x][0])][int(X_test[x][1])] = 2
for x in range(len(X_train)):
    if X_train[x][2] == 'pos':
        remappedTrain[int(X_train[x][0])][int(X_train[x][1])] = 1
    else:
        remappedTrain[int(X_train[x][0])][int(X_train[x][1])] = 2
'''
with open("testdata.csv", "w", newline='' ) as fp:
    writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['X_coord', 'Y_coord', 'Decision'])
    for x in range(len(X_test)):
        writer.writerow(X_test[x])

with open("traindata.csv", "w", newline='' ) as fp:
    writer = csv.writer(fp, delimiter=',', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['X_coord', 'Y_coord', 'Decision'])

    for x in range(len(X_train)):
        writer.writerow(X_train[x])


'''

plt.matshow(remappedTrain)
plt.matshow(remappedTest)
plt.show()
'''
'''
a_test, a_train = train_test_split(a, test_size=0.25, random_state=42)
#a.showMat()
plt.matshow(a_test)
plt.matshow(a_train)
plt.show()
'''

