from sklearn import tree, metrics
import pandas as pd
import graphviz
import matplotlib.pyplot as plt
import numpy as np

n_classes = 3
plot_colors = "rby"
plot_step = 0.02


df = pd.read_csv(r'C:\Users\minim\PycharmProjects\decisiontree\traindata.csv')
pred = pd.read_csv(r'C:\Users\minim\PycharmProjects\decisiontree\testdata.csv')


def class_value(classV):
    if classV == 'pos':
        return 1
    else:
        return 0

df['Decision'] = df['Decision'].apply(class_value)
pred['Decision'] = pred['Decision'].apply(class_value)
correct = pred['Decision']


#clf = tree.DecisionTreeClassifier(criterion='entropy')
X = df.drop(labels='Decision', axis=1)
#clf = clf.fit(X, df['Decision'])
verify = pred['Decision']
pred = pred.drop(labels='Decision', axis=1)
#print(pred.head)
#Z = clf.predict(pred)
#tree.plot_tree(clf)




'''
tree.plot_tree(clf, filled=True)
plt.show()
'''

    # We only take the two corresponding features

y = df["Decision"]

# Train
clf = tree.DecisionTreeClassifier(criterion='entropy').fit(X, y)

# Plot the decision boundary
#plt.subplot(2, 3, pairidx + 1)

Xarr = X.to_numpy()
#print(Xarr)
x_min, x_max = Xarr[:, 0].min() - 1, Xarr[:, 0].max() + 1
y_min, y_max = Xarr[:, 1].min() - 1, Xarr[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))
plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

predi = np.asarray(clf.predict(pred))
print(metrics.classification_report(verify, predi))

pred['Decision'] = predi
for index, instance in pred.iterrows():
    if pred.iat[index, len(pred.columns)-1] != verify[index]:
        pred.iat[index, len(pred.columns)-1] = 2
results = pred.to_numpy()
predres = pred['Decision'].to_numpy()
print(results)
#print(results.shape)
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')

# Plot the training points
for i, color in zip(range(n_classes), plot_colors):
    '''idx = np.where(y == i)
    plt.scatter(Xarr[idx, 0], Xarr[idx, 1], c=color,
                cmap=plt.cm.RdYlBu, edgecolor='black', s=15, label=i)
                '''
    idx2 = np.where(predres == i)
    plt.scatter(results[idx2, 0], results[idx2, 1], c=color,
                cmap=plt.cm.RdYlBu, edgecolor='black', s=15, label=i)
    #plt.scatter()
plt.suptitle("Decision surface of decision tree")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
#plt.figure()
'''
remappedTest = np.zeros((100, 100))
remappedTrain = np.zeros((100, 100))
remappedPrediction = np.zeros((100, 100))
remappedAcc = np.zeros((100, 100))

X_test = pred.values
X_train = df.values
X_pred = results

accuracy = np.zeros((len(results), 3))
for ind in range(len(results)):
    if results[ind][2] == verify[ind]:
        accuracy[ind] = [results[ind][0], results[ind][1], 1]
    else:
        accuracy[ind] = [results[ind][0], results[ind][1], 0]




for x in range(len(X_test)):
    if X_test[x][len(X_test[0])-1] == 1:
        #print(X_test[x][0], X_test[x][1])
        remappedTest[int(X_test[x][0])][int(X_test[x][1])] = 2
    else:
        remappedTest[int(X_test[x][0])][int(X_test[x][1])] = 1
for x in range(len(X_train)):
    if X_train[x][len(X_train[0])-1] == 1:
        remappedTrain[int(X_train[x][0])][int(X_train[x][1])] = 2
    else:
        remappedTrain[int(X_train[x][0])][int(X_train[x][1])] = 1
for x in range(len(X_pred)):
    if X_pred[x][len(X_pred[0])-1] == 1:
        remappedPrediction[int(X_pred[x][0])][int(X_pred[x][1])] = 2
    else:
        remappedPrediction[int(X_pred[x][0])][int(X_pred[x][1])] = 1
for x in range(len(accuracy)):
    if accuracy[x][len(accuracy[0])-1] == 1:
        remappedAcc[int(accuracy[x][0])][int(accuracy[x][1])] = 2
    else:
        remappedAcc[int(accuracy[x][0])][int(accuracy[x][1])] = 1


plt.matshow(remappedTrain, cmap='Greys')
plt.matshow(remappedTest, cmap='Greys')
plt.matshow(remappedPrediction, cmap='Greys')
plt.matshow(remappedAcc)
plt.show()
'''




#plt.figure()
#clf = tree.DecisionTreeClassifier().fit(X, y)
#tree.plot_tree(clf, filled=True)
plt.show()

