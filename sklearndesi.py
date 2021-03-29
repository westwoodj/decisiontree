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
clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2, min_samples_leaf=1, max_depth=None).fit(X, y)

# --------------------- DEFAULTS ===  min_samples_split=2, min_samples_leaf=1, max_depth=None -------------------------



# --------------------------------   PRUNING DATA   --------------------------------------


#   https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py

print("Tree depth - ", str(clf.get_depth()))
print("# of leaves - ", str(clf.get_n_leaves()))
path = clf.cost_complexity_pruning_path(X, y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X, y)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1] #remove last tree and ccp


node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()




train_scores = [clf.score(X, y) for clf in clfs]
test_scores = [clf.score(pred, verify) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()







# ------------------------ PLOT DECISION SPACE -------------------------
clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_split=2, min_samples_leaf=1, max_depth=None).fit(X, y)
plt.figure()
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
    
    idx2 = np.where(predres == i)
    plt.scatter(results[idx2, 0], results[idx2, 1], c=color,
                cmap=plt.cm.RdYlBu, edgecolor='black', s=15, label=i)
    #plt.scatter()
plt.suptitle("Decision surface of decision tree")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")
#plt.figure()






#plt.figure()
#clf = tree.DecisionTreeClassifier().fit(X, y)
#tree.plot_tree(clf, filled=True)
plt.show()

