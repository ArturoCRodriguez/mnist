
#%%
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

mnist_train = pd.read_csv("mnist_train.csv")
mnist_test = pd.read_csv("mnist_test.csv")
X_train, y_train = mnist_train.loc[:,mnist_train.columns != "label"].values , mnist_train.loc[:,["label"]].values
X_test, y_test = mnist_test.loc[:,mnist_test.columns != "label"].values , mnist_test.loc[:,["label"]].values
# print(type(X_train))
print(X_test.shape)
print(y_test.shape)

some_digit = X_train[35999]
# some_digit_image = some_digit.reshape(28,28)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()
# print(y_train[35999])
#%%
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index] , y_train[shuffle_index]
#%%
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)
#%%
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train,y_train_5)
sgd_clf.predict([some_digit])

# %%
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf,X_train, y_train_5, cv=3, scoring="accuracy")
#%%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred = cross_val_predict(sgd_clf,X_train, y_train_5, cv=3)
confusion_matrix(y_train_5, y_train_pred)

# %%
from sklearn.metrics import precision_score, recall_score, f1_score
print(precision_score(y_train_5, y_train_pred))
print(recall_score(y_train_5, y_train_pred))
print(f1_score(y_train_5, y_train_pred))

# %%
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
from sklearn.metrics import precision_recall_curve
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# %%
