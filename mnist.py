
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

# some_digit = X_train.iloc[36000]
# some_digit_image = some_digit.values.reshape(28,28)
# plt.imshow(some_digit_image, cmap = matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()
# print(y_train.iloc[36000])
#%%
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index] , y_train[shuffle_index]
