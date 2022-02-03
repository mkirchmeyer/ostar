"""
data are from http://csr.bu.edu/ftp/visda17/clf/
"""

import numpy as np
import pandas as pd

classe_vec = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
df = pd.read_csv('train_train.csv')  # doctest: +SKIP

aux = [str(i) for i in classe_vec]
filename = 'visda-train' + ''.join(aux)

ind = np.empty((0), dtype='int')
y = np.empty((0))
for i, classe in enumerate(classe_vec):
    aux = np.argwhere(df.values[:, 2048] == classe).squeeze()
    ind = np.concatenate((ind, aux), axis=0)
    y = np.concatenate((y, i * np.ones(len(aux))), axis=0)
X = df.values[ind, 0:2048]
np.savez(filename, X=X, y=y)
df = pd.read_csv('train_validation.csv')  # doctest: +SKIP​

# %%
aux = [str(i) for i in classe_vec]
filename = 'visda-val' + ''.join(aux)

# %%
ind = np.empty((0), dtype='int')
y = np.empty((0))
for i, classe in enumerate(classe_vec):
    aux = np.argwhere(df.values[:, 2048] == classe).squeeze()
    ind = np.concatenate((ind, aux), axis=0)
    y = np.concatenate((y, i * np.ones(len(aux))), axis=0)
X = df.values[ind, 0:2048]

np.savez(filename, X=X, y=y)