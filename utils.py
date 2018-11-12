# Copyright (C) 2018  Dan Tran

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np

def load_data():
    Xtrain = np.loadtxt('data/xtrain.csv', delimiter=',')
    ytrain = np.loadtxt('data/ytrain.csv', dtype=int)
    Xval = np.loadtxt('data/xval.csv', delimiter=',')
    yval = np.loadtxt('data/yval.csv', dtype=int)
    Xtest = np.loadtxt('data/xtest.csv', delimiter=',')
    ytest = np.loadtxt('data/ytest.csv', dtype=int)

    return Xtrain, ytrain, Xval, yval, Xtest, ytest

def split(X, y, val_div, test_div):
    val_num = int(X.shape[0] * val_div)
    test_num = int(X.shape[0] * test_div)

    idv = {-1}
    for i in range(val_num):
        r = -1
        while r in idv:
            r = np.random.randint(len(y))
        idv.add(r)
    idv.remove(-1)

    idt = {-1}
    for i in range(test_num):
        r = -1
        while r in idt or r in idv:
            r = np.random.randint(len(y))
        idt.add(r)
    idt.remove(-1)

    trainx = []
    trainy = []

    valx = []
    valy = []

    testx = []
    testy = []

    # Move points to correct set
    for i in range(len(y)):
        if i in idv:
            valx.append(X[i])
            valy.append(y[i])
        elif i in idt:
            testx.append(X[i])
            testy.append(y[i])
        else:
            trainx.append(X[i])
            trainy.append(y[i])

    np.savetxt('data/xtrain.csv', trainx, fmt='%d', delimiter=',')
    np.savetxt('data/ytrain.csv', trainy, fmt='%d', delimiter=',')
    np.savetxt('data/xval.csv', valx, fmt='%d', delimiter=',')
    np.savetxt('data/yval.csv', valy, fmt='%d', delimiter=',')
    np.savetxt('data/xtest.csv', testx, fmt='%d', delimiter=',')
    np.savetxt('data/ytest.csv', testy, fmt='%d', delimiter=',')
