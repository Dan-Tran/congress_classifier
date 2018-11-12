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

import matplotlib.pyplot as plt
import metrics as mt
import numpy as np
import optimizers as op
import utils

def getlmax(X, y):
    avg = np.mean(y)
    lmax = -float('inf')
    for k in range(X.shape[1]):
        sum = 0.0
        for i in range(X.shape[0]):
            sum += X[i,k] * (y[i] - avg)
        lmax = max(lmax, 2.0 * abs(sum))
    return lmax

Xtrain, ytrain, Xval, yval, Xtest, Ytest = utils.load_data()

lmax = getlmax(Xtrain, ytrain)

nz = []
ls = []
trerr = []
verr = []

it = 0.0
verr_prev = float('inf')
wint = np.zeros(Xtrain.shape[1])
verr_best = float('inf')
lbest = None
wbest = None
bbest = None
same = 0

try:
    while True:
        l = lmax / float(1.25 ** it)
        wh, bh = op.coorddesc(Xtrain, ytrain, l)
        nonzeros = 0

        for i in range(wh.shape[0]):
            if wh[i] != 0:
                nonzeros += 1

        trainerr = mt.error(Xtrain, ytrain, wh, bh)
        valerr = mt.error(Xval, yval, wh, bh)

        nz.append(nonzeros)
        trerr.append(trainerr)
        verr.append(valerr)
        ls.append(l)

        if valerr < verr_best:
            verr_best = valerr
            lbest = l
            wbest = np.copy(wh)
            bbest = bh

        print('lambda: ', l, ' - verr: ', valerr)
        if valerr - verr_prev > 0.001:
            wint = wh
            break
        if valerr == verr_prev:
            same += 1
        else:
            same = 0
        if same > 100:
            break

        wint = wh
        verr_prev = valerr
        it += 1.0
finally:
    plt.plot(ls, nz)
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('# of Non Zeros')
    plt.title('# Non Zeros for Coordinate Descent')
    plt.gca().invert_xaxis()
    plt.show()

    plt.plot(ls, trerr, label='Training Error')
    plt.plot(ls, verr, label = 'Validation Error')
    plt.xscale('log')
    plt.legend()
    plt.xlabel('Lambda')
    plt.ylabel('Misclassification Error')
    plt.title('Error for Coordinate Descent')
    plt.gca().invert_xaxis()
    plt.show()

    print('Best Lambda: ', lbest)
    print('Train Misclassification Error: ', mt.error(Xtrain, ytrain, wbest, bbest))
    print('Validation Misclassification Error: ', mt.error(Xval, yval, wbest, bbest))
    #print('Test Square Error: ', sqerr(Xtest, ytest, wbest, bbest))

    print('bhat: ', bbest)
    np.savetxt('what', wbest)

    wtemp = np.array([abs(i) for i in wbest])
    ind = np.argpartition(wtemp, -10)[-10:]
    ind = ind[np.argsort(wtemp[ind])]
    print('10 Biggest Indices: ', ind)
