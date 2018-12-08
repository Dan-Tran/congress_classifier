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

import metrics as mt
import numpy as np
import kmeans as km
import utils

Xtrain, ytrain, Xval, yval, Xtest, Ytest = utils.load_data()

X = Xtrain
y = ytrain

ks = 10

kinit = km.kmeanspp(ks, X)
k, classes = km.kmeans(kinit, X)

classlist = [[] for i in range(ks)]
for i in range(X.shape[0]):
    classlist[int(classes[i])].append(y[i])

print('Training')
for i in range(len(classlist)):
    rs = 0
    ds = 0
    for y in classlist[i]:
        if y > 0:
            ds += 1
        else:
            rs += 1
    print('Class', i, ':', rs, 'Republicans,', ds, 'Democrats')

valclasses = km.classify(k, Xval)

valclasslist = [[] for i in range(ks)]
for i in range(Xval.shape[0]):
    valclasslist[valclasses[i]].append(yval[i])

print('Validation')
for i in range(len(valclasslist)):
    rs = 0
    ds = 0
    for y in valclasslist[i]:
        if y > 0:
            ds += 1
        else:
            rs += 1
    print('Class', i, ':', rs, 'Republicans,', ds, 'Democrats')
