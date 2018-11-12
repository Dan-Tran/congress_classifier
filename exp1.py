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
import optimizers as op
import utils

trials = 1000

gradsum = 0.0
sgdsum = 0.0
batchsum = 0.0
coordsum = 0.0

for i in range(trials):
    Xtrain, ytrain, Xval, yval, Xtest, Ytest = utils.load_data()

    wgrad, bgrad = op.grad_desc(Xtrain, ytrain, 0.01, 0.000001)
    wsgd, bsgd = op.sgd(Xtrain, ytrain, 0.01, 0.000001)
    wbsgd, bbsgd = op.batch_sgd(Xtrain, ytrain, 0.01, 0.000001, 50)
    wcord, bcord = op.coorddesc(Xtrain, ytrain, 0.01)

    gradsum += mt.error(Xval, yval, wgrad, bgrad)
    sgdsum += mt.error(Xval, yval, wsgd, bsgd)
    batchsum += mt.error(Xval, yval, wbsgd, bbsgd)
    coordsum += mt.error(Xval, yval, wcord, bcord)

print('Average Grad Misclassification: ', gradsum / float(trials))
print('Average SGD Misclassification: ', sgdsum / float(trials))
print('Average Batch SGD Misclassification: ', batchsum / float(trials))
print('Average LASSO Misclassification: ', coordsum / float(trials))
