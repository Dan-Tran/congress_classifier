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
from sklearn import linear_model, neural_network
from sklearn.externals import joblib
from sklearn.metrics import mean_squared_error
import utils

Xtrain, ytrain, Xval, yval, Xtest, Ytest = utils.load_data()

model = neural_network.MLPClassifier(hidden_layer_sizes=(16, 10, 10, 2), max_iter=2000, alpha=0.5)
model.fit(Xtrain, ytrain)

predicty = model.predict(Xtrain)
acc = 0.0
total = 0.0
for i in range(len(predicty)):
    total += 1.0
    if predicty[i] == ytrain[i]:
        acc += 1.0

print('Training Error:', 1.0 - acc/total)

predicty = model.predict(Xval)
acc = 0.0
total = 0.0
for i in range(len(predicty)):
    total += 1.0
    if predicty[i] == yval[i]:
        acc += 1.0

print('Validation Error:', 1.0 - acc/total)
