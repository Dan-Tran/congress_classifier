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

from joblib import dump, load
import numpy as np
from sklearn import ensemble
import utils

model = load('exp6.model')

Xtrain, ytrain, Xval, yval, Xtest, ytest = utils.load_data()

predicty = model.predict(Xtest)
acc = 0.0
total = 0.0
for i in range(len(predicty)):
    total += 1.0
    if predicty[i] == ytest[i]:
        acc += 1.0

print('Test Error:', 1.0 - acc/total)