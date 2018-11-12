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

def J(X, y, w, b, l):
    j = 0.0
    for i in range(X.shape[0]):
        j += np.log(1.0 + np.exp(-y[i] * (b + np.dot(X[i], w))))
    return j / float(X.shape[0]) + l * np.linalg.norm(w, 2)

def error(X, y, w, b):
    err = 0.0
    for i in range(X.shape[0]):
        if np.sign(b + np.dot(X[i], w)) != y[i]:
            err += 1.0
    return err / float(X.shape[0])
