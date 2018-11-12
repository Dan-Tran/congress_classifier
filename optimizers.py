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

def mu(xi, yi, w, b):
    return 1.0 / float(1.0 + np.exp(-yi * (b + np.dot(xi, w))))

def grad_desc(X, y, l, step):
    w_prev = np.ones(X.shape[1])
    w = np.zeros(X.shape[1])
    b = 0.0

    try:
        while True:
            gw = None
            gb = 0.0
            for i in range(X.shape[0]):
                gb += y[i] * mu(X[i], y[i], w, b) - y[i]
                if gw is None:
                    gw = np.array(y[i] * X[i] * (mu(X[i], y[i], w, b) - 1))
                else:
                    gw += np.array(y[i] * X[i] * (mu(X[i], y[i], w, b) - 1))
            gw = gw / float(X.shape[0]) + 2 * l * w
            gb = gb / float(X.shape[0])

            w = w - step * gw
            b = b - step * gb

            if np.allclose(w, w_prev, 0.001, 0.00001):
                break
            w_prev = np.copy(w)
    finally:
        return w, b

def sgd(X, y, l, step):
    w_prev = np.ones(X.shape[1])
    w = np.zeros(X.shape[1])
    b = 0.0

    try:
        while True:
            gw = None
            gb = 0.0

            i = np.random.randint(X.shape[0])

            gb += y[i] * mu(X[i], y[i], w, b) - y[i]
            if gw is None:
                gw = np.array(y[i] * X[i] * (mu(X[i], y[i], w, b) - 1))
            else:
                gw += np.array(y[i] * X[i] * (mu(X[i], y[i], w, b) - 1))
            gw = gw + 2 * l * w

            w = w - step * gw
            b = b - step * gb

            if np.allclose(w, w_prev, 0.001, 0.00001):
                break
            w_prev = np.copy(w)
    finally:
        return w, b

def batch_sgd(X, y, l, step, bat):
    w_prev = np.ones(X.shape[1])
    w = np.zeros(X.shape[1])
    b = 0.0

    try:
        bstart = 0
        while True:
            gw = None
            gb = 0.0
            for i in range(bat):
                bi = (bstart + i) % X.shape[0]
                gb += y[bi] * mu(X[bi], y[bi], w, b) - y[bi]
                if gw is None:
                    gw = np.array(y[bi] * X[bi] * (mu(X[bi], y[bi], w, b) - 1))
                else:
                    gw += np.array(y[bi] * X[bi] * (mu(X[bi], y[bi], w, b) - 1))
            gw = gw / float(bat) + 2 * l * w
            gb = gb / float(bat)

            w = w - step * gw
            b = b - step * gb

            bstart = (bstart + bat) % X.shape[0]

            if np.allclose(w, w_prev, 0.001, 0.00001):
                break
            w_prev = np.copy(w)
    finally:
        return w, b

def coorddesc(X, y, l):
    b = 0.0
    w_old = np.ones(X.shape[1])
    w = np.zeros(X.shape[1])

    a = 2.0 * np.sum(np.square(X), axis=0)

    while True:
        for j in range(X.shape[1]):
            b = np.mean(np.array(y) - np.matmul(X, w))

            c = 2.0 * np.dot(X[:,j], y - (b + np.matmul(X, w) - (w[j] * X[:,j])))

            if c < -l:
                w[j] = float(c + l) / float(a[j])
            elif c > l:
                w[j] = float(c - l) / float(a[j])
            else:
                w[j] = 0.0

        if np.allclose(w, w_old, 0.001, 0.00001):
            break
        w_old = np.copy(w)
    return w, b
