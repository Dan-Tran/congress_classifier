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

def classify(k, X):
    classes = np.empty(X.shape[0], dtype=int)
    for i in range(X.shape[0]):
        distmin = None
        cmin = None
        for c in range(k.shape[0]):
            dist = np.linalg.norm(X[i] - k[c])
            if distmin == None or dist < distmin:
                distmin = dist
                cmin = c
        classes[i] = cmin
    return classes

def get_means(k, X, classes):
    means = np.zeros([k.shape[0], X.shape[1]])
    counts = np.zeros(k.shape[0])
    for i in range(X.shape[0]):
        means[classes[i]] += X[i]
        counts[classes[i]] += 1.0
    for c in range(k.shape[0]):
        if counts[c] == 0:
            means[c] = k[c]
        else:
            means[c] /= counts[c]
    return means

def kmeans(kinit, X):
    k = kinit
    kold = np.copy(k)
    classes = None
    i = 0
    while True:
        classes = classify(k, X)
        k = get_means(k, X, classes)
        if np.allclose(k, kold, 0.0001, 0.000001):
            break
        kold = np.copy(k)
        i += 1
    return k, classes 

def get_kinit(ks, X):
    k = np.zeros((ks, X.shape[1]))
    seen = {-1}
    for i in range(ks):
        ki = np.random.randint(X.shape[0])
        while ki in seen:
            ki = np.random.randint(X.shape[0])
        seen.add(ki)
        k[i] = X[ki]
    return k

def kmeanspp(ks, X):
    kinit = np.zeros((ks, X.shape[1]))
    p = np.zeros(X.shape[0])

    ki = np.random.randint(X.shape[0])
    seen = {ki}
    kinit[0] = X[ki]

    for c in range(1, ks):
        sumd = 0.0
        for i in range(X.shape[0]):
            mind = None
            for k in kinit:
                d = np.linalg.norm(X[i] - k, 2) ** 2
                if mind is None or d < mind:
                    mind = d
            p[i] = mind
            sumd += mind
        p /= sumd

        newki = np.random.choice(X.shape[0], p=p)
        while newki in seen:
            newki = np.random.choice(X.shape[0], p=p)
            seen.add(newki)
        kinit[c] = X[newki]

    return kinit