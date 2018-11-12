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
import seaborn as sns

sns.set_palette(sns.color_palette("hls", 16))

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

ws = []
ls = []

try:
    for it in range(20):
        l = lmax / float(1.25 ** it)
        wh, bh = op.coorddesc(Xtrain, ytrain, l)

        ws.append(wh)
        ls.append(l)

finally:
    fig = plt.figure()
    ax = plt.subplot(111)

    ws = np.array(ws)
    for i in range(ws.shape[1]):
        ax.plot(ls, ws[:,i], label='w_%d' % i)

    plt.xlabel('Lambda')
    plt.ylabel('Weight Value')
    plt.title('Weight Values vs Lambda')
    plt.xscale('log')
    plt.gca().invert_xaxis()

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

