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

raw = np.loadtxt('data/house-votes-84.data', dtype=str, delimiter=',')

label = []
data = []

for entry in raw:
    if entry[0] == 'republican':
        label.append(-1)
    elif entry[0] == 'democrat':
        label.append(1)
    else:
        raise ValueError

    datum = []
    for xi in entry[1:]:
        if xi == 'n':
            datum.append(-1.0)
        elif xi == 'y':
            datum.append(1.0)
        elif xi == '?':
            datum.append(0.0)
        else:
            raise ValueError
    data.append(datum)

label = np.array(label, dtype=int)
data = np.array(data)

np.savetxt('data/data.csv', data, fmt='%d', delimiter=',')
np.savetxt('data/label.csv', label, fmt='%d', delimiter=',')
