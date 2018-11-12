# Congress Classifier

This project explores how a representative’s voting patterns on
key issues can determine the ideology of said representative.
We will use 16 key votes as determined by the Congressional
Quarterly Almanac and learn how to associate the voting records
to classify representatives as either being a Democrat or a
Republican.

## Dependencies

The code is written in Python 3 using the `numpy` and `matplotlib`
libraries, with `seaborn` being used in `exp3.py` for better
color palettes for the graphs.

## Data

This project uses the congressional voting records from Volume
XL of the Congressional Quarterly Almanac that includes votes
for each member of the U.S. House of Representative on 16 key
issues. This data set was retrieved from the[UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records).
Each representative is represented by a vector with the first
element being either "democrat" or "republican" and the rest
being either "y," "n," or "?," corresponding to a yes vote, no
vote, or other respectively for the 16 votes. To process the
data, for each representative I made it's label "1" if they are
a democrat or "-1" if they are a republican.

For the vote features, there is an interesting minor complication
on how to represent the disposition. If the vote data only
consists of yes votes and no votes, it would be simple to just
do a binary encoding of the vote, with 1 being yes and 0 being
no. However, we also have the other vote, represented by a "?"
in the original data. To deal with this, I made the feature "1"
for "y," "-1" for "n," and "0" for "?." That way, intuitively we
have a no vote having an appropriate negative affect, a yes vote
having an appropriate positive affect, and an other vote having
a neutral affect, neither positive or negative.

For further use, we stored the classification labels as an CSV
file named "labels.csv" and the feature vectors as a CSV file
named "data.csv" which is found in the `data\` directory.

## Methods

This repository currently contains code for L2 logistic regression
with gradient descent, L2 logistic regression with stochastic
gradient descent (batch size 1), L2 logistic regression with
stochastic gradient descent (batch size 50), and LASSO least
squares regression with coordinate descent.

## Experiments

The experiment code are labeled `exp#.py`, where "#" refers to
the experiment number. This repository currently contains 3
such experiments. Experiment 1 looked at the average
misclassification errors for the 4 methods mentioned above.
Experiment 2 looked at hyperparameter settings for LASSO least
squares regression with coordinate descent. Experiment 3 looked
at the weights as the lambda value for LASSO least squares
regression with coordinate descent changed.


## License

The original raw data, specifically files `house-votes-84.data`
and `house-votes-84.names` in the `data\` directory are from
the UCI Machine Learning repository and thus are licensed on
their terms. A link to the dataset and the citation are given here:

```
https://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records

Dua, D. and Karra Taniskidou, E. (2017). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science. 
```

All other files are licensed under the GNU General Public License, version 2 or (at your option) any later version.

```
Copyright (C) 2018  Dan Tran

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
```
