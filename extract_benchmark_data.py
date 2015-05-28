# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:37:53 2015

@author: yinchengng
"""

import scipy.io
import numpy as np
import pickle
from copula_experiments import *

data = scipy.io.loadmat('../data/benchmarks.mat')
for key_array in data['benchmarks'][0]:
    key = str(key_array[0])
    print key
    x = data[key]['x'][0, 0]
    y = np.transpose(data[key]['t'][0,0])[0]
    with open('./data/'+key+'.pickle', 'w') as f:
        pickle.dump([x, y], f)

with open('../data/simulated.pickle', 'w') as f:
    pickle.dump(confabulate_logit_data(d=20, N=520))