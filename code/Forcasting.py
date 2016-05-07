# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

'''
author: Alvason Zhenhua Li
date:   07/18/2015
'''

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import os
dir_path = '/Users/al/Desktop/GitHub/correlation-coefficient/figure'
file_name = 'linear-fitness'

import alva_machinery_statistics as alva

AlvaFontSize = 23
AlvaFigSize = (16, 7)
numberingFig = 0


aa = np.array([[0.9, 0.1], [0.5, 0.5]])
bb = np.identity(2)

for i in range(30):
    print (i, bb)
    bb = np.dot(aa, bb)
    
    

# <codecell>


