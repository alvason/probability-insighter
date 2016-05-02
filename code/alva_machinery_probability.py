
# coding: utf-8

# # Probability-insighter
# https://github.com/alvason/probability-insighter
# 
# ### home-made machinery for insight into randomness

# In[ ]:

'''
author: Alvason Zhenhua Li
date:   04/16/2015

Home-made machinery for sorting a list from min-max
'''
import numpy as np


get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
###############
# time-watching and progress-bar
class TimeWatch(object):
    def __init__(cell):
        import time  
        cell.start_time = time.time()
    
    def progressBar(cell, starting , current_step, stopping):
        progressing = float(current_step - starting) / (stopping - starting) 
        from IPython.core.display import clear_output
        clear_output(wait = True) 
        import time 
        current_time = time.time()
        print('[{:6.6f} second {:} {:}% {:}]'.format(current_time - cell.start_time
                                              , int(10 * progressing) * '--'
                                              , int(100 * progressing)
                                              , int(10 - 10 * progressing) * '++'))
    def runTime(cell):
        import time 
        current_time = time.time()
        total_time = current_time - cell.start_time
        print('[running time = {:6.6f} second]'.format(total_time))
        return total_time
###############
import datetime
previous_running_time = datetime.datetime.now()
print ('Previous running time is {:}').format(previous_running_time)


# In[ ]:

### 
def productA(xx):
    # if xx is a scalar not array
    if isinstance(xx, (int, float)):
        xx = [xx]
    # for converting numpy-scalar (0-dimensional array()) to 0-dimensional array([]) 
    xx = np.atleast_1d(xx)
    xx = np.asarray(xx, dtype = int)
    total_point = len(xx)
    # set 0! = 1
    productX = np.zeros(total_point) + 1
    for j in range(total_point):
        for k in range(1, xx[j] + 1):        
            productX[j] = productX[j]*k
    return productX


# In[1]:

# min-max sorting
def AlvaMinMax(data):
    totalDataPoint = np.size(data)
    minMaxListing = np.zeros(totalDataPoint)   
    for i in range(totalDataPoint):
        # searching the minimum in current array
        jj = 0 
        minMaxListing[i] = data[jj] # suppose the 1st element [0] of current data-list is the minimum
        for j in range(totalDataPoint - i):
            if data[j] < minMaxListing[i]: 
                minMaxListing[i] = data[j]
                jj = j # recording the position of selected element
        # reducing the size of searching zone (removing the minmum from current array)
        data = np.delete(data, jj)
    return (minMaxListing)

'''
author: Alvason Zhenhua Li
date:   04/16/2015

Home-made machinery for leveling a list by using min-max way
'''
# leveling by using min-max way
def AlvaLevel(data, totalLevel, normalization = True):
    totalDataPoint = np.size(data)
    minMaxListing = AlvaMinMax(data)
    # searching minimum and maximum values
    minValue = minMaxListing[0]
    maxValue = minMaxListing[-1]
    spacingValue = np.linspace(minValue, maxValue, num = totalLevel + 1, retstep = True)        
    gLevel = np.delete(spacingValue[0], 0)
    # catogerizing the level set
    # initialize the levelspace by a 'null' space
    levelSpace = np.zeros([2])
    numberLevel = np.zeros([totalLevel])
    jj = 0 # counting the checked number
    for i in range(totalLevel): 
        n = 0 # counting the number in each level
        for j in range(jj, totalDataPoint):
            if minMaxListing[j] <= gLevel[i]: 
                levelSpace = np.vstack((levelSpace, [i, minMaxListing[j]]))
                n = n + 1
        numberLevel[i] = n
        jj = jj + n
    # delete the inital 'null' space
    levelSpace = np.delete(levelSpace, 0, 0) 
    if normalization == True:
        numberLevel = numberLevel/AlvaMinMax(numberLevel)[-1]
    return (gLevel, numberLevel, levelSpace)

