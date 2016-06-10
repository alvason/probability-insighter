
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

'''
author: Alvason Zhenhua Li
date:   07/16/2015

Home-made machinery for correlation-coefficient 
'''
# coefficient of determination --- r2 (for a linear fitness)
def AlvaLinearFit(x, y):
    meanRaw = y.sum(axis = 0)/y.size
    variance_raw = np.sum((y - meanRaw)**2)  
    # linear fitting
    linearFit = np.polyfit(x, y, 1)
    slopeFit = linearFit[0]
    constantFit = linearFit[1]
    yFit = slopeFit*x + linearFit[1]
    variance_fit = np.sum((y - yFit)**2)
    r2 = 1 - variance_fit/variance_raw
    return (slopeFit, constantFit, r2)


'''
author: Alvason Zhenhua Li
date:   04/16/2015

Home-made machinery for sorting a list from min-max

Home-made machinery for leveling a list by using min-max way
'''
# min-max sorting
def minMaxA(data):
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

# leveling by using numpy way
def AlvaPDF(dataX, minX = None, maxX = None, total_level = None, normalization = True, empty_leveler_filter = True):
    dataX = np.asarray(dataX)
    if minX is None:
        minX = min(dataX)       
    if maxX is None:
        maxX = max(dataX)
    if total_level is None:
        leveler = np.linspace(minX, maxX, num = 10 + 1)[1:]
        total_level = len(leveler)
    else:
        leveler = np.linspace(minX, maxX, num = total_level + 1)[1:]
    leveleee = np.zeros([total_level])
    for i in range(total_level):
        total_under = np.sum([dataX[:] <= leveler[i]]) 
        leveleee[i] = total_under - np.sum(leveleee[0:i])
    if normalization:
        leveleee = leveleee / np.sum(leveleee)
    # associating (leveler_n, levelee_n)...it is important for the next filter-zero step
    PDF_distribution = np.array([leveler, leveleee]).T   
    # filter out empty-leveler (nothing inside the leveler)...it is important for the future log-step
    if empty_leveler_filter:
        PDF_distribution_update = PDF_distribution[PDF_distribution[:, 1] > 0.0].T
    else: PDF_distribution_update = PDF_distribution.T
    return (PDF_distribution_update)
# # leveling by using min-max way
# def AlvaPDF(dataX, total_level, normalization = True, empty_leveler_filter = True):
#     totalDataPoint = np.size(data)
#     minMaxListing = minMaxA(data)
#     # searching minimum and maximum values
#     minValue = minMaxListing[0]
#     maxValue = minMaxListing[-1]
#     spacingValue = np.linspace(minValue, maxValue, num = totalLevel + 1, retstep = True)        
#     leveler = np.delete(spacingValue[0], 0)
#     # catogerizing the level set
#     # initialize the levelspace by a 'null' space
#     levelSpace = np.zeros([2])
#     levelee = np.zeros([totalLevel])
#     jj = 0 # counting the checked number
#     for i in range(totalLevel): 
#         n = 0 # counting the number in each level
#         for j in range(jj, totalDataPoint):
#             if minMaxListing[j] <= gLevel[i]: 
#                 levelSpace = np.vstack((levelSpace, [i, minMaxListing[j]]))
#                 n = n + 1
#         levelee[i] = n
#         jj = jj + n
#     # delete the inital 'null' space
#     levelSpace = np.delete(levelSpace, 0, 0) 
#     if normalization == True:
#         levelee = levelee / np.sum(minMaxA(levelee))
#     # associating (leveler_n, levelee_n)...it is important for the next filter-zero step
#     PDF_distribution = np.array([leveler, leveleee]).T   
#     # filter out empty-leveler (nothing inside the leveler)...it is important for the future log-step
#     if empty_leveler_filter:
#         PDF_distribution_update = PDF_distribution[PDF_distribution[:, 1] > 0.0].T
#     else: PDF_distribution_update = PDF_distribution.T
#     return (PDF_distribution_update)

'''
author: Alvason Zhenhua Li
date:   02/14/2016

Home-made machinery for probability mass function 
'''
def AlvaPMF(dataX, minX = None, maxX = None, total_level = None, normalization = True, empty_leveler_filter = True):
    dataX = np.asarray(dataX)
    # a integering-data step will secure the next leveling-data step
    dataX = np.int64(dataX)
    # filter out negative and zero data
    dataX = dataX[dataX > 0]
    if minX is None:
        minX = min(dataX)       
    if maxX is None:
        maxX = max(dataX)
    if total_level is None:
        leveler = np.arange(int(minX), int(maxX) + 1) 
        total_level = len(leveler)
    else:
        leveler = np.arange(int(minX), int(maxX) + 1, (maxX - minX) / total_level) 
    leveleee = np.zeros([total_level])
    # sorting data into level (leveling-data step)
    for i in range(total_level):
        leveleee[i] = np.sum([dataX[:] == leveler[i]]) 
    if normalization:
        leveleee = leveleee / np.sum(leveleee)
    # associating (leveler_n, levelee_n)...it is important for the next filter-zero step
    PMF_distribution = np.array([leveler, leveleee]).T   
    # filter out empty-leveler (nothing inside the leveler)...it is important for the future log-step
    if empty_leveler_filter:
        PMF_distribution_update = PMF_distribution[PMF_distribution[:, 1] > 0.0].T
    else: PMF_distribution_update = PMF_distribution.T
    return (PMF_distribution_update)




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

