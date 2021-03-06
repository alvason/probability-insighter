
# coding: utf-8

# # Probability-insighter
# https://github.com/alvason/probability-insighter
# 
# ### Power-law distribution

# In[1]:

'''
author: Alvason Zhenhua Li
date:   03/19/2015
'''
get_ipython().magic(u'matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import os

import alva_machinery_probability as alva

AlvaFontSize = 23
AlvaFigSize = (16, 6)
numberingFig = 0
# for saving figure
saving_dir_path = '/Users/al/Desktop/GitHub/probability-insighter/figure'
file_name = 'power-law-distribution'
AlvaColorCycle = ['blue', 'green', 'cyan'
                  , 'pink', 'purple', 'deepskyblue'
                  , 'red', 'lime']
###############
import datetime
previous_running_time = datetime.datetime.now()
print ('Previous running time is {:}').format(previous_running_time)


# ### Probability of only one member with a unique face (k = 1 from m-dice of b-face)
# ### $ Pr(k = 1 | b, m) = m (b - 1)^{(m - k)} $
# 
# ### Probability of only two members with a unique face (k = 2 from m-dice of b-face)
# ### $ Pr(k = 2 | b, m) = \frac{m!}{k!(m - k)!} (b - 1)^{(m - k)} $

# In[2]:

# 1'23456'---1 work_way
# 1'23456'---1 work_way 
# '1'23456---5 work_way
# '1'23456---5 work_way
# '1'23456---5 work_way


# In[3]:

class multinomial_D(object):
    def __init__(cell, base = None, digit = None
                 , wanted_event = None, total_wanted = None
                 , total_sampling = None, **kwargs):
        if base is None:
            base = 6
        cell.base = base
        if digit is None:
            digit = 10
        cell.digit = digit
        if wanted_event is None:
            wanted_event = 0
        cell.wanted_event = wanted_event
        if total_wanted is None:
            total_wanted = 1
        cell.total_wanted = total_wanted
        if total_sampling is None:
            total_sampling = 10**4
        cell.total_sampling = total_sampling
        # initialzing the possible_way_all
        cell.possible_way_all = cell.possible_way()
        # initialzing the work_way for randomSeed
        cell.work_way_all_all = cell.sampling_pmf()
    
    # distribution of probability_mass_function
    def sampling_pmf(cell, digitX = None):
        if digitX is None:
            digitX = np.arange(cell.digit + 1)
        digitX = np.asarray(digitX)
        # a integering-data step
        digitX = np.int64(digitX)
        # filter out negative and zero data
        digitX = digitX[digitX >= 0]
        watching = alva.TimeWatch()
        probability = np.zeros([len(digitX)])
        work_way_all_all = []
        for xn in range(len(digitX)):
            cell.total_wanted = digitX[xn]
            possible_way_all = cell.possible_way()
            work_way_all = cell.work_way()
            work_way_all_all.append(possible_way_all.loc[work_way_all.index])
            total_possible_way = len(possible_way_all)
            total_work_way = len(work_way_all)
            probability[xn] = float(total_work_way) / total_possible_way  
            watching.progressBar(1, xn + 1, len(digitX))
        cell.work_way_all_all = work_way_all_all
        return (digitX, probability, work_way_all_all)
        
    def possible_way(cell, base = None, digit = None, total_sampling = None):
        if base is None:
            base = cell.base
        if digit is None:
            digit = cell.digit
        if total_sampling is None:
            total_sampling = cell.total_sampling
        ###
        sampling_way_all = np.zeros([total_sampling, digit])
        for sn in range(total_sampling):
            sampling_way = np.zeros([digit])
            for dn in range(digit):
                sampling_way[dn] = int(base * np.random.random())
            sampling_way_all[sn] = sampling_way
        way_all = pd.DataFrame(sampling_way_all, columns = ['event_unit_' + str(i) for i in np.arange(digit)])
        possible_way_all = way_all.drop_duplicates()
        cell.possible_way_all = possible_way_all
        return (cell.possible_way_all)

    def work_way(cell, possible_way_all = None, wanted_event = None, total_wanted = None):
        if possible_way_all is None:
            possible_way_all = cell.possible_way_all
        if wanted_event is None:
            wanted_event = cell.wanted_event
        if total_wanted is None:
            total_wanted = cell.total_wanted
        # like_way is a way with at least one-wanted 
        like_way_all = possible_way_all[possible_way_all == wanted_event]
        work_way_all = like_way_all[like_way_all.isnull().sum(axis = 1) == (cell.digit - total_wanted)]
        cell.work_way_all = work_way_all
        return (cell.work_way_all)
    
    # distribution of probability_mass_function
    def reality_pmf(cell, digitX = None):
        if digitX is None:
            digitX = np.arange(cell.digit + 1)
        digitX = np.asarray(digitX)
        # a integering-data step
        digitX = np.int64(digitX)
        # filter out negative and zero data
        digitX = digitX[digitX >= 0]
        probability = np.zeros([len(digitX)])
        for xn in range(len(digitX)):
            cell.total_wanted = digitX[xn]
            probability[xn] = cell.base_digit_reality()
        return (digitX, probability)
    
    def base_digit_reality(cell, base = None, digit = None, total_wanted = None):
        if base is None:
            base = cell.base
        if digit is None:
            digit = cell.digit
        if total_wanted is None:
            total_wanted = cell.total_wanted
        base = float(base)
        digit = float(digit)
        k = float(total_wanted)
        total_possible_way = base**digit
        binomial_coefficient = float(alva.productA(digit)) / (alva.productA(k) * alva.productA(digit - k)) 
        total_work_way = binomial_coefficient * (base - 1)**(digit - k)
        probability = total_work_way / total_possible_way
        return (probability)

    def randomSeed_multinomial(cell, work_way_all_all = None):
        if work_way_all_all is None:
            work_way_all_all = cell.work_way_all_all
        total_level = len(work_way_all_all)
        output_range = total_level
        total_work_way = np.zeros([total_level])
        leveleee_all = []
        for i in range(total_level):
            total_work_way[i] = work_way_all_all[i].shape[0]
            leveleee = (output_range * float(i + 1) / total_level) * np.ones([total_work_way[i]]) 
            leveleee_all.append(leveleee)
        # randomly picking up work_way
        output_work_way = np.concatenate(leveleee_all)
        randomSeed = np.zeros([len(output_work_way)])
        index_record = np.array([0])
        for i in range(len(output_work_way)):
            random_index = int(len(output_work_way) * np.random.random())
            while (index_record.any() == random_index): 
                random_index = int(len(output_work_way) * np.random.random())
            randomSeed[i] = output_work_way[random_index]
            index_record = np.append(index_record, random_index) 
        return (randomSeed)  
############################# 
#if __name__ == '__main__':
aMD = multinomial_D(base = 2, digit = 8, total_sampling = 1000)
ppp = aMD.possible_way()
www = aMD.work_way(total_wanted = 7)
ppp.loc[www.index]


# In[4]:

##########################################
xx_all = []
pp_all = []
xx_reality_all = []
pp_reality_all = []
base_list = np.array([2])
for bn in range(len(base_list)):
    aMD = multinomial_D(base = base_list[bn], digit = 32, total_sampling = 1000)
    samplingD = aMD.sampling_pmf()
    xx_all.append(samplingD[0])
    pp_all.append(samplingD[1])
    work_way_all_all = samplingD[2]
    ##
    realityD = aMD.reality_pmf()
    xx_reality_all.append(realityD[0])
    pp_reality_all.append(realityD[1])
    
### plotting
figure_name = '-sampling-reality-base{:}'.format(aMD.base)
file_suffix = '.png'
save_figure = os.path.join(saving_dir_path, file_name + figure_name + file_suffix)
numberingFig = numberingFig + 1
# plotting1
figure = plt.figure(numberingFig, figsize = (16, 9))
window1 = figure.add_subplot(1, 1, 1)
for i in range(len(base_list)):
    window1.plot(xx_reality_all[i], pp_reality_all[i], marker ='o', markersize = 6
           , color = AlvaColorCycle[i], alpha = 0.9, label = 'reality (base-b = {:})'.format(base_list[i]))
    window1.plot(xx_all[i], pp_all[i], marker = 'o', markersize = 20
                 , color = AlvaColorCycle[i], alpha = 0.5, linewidth = 0
                 , label = 'sampling (size={:}), totalP = {:1.3f}'.format(aMD.total_sampling, pp_all[i].sum()))
plt.ylim(0, 0.5)
plt.title(r'$ Multinomial \ distribution-PMF $'
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ k (total-wanted \ in \ digit-d = {:}) $'.format(aMD.digit), fontsize = AlvaFontSize)
plt.ylabel(r'$ Pr(k|b, d) $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.8)
plt.yticks(fontsize = AlvaFontSize*0.8) 
plt.grid(True)
plt.legend(loc = (1, 0), fontsize = AlvaFontSize)
figure.tight_layout() 
plt.savefig(save_figure, dpi = 300, bbox_inches = 'tight')
plt.show()


# In[5]:

randomSeed = aMD.randomSeed_multinomial(work_way_all_all)
meanP = randomSeed.sum() / len(randomSeed)
total_event = len(randomSeed)
totalLevel = int(total_event/1)
category = alva.AlvaLevel(randomSeed, totalLevel, False)
gLevel = category[0]
numberLevel = category[1]

gInput = np.arange(total_event)


# plotting
figure_name = '-random_seed_base{:}'.format(aMD.base)
file_suffix = '.png'
save_figure = os.path.join(saving_dir_path, file_name + figure_name + file_suffix)

numberingFig = numberingFig + 1
figure = plt.figure(numberingFig, figsize = AlvaFigSize)
plot1 = figure.add_subplot(1, 2, 1)
plot1.plot(gInput, randomSeed, color = 'gray', marker = 'o', label = 'data')
plot1.plot(gInput, alva.AlvaMinMax(randomSeed), color = 'red', marker = 'o', label = 'minMaxSorting')
plot1.grid(True)
plt.title(r'$ Multinomial \ randomness \ (base-b = {:}) $'.format(aMD.base), fontsize = AlvaFontSize)
plt.xlabel(r'$ event-input $', fontsize = AlvaFontSize)
plt.ylabel(r'$ output $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 
plt.legend(loc = (0, -0.2))

plot2 = figure.add_subplot(1, 2, 2)
plot2.plot(numberLevel, gLevel, color = 'red', marker = 'o', label = 'category') 

plot2.grid(True)
plt.title(r'$ Multinomial \ distribution \ (events = {ev:},\ levels = {le:}) $'.format(ev = total_event, le = totalLevel)
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ event/level $', fontsize = AlvaFontSize)
plt.ylabel(r'$ level-range $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 
plt.legend(loc = (0, -0.2))

figure.tight_layout()
plt.savefig(save_figure, dpi = 300)
plt.show()


# In[6]:

class AlvaDistribution(object):
    def __init__(cell, dataX, minX = None, maxX = None, fit_method = None, parameter = None, **kwargs):
        dataX = dataX[dataX > 0]
        if minX is None:
            minX = min(dataX)       
        if maxX is None:
            maxX = max(dataX)
        dataX = dataX[dataX >= minX]
        dataX = dataX[dataX <= maxX]
        cell.dataX = dataX
        cell.minX = minX
        cell.maxX = maxX
        if fit_method is None:
            fit_method = 'Likelihood'
        cell.fit_method = fit_method
        if parameter is None:
            cell.parameter = cell._initial_parameter()
        else:
            cell._set_parameter(parameter)
        
    def fit(cell):
        if cell.current_distribution == 'EwensSampling':
            EwensPMF = cell._probability_mass_function(cell.dataX, cell.minX)
            cell.EwensPMFxx = EwensPMF[0]
            cell.EwensPMFyy = EwensPMF[1]
            optimized_meter = EwensPMF[2]
            # logLikelihood
            # firstly, for accurate AIC, need to restore to original-size from unique-size
            ePMF = AlvaPMF(cell.dataX, minX = cell.minX, normalization = False)
            all_data = cell.EwensPMFyy * ePMF[1][0:len(cell.EwensPMFyy)]
            log_pmf = np.log(all_data)
            logLike = np.sum(log_pmf)
            cell.max_logLikelihood = logLike
            # update parameter
            cell._set_parameter(optimized_meter) 
            return (optimized_meter)
        else:
            if cell.fit_method == 'Likelihood':
                def fit_function(alpha_meter):
                    return cell.logLikelihood(alpha_meter)
            # search the maximum (minimum for negative values)
            from scipy import optimize
            optimizing = optimize.fmin(lambda alpha_meter: -fit_function(alpha_meter)
                                       , cell._initial_parameter(), full_output = 1, disp = False)

            optimized_meter = optimizing[0]
            negative_max_logLike = optimizing[1]
            cell.max_logLikelihood = -negative_max_logLike
        # update parameter
        cell._set_parameter(optimized_meter) 
        return (optimized_meter)

    def logLikelihood(cell, alpha_meter):
        cell._set_parameter(alpha_meter)
        log_pmf = np.log(cell.pmf(cell.dataX, cell.minX)[1])
        logL = np.sum(log_pmf)
        return (logL)

    # distribution of probability_mass_function
    def pmf(cell, dataX, minX):
        if cell.current_distribution == 'EwensSampling':
            xx = cell.EwensPMFxx
            probability = cell.EwensPMFyy
        else:
            xx = np.unique(dataX)
            probability = cell._probability_mass_function(dataX, minX)
        return (xx, probability)
    
    def random_integer_simulator(cell, total_dataX = None, alpha = None, minX = None):
        if total_dataX is None:
            total_dataX = len(cell.dataX)
        if alpha is None:
            alpha = cell.alpha
        if minX is None:
            minX = cell.minX
        rr = np.random.uniform(size = total_dataX)
        def pdf_cdf_connection(x, randomP):
            connection = cell._cumulative_distribution_function(dataX = x, minX = minX, alpha = alpha) - (1 - randomP)
            return connection
        from scipy import optimize
        xx = []
        for r in rr:
            solving = optimize.root(lambda x: pdf_cdf_connection(x, r), minX)
            xx.append(solving.x[0])
        xx = np.asarray(xx, dtype = int)
        #xx = np.floor(xx)
        return xx

    def information_criterion(cell, logLike = None, total_parameter = None, total_sample = None):
        if logLike is None:
            logLike = cell.max_logLikelihood
        if total_parameter is None:
            total_parameter = len(cell.parameter)
        if total_sample is None:
            total_sample = len(np.unique(cell.dataX))
        AIC = -2 * logLike + (2 * total_parameter)
        BIC = -2 * logLike + total_parameter * np.log(total_sample)
        return np.array([AIC, BIC])

    def AlvaIntSequence(cell):
        xx = np.arange(1, len(cell.dataX))
        yy = cell._probability_mass_function(xx)
        yy = yy / yy.min()
        integer_sequence = np.floor(yy)
        return (integer_sequence)

###
class PowerLaw(AlvaDistribution):
    def __init__(cell, dataX, **kwargs):
        AlvaDistribution.__init__(cell, dataX, **kwargs)
        cell.current_distribution = 'PowerLaw'
    
    def _probability_mass_function(cell, dataX = None, minX = None):
        if dataX is None:
            dataX = cell.dataX
        if minX is None:
            minX = cell.minX
        from scipy.special import zeta
        constantN = 1.0 / zeta(cell.alpha, minX)
        xPower = dataX**(-cell.alpha)
        c_xPower = constantN * xPower 
        return (c_xPower) 

    def _cumulative_distribution_function(cell, dataX = None, minX = None, alpha = None):
        if dataX is None:
            dataX = cell.dataX
        if minX is None:
            minX = cell.minX
        if alpha is None:
            alpha = cell.alpha
        from scipy.special import zeta
        total_level = len(dataX)
        power_cdf = np.zeros(total_level)
        for xn in range(total_level):
            power_cdf[xn] = zeta(alpha,  dataX[xn]) / zeta(alpha,  minX)
        return power_cdf
        
    # distributions with alpha <= 1 are not valid (non-normalizable)
    def _valid_parameter_range(cell):
        return (cell.alpha > 1)
    
    def _initial_parameter(cell):
        # using the exact value of continuous-case as the initial guessing-value of discrete-case fitting
        # cell.alpha = 1 + (len(cell.dataX) / np.sum(np.log(cell.dataX / (cell.minX))))
        cell.alpha = 2.0
        cell.parameter = np.array([cell.alpha])
        return (cell.parameter)

    def _set_parameter(cell, parameter):
        # if parameter is a scalar not array
        if isinstance(parameter, (int, float)):
            parameter = [parameter]
        # for converting numpy-scalar (0-dimensional array()) to 0-dimensional array([]) 
        parameter = np.atleast_1d(parameter)
        parameter = np.asarray(parameter, dtype = float)
        cell.parameter = parameter
        cell.alpha = cell.parameter[0]
        return(cell.parameter)
##############################
###### test-data 1 ###########
test_alpha = float(2.308)
total_event = 300
xx = np.arange(1, total_event + 1)
ap = PowerLaw(xx)
pp = ap.random_integer_simulator(total_dataX = total_event, minX = 1, alpha = test_alpha)
ppP = np.sort(pp)
ppP = ppP[::-1]
#print ('ppP', ppP)
pdm =alva.AlvaPMF(pp)

### plotting
figure = plt.figure(figsize = (18, 6))
window1 = figure.add_subplot(1, 2, 1)
window1.plot(xx, pp, marker = 'o', markersize = 10, color = 'red', alpha = 0.3)
plt.title(r'$ ramdon-process-PMF $', fontsize = AlvaFontSize*0.8)
plt.xlabel(r'$ cell-number (N={:}) $'.format(total_event), fontsize = AlvaFontSize*0.8)
plt.ylabel(r'$ P(n|N) $', fontsize = AlvaFontSize*0.8)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6)
plt.xscale('log', basex = 10)
plt.yscale('log', basex = 10)
plt.grid(True)

### plotting
window2 = figure.add_subplot(1, 2, 2)
window2.plot(pdm[0], pdm[1], marker = '*', markersize = 10, color = 'blue', alpha = 0.3)
### raw data
barWidth = 1.0/2
window2.bar(pdm[0] - barWidth/2, pdm[1], width = barWidth
           , edgecolor = 'blue', color = 'blue', alpha = 0.4
           , label = '$ Empirical-distribution $')
plt.title(r'$ ramdon-process-PMF $', fontsize = AlvaFontSize*0.8)
plt.xlabel(r'$ cell-number (N={:}) $'.format(total_event), fontsize = AlvaFontSize*0.8)
plt.ylabel(r'$ P(n|N) $', fontsize = AlvaFontSize*0.8)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6)
plt.xscale('log', basex = 10)
plt.yscale('log', basex = 10)
plt.grid(True)
plt.xlim(0, 1000)

plt.show()


# In[76]:

test_alpha = float(2.9)
birth_probability = 0.6
total_event = 10**3
xx = np.arange(1, total_event + 1)
###
minX = 1

m = (1 - birth_probability) / birth_probability

c = (test_alpha - 2) * m - minX
power_alpha = 2 + (minX + c) / m
print ('power_alpha = {:}'.format(power_alpha))
species = np.zeros([total_event])

i = 0
species[i] = np.ceil(minX + c)
s = species[i]

while i < (total_event - 1):
    # random (0, 1)
    random_seed = np.random.random()
    if random_seed < birth_probability:
        i = i + 1
        species[i] = np.ceil(minX + c)
        s = s + species[i]
    else:
        # random (0, s)
        random_seed = s * np.random.random()
        x = 1
        sp = species[0]
        while (random_seed > sp) and x < (total_event - 1):
            x = x + 1
            sp = sp + species[x]
        species[x] = species[x] + 1

pp = species

ppP = np.sort(pp)
ppP = ppP[::-1]
print ('ppP', ppP[0:100])
pdm =alva.AlvaPMF(pp)

### plotting
figure = plt.figure(figsize = (18, 6))
window1 = figure.add_subplot(1, 2, 1)
window1.plot(xx, pp, marker = 'o', markersize = 10, color = 'red', alpha = 0.3)
plt.title(r'$ Yule-process $', fontsize = AlvaFontSize*0.8)
plt.xlabel(r'$ Process-step $'.format(total_event), fontsize = AlvaFontSize*0.8)
plt.ylabel(r'$ Circle-area $', fontsize = AlvaFontSize*0.8)
plt.xticks(fontsize = AlvaFontSize*0.8)
plt.yticks(fontsize = AlvaFontSize*0.8)
plt.xscale('log', basex = 10)
plt.yscale('log', basex = 10)
plt.grid(True)

### plotting
window2 = figure.add_subplot(1, 2, 2)
window2.plot(pdm[0], pdm[1], marker = '*', markersize = 10, color = 'blue', alpha = 0.3)
### raw data
barWidth = 1.0/2
window2.bar(pdm[0] - barWidth/2, pdm[1], width = barWidth
           , edgecolor = 'blue', color = 'blue', alpha = 0.4
           , label = '$ Empirical-distribution $')
plt.title(r'$ Yule-process-PMF $', fontsize = AlvaFontSize*0.8)
plt.xlabel(r'$ Circle-area $', fontsize = AlvaFontSize*0.8)
plt.ylabel(r'$ Count \ of \ circle $', fontsize = AlvaFontSize*0.8)
plt.xticks(fontsize = AlvaFontSize*0.8)
plt.yticks(fontsize = AlvaFontSize*0.8)
plt.xscale('log', basex = 10)
plt.yscale('log', basex = 10)
plt.grid(True)
plt.xlim(0, 1000)
plt.legend(fontsize = AlvaFontSize*0.8)
plt.show()


# In[ ]:



