
# coding: utf-8

# # Probability-insighter
# https://github.com/alvason/probability-insighter
# 
# ### Binomial random distribution

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
file_name = 'multinomial-distribution'
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
                 , wanted_event = None
                 , total_wanted = None
                 , total_sampling = None
                 , total_unit = None, **kwargs):
        if base is None:
            base = 6
        cell.base = base
        if digit is None:
            digit = 2
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
    
    # distribution of probability_mass_function
    def sampling_pmf(cell, digitX):
        #digitX = np.asarray(digitX)
        # a integering-data step
        digitX = np.int64(digitX)
        # filter out negative and zero data
        digitX = digitX[digitX > 0]
        # avoiding invalid input because digit >= total_wanted 
        digitX = digitX[digitX >= cell.total_wanted]
        probability = []
        watching = alva.TimeWatch()
        for xn in digitX:
            cell.digit = xn
            possible_way_all = cell.possible_way()
            work_way_all = cell.work_way()
            total_possible_way = len(possible_way_all)
            total_work_way = len(work_way_all)
            pp = float(total_work_way) / total_possible_way
            probability.append(pp)
            watching.progressBar(1, np.argwhere([digitX == xn])[0][1] + 1, len(digitX))
        return (digitX, probability)
        
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
    def reality_pmf(cell, digitX):
        #digitX = np.asarray(digitX)
        # a integering-data step
        digitX = np.int64(digitX)
        # filter out negative and zero data
        digitX = digitX[digitX > 0]
        # avoiding invalid input because digit >= total_wanted 
        digitX = digitX[digitX >= cell.total_wanted]
        probability = []
        for xn in digitX:
            cell.digit = xn
            aaa = cell.base_digit_reality()
            probability.append(aaa)
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
############################# 
#if __name__ == '__main__':


# In[4]:

##########################################
max_member = 50
digitX = np.arange(1, max_member)
###
kk = np.arange(1, 16, 2)
xx_all = []
pp_all = []
xx_reality_all = []
pp_reality_all = []
for kn in kk:
    aMD = multinomial_D(base = 2, total_wanted = kn, total_sampling = 1000)
    samplingD = aMD.sampling_pmf(digitX)
    xx_all.append(samplingD[0])
    pp_all.append(samplingD[1])
    ##
    realityD = aMD.reality_pmf(digitX)
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
for kn in np.arange(len(kk)):
    window1.plot(xx_reality_all[kn], pp_reality_all[kn], marker ='o', markersize = 6
           , color = AlvaColorCycle[kn], alpha = 0.9, label = 'reality (k = {:})'.format(kk[kn]))
    window1.plot(xx_all[kn], pp_all[kn], marker = 'o', markersize = 20
         , color = AlvaColorCycle[kn], alpha = 0.5, label = 'sampling ({:})'.format(aMD.total_sampling), linewidth = 0)
plt.ylim(0, 0.6)
plt.title(r'$ Multinomial \ distribution-PMF \ (base \ b = {:}) $'.format(aMD.base), fontsize = AlvaFontSize)
plt.xlabel(r'$ m \ (member/run) $', fontsize = AlvaFontSize)
plt.ylabel(r'$ Pr(k|b, m) $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.8)
plt.yticks(fontsize = AlvaFontSize*0.8) 
plt.grid(True)
plt.legend(loc = (1, 0), fontsize = AlvaFontSize)
figure.tight_layout() 
plt.savefig(save_figure, dpi = 300, bbox_inches = 'tight')
plt.show()


# In[5]:

##########################################
max_member = 50
digitX = np.arange(1, max_member)
###
kk = np.arange(1, 16, 2)
xx_all = []
pp_all = []
xx_reality_all = []
pp_reality_all = []
for kn in kk:
    aMD = multinomial_D(base = 6, total_wanted = kn, total_sampling = 1000)
    samplingD = aMD.sampling_pmf(digitX)
    xx_all.append(samplingD[0])
    pp_all.append(samplingD[1])
    ##
    realityD = aMD.reality_pmf(digitX)
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
for kn in np.arange(len(kk)):
    window1.plot(xx_reality_all[kn], pp_reality_all[kn], marker ='o', markersize = 6
           , color = AlvaColorCycle[kn], alpha = 0.9, label = 'reality (k = {:})'.format(kk[kn]))
    window1.plot(xx_all[kn], pp_all[kn], marker = 'o', markersize = 20
         , color = AlvaColorCycle[kn], alpha = 0.5, label = 'sampling ({:})'.format(aMD.total_sampling), linewidth = 0)
plt.ylim(0, 0.6)
plt.title(r'$ Multinomial \ distribution-PMF \ (base \ b = {:}) $'.format(aMD.base), fontsize = AlvaFontSize)
plt.xlabel(r'$ m \ (member/run) $', fontsize = AlvaFontSize)
plt.ylabel(r'$ Pr(k|b, m) $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.8)
plt.yticks(fontsize = AlvaFontSize*0.8) 
plt.grid(True)
plt.legend(loc = (1, 0), fontsize = AlvaFontSize)
figure.tight_layout() 
plt.savefig(save_figure, dpi = 300, bbox_inches = 'tight')
plt.show()


# In[6]:

# plotting
figure_name = '-equation'
file_suffix = '.png'
save_figure = os.path.join(saving_dir_path, file_name + figure_name + file_suffix)

numberingFig = numberingFig + 1
plt.figure(numberingFig, figsize=(9, 6))
plt.axis('off')
plt.title(r'$ Binomial-distribution---probability-mass-function $',fontsize = AlvaFontSize)
plt.text(0, 5.0/6, r'$ P(n|N) = \frac{N!}{n!(N - n)!} p^n (1 - p)^{N - n} $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 4.0/6, r'$ 1-- \ a \ set \ of \ N \ events $', fontsize = AlvaFontSize)
plt.text(0, 3.0/6, r'$ 2-- \ either \ Success \ or \ Failure \ for \ each \ event $', fontsize = AlvaFontSize)
plt.text(0, 2.0/6, r'$ 3-- \ the \ probability-p \ is \ the \ same \ for \ all \ events $', fontsize = AlvaFontSize)
plt.text(0, 1.0/6, r'$ 4-- \ P(n|N) \ is \ the \ probability \ with \ n-success \ events \ in \ a \ total \ of \ N \ events $',
         fontsize = AlvaFontSize)
plt.savefig(save_figure, dpi = 300)
plt.show()


# In[7]:

numberingFig = numberingFig + 1
plt.figure(numberingFig, figsize=(9, 6))
plt.axis('off')
plt.title(r'$ Binomial-distribution---cumulative-distribution-function $',fontsize = AlvaFontSize)
plt.text(0, 5.0/6, r'$ P(n|N) = \sum_{m=0}^n \ \frac{N!}{m!(N - m)!} p^m (1 - p)^{N - m} $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 4.0/6, r'$ 1-- \ a \ set \ of \ N \ events $', fontsize = AlvaFontSize)
plt.text(0, 3.0/6, r'$ 2-- \ either \ Success \ or \ Failure \ for \ each \ event $', fontsize = AlvaFontSize)
plt.text(0, 2.0/6, r'$ 3-- \ the \ probability-p \ is \ the \ same \ for \ all \ events $', fontsize = AlvaFontSize)
plt.text(0, 1.0/6, r'$ 4-- \ P(n|N) \ is \ the \ cumulative-probability \ with \ n-success \ events \ in \ a \ total \ of \ N \ events $',
         fontsize = AlvaFontSize)
plt.show()


# In[8]:

#testing
i = 100
print ('AlvaM =', alva.productA(i))
print ('NumPy =', np.arange(1, i + 1).prod(dtype = float))


def AlvaBinomialD(i, N, p):
    B_distribution = 0.0*i
    B_distribution[:] = alva.productA(N) / (alva.productA(i[:]) * alva.productA(N - i[:]))                         * p**i[:] * (1 - p)**(N - i[:])
    return (B_distribution)

def AlvaBinomialC(m, N, p, binomialD):
    B_C = 0.0*m
    for j in range(np.size(m)):
        for k in range(m[j]):
            i = np.arange(k + 1)
            B_distribution = binomialD(i, N, p)
        B_C[j] = B_C[j] + B_distribution.sum()
    return (B_C)

total_event = int(30)
i_event = np.arange(1, total_event + 1)
totalPoint_Input = total_event
probability_each = 0.5

binomial_D = AlvaBinomialD(i_event, total_event, probability_each)
print ('total-probability = {:f}'.format(binomial_D.sum()))
# plotting1
figure = plt.figure(numberingFig, figsize = AlvaFigSize)
plot1 = figure.add_subplot(1, 2, 1)
plot1.plot(i_event, binomial_D, marker ='o', color = 'green')
plt.title(r'$ Binomial \ distribution-PMF \ (p={:}) $'.format(probability_each), fontsize = AlvaFontSize)
plt.xlabel(r'$ n-event \ within \ total-event \ (N={:}) $'.format(total_event), fontsize = AlvaFontSize)
plt.ylabel(r'$ P(n|N) $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 
plt.ylim(0, 1) 
# plotting2
i_event = np.arange(1, total_event + 1)
binomial_C = AlvaBinomialC(i_event, total_event, probability_each, AlvaBinomialD)
plot2 = figure.add_subplot(1, 2, 2)
plot2.plot(i_event, binomial_C, marker ='o', color = 'red')
plt.title(r'$ Binomial \ distribution-CDF \ (p={:}) $'.format(probability_each), fontsize = AlvaFontSize)
plt.xlabel(r'$ n-event \ within \ total-event \ (N={:}) $'.format(total_event), fontsize = AlvaFontSize)
plt.ylabel(r'$ P(n|N) $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6)   
figure.tight_layout()
plt.show()


# In[9]:

# plotting a list
number_list = 9
probability_each_list = np.arange(1, number_list + 1)*1.0/(number_list + 1)
print ('probability_each_list =', probability_each_list)

binomial_D_list = np.zeros([number_list, total_event])
for i in range(number_list):
    binomial_D_list[i] = AlvaBinomialD(i_event, total_event, probability_each_list[i])
#    print ('total-probability = {:f}'.format(binomial_D_list[i].sum()))
binomial_C_list = np.zeros([number_list, total_event])
for i in range(number_list):
    binomial_C_list[i] = AlvaBinomialC(i_event, total_event, probability_each_list[i], AlvaBinomialD)

# plotting
figure = plt.figure(numberingFig, figsize = AlvaFigSize)
plot1 = figure.add_subplot(1, 2, 1)
plot1.plot(i_event, binomial_D_list.T, marker ='o')
if totalPoint_Input < 100:
    plot1.set_xticks(i_event, minor = True) 
    plot1.set_yticks(binomial_D_list[(number_list + 1)/2], minor = True) 
    plot1.grid(True, which = 'minor')
else:
    plot1.grid(True, which = 'major')
plt.title(r'$ Binomial \ distribution-PMF $', fontsize = AlvaFontSize)
plt.xlabel(r'$ n-event \ within \ total-event \ (N={:}) $'.format(total_event), fontsize = AlvaFontSize)
plt.ylabel(r'$ P(n|N) $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 

plot2 = figure.add_subplot(1, 2, 2)
plot2.plot(i_event, binomial_C_list.T, marker ='o')
if totalPoint_Input < 100:
    plot2.set_xticks(i_event, minor = True) 
    plot2.set_yticks(binomial_C_list[(number_list + 1)/2], minor = True) 
    plot2.grid(True, which = 'minor')
else:
    plot2.grid(True, which = 'major')
plt.title(r'$ Binomial \ distribution-CDF $', fontsize = AlvaFontSize)
plt.xlabel(r'$ n-event \ within \ total-event \ (N={:}) $'.format(total_event), fontsize = AlvaFontSize)
plt.ylabel(r'$ P(n|N) $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 

figure.tight_layout()
plt.show()


# In[10]:

'''Binomial randomness --- Binomial distribution'''

total_event = int(100)
gInput = np.arange(total_event)
output_level = total_event
probability_peak = 0.1
randomSeed = np.random.binomial(output_level, probability_peak, total_event)

sumP = 0
for i in range(total_event):
    sumP = sumP + randomSeed[i]
meanP = sumP/(total_event)

totalLevel = int(total_event/1)
category = alva.AlvaLevel(randomSeed, totalLevel, False)
gLevel = category[0]
numberLevel = category[1]

binomial_D = total_event*AlvaBinomialD(np.arange(totalLevel), totalLevel, probability_peak)

# plotting
figure_name = ''
file_suffix = '.png'
save_figure = os.path.join(saving_dir_path, file_name + figure_name + file_suffix)

numberingFig = numberingFig + 1
figure = plt.figure(numberingFig, figsize = AlvaFigSize)
plot1 = figure.add_subplot(1, 2, 1)
plot1.plot(gInput, randomSeed, color = 'gray', marker = 'o', label = 'data')
plot1.plot(gInput, alva.AlvaMinMax(randomSeed), color = 'red', marker = 'o', label = 'minMaxSorting')
if total_event < 100:
    plot1.set_xticks(gInput, minor = True) 
    plot1.set_yticks(randomSeed, minor = True)
    plot1.grid(True, which = 'minor')
else:
    plot1.grid(True, which = 'major')
plt.title(r'$ Binomial \ randomness \ (mean = {:1.6f}) $'.format(meanP), fontsize = AlvaFontSize)
plt.xlabel(r'$ event-input $', fontsize = AlvaFontSize)
plt.ylabel(r'$ output $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 
plt.legend(loc = (0, -0.2))

plot2 = figure.add_subplot(1, 2, 2)
plot2.plot(numberLevel, gLevel, color = 'red', marker = 'o', label = 'category') 
plot2.plot(binomial_D, np.arange(totalLevel), color = 'blue', marker = 'o', label = 'Binomial distribution') 
if total_event < 100:
    plot2.set_xticks(numberLevel, minor = True) 
    plot2.set_yticks(gLevel, minor = True)
    plot2.grid(True, which = 'minor')
else:
    plot2.grid(True, which = 'major')
plt.title(r'$ Binomial \ distribution \ (events = {ev:},\ levels = {le:}) $'.format(ev = total_event, le = totalLevel)
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ event/level $', fontsize = AlvaFontSize)
plt.ylabel(r'$ level-range $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 
plt.legend(loc = (0, -0.2))

figure.tight_layout()
plt.savefig(save_figure, dpi = 300)
plt.show()

