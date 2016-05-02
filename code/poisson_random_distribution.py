
# coding: utf-8

# # Probability-insighter
# https://github.com/alvason/probability-insighter
# 
# ### Poisson random distribution 

# In[1]:

'''
author: Alvason Zhenhua Li
date:   03/19/2015
'''

get_ipython().magic(u'matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import os
dir_path = '/Users/al/Desktop/GitHub/probability-insighter/figure'
file_name = 'poisson-distribution'

import alva_machinery_probability as alva

AlvaFontSize = 23
AlvaFigSize = (16, 7)
numberingFig = 0

# plotting
figure_name = '-equation'
file_suffix = '.png'
save_figure = os.path.join(dir_path, file_name + figure_name + file_suffix)

numberingFig = numberingFig + 1
plt.figure(numberingFig, figsize=(9, 6))
plt.axis('off')
plt.title(r'$ Poisson-distribution---probability-mass-function $',fontsize = AlvaFontSize)
plt.text(0, 3.0/4, r'$ P(m|n) = e^{-m} \frac{m^n}{n!} $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 2.0/4, r'$ 1-- \ m \ is \ the \ mean \ number \ of \ events \ per \ interval \ (time \ or \ space) $', fontsize = AlvaFontSize)
plt.text(0, 1.0/4, r'$ 2-- \ P(m|n) \ is \ the \ probability \ with \ n-events \ per \ interval $',
         fontsize = AlvaFontSize)
plt.savefig(save_figure, dpi = 300)
plt.show()


# In[2]:

def AlvaProduct(i):
    if type(i) != np.ndarray:
        i = np.array([i], dtype = int)
    else:
        i = np.int32(i)
    # set 0! = 1
    A_product = 0.0*i + 1
    for j in range(np.size(i)):
        for k in range(1, i[j] + 1):        
            A_product[j] = A_product[j]*k
    if np.size(i) == 1:
        # get rid of []
        A_product = A_product[0]
    return (A_product)

#testing
i = 100
print ('AlvaM_N! =', AlvaProduct(i))
print ('NumPy_N! =', np.prod(np.arange(1, i + 1), dtype = np.float64))

def AlvaPoissonD(i, meanP):
    P_distribution = 0.0*i
    P_distribution[:] = np.exp(-meanP) * meanP**i[:] / AlvaProduct(i[:])
    return (P_distribution)

def AlvaPoissonC(m, meanP, poissonD):
    B_C = 0.0*m
    for j in range(np.size(m)):
        for k in range(m[j]):
            i = np.arange(k + 1)
            B_distribution = poissonD(i, meanP)
        B_C[j] = B_C[j] + B_distribution.sum()
    return (B_C)

total_event = int(30)
i_event = np.arange(1, total_event + 1)
totalPoint_Input = total_event
meanP = total_event/30.0

poisson_D = AlvaPoissonD(i_event, meanP)

print ('total-probability = {:f}'.format(poisson_D.sum()))
# plotting1
figure = plt.figure(numberingFig, figsize = AlvaFigSize)
plot1 = figure.add_subplot(1, 2, 1)
plot1.plot(i_event, poisson_D, marker ='o', color = 'green')
if totalPoint_Input < 100:
    plot1.set_xticks(i_event, minor = True) 
    plot1.set_yticks(poisson_D, minor = True) 
    plot1.grid(True, which = 'minor')
else:
    plot1.grid(True, which = 'major')
plt.title(r'$ Poisson \ distribution-PMF $', fontsize = AlvaFontSize)
plt.xlabel(r'$ n-event \ with \ mean-event \ (m = {:}) $'.format(meanP), fontsize = AlvaFontSize)
plt.ylabel(r'$ P(m|n) $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 

# plotting2
i_event = np.arange(1, total_event + 1)
poisson_C = AlvaPoissonC(i_event, meanP, AlvaPoissonD)
plot2 = figure.add_subplot(1, 2, 2)
plot2.plot(i_event, poisson_C, marker ='o', color = 'red')
if totalPoint_Input < 100:
    plot2.set_xticks(i_event, minor = True) 
    plot2.set_yticks(poisson_C, minor = True) 
    plot2.grid(True, which = 'minor')
else:
    plot2.grid(True, which = 'major')
plt.title(r'$ Poisson \ distribution-CDF $', fontsize = AlvaFontSize)
plt.xlabel(r'$ n-event \ with \ mean-event \ (m = {:}) $'.format(meanP), fontsize = AlvaFontSize)
plt.ylabel(r'$ P(m|n) $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 

figure.tight_layout()
plt.show()


# In[3]:

'''Poisson randomness --- Poisson distribution'''
total_event = int(300)
gInput = np.arange(total_event)
meanP = 10.0
randomSeed = np.random.poisson(meanP, total_event)

sumP = 0
for i in range(total_event):
    sumP = sumP + (meanP - randomSeed[i])**2
deviationP = (sumP/total_event)**(1.0/2)

totalLevel = int(total_event/10)
category = alva.AlvaLevel(randomSeed, totalLevel, False)
gLevel = category[0]
gLevel_int = gLevel.astype(int)
numberLevel = category[1]
print ('level =', gLevel)
print ('level_int =', gLevel_int)

poisson_D = total_event*AlvaPoissonD(gLevel_int, meanP)
# plotting
figure_name = ''
file_suffix = '.png'
save_figure = os.path.join(dir_path, file_name + figure_name + file_suffix)

numberingFig = numberingFig + 1
figure = plt.figure(numberingFig, figsize = AlvaFigSize)
plot1 = figure.add_subplot(1, 2, 1)
plot1.plot(gInput, randomSeed, color = 'gray', marker = 'o', label = 'data')
plot1.plot(gInput, alva.AlvaMinMax(randomSeed), color = 'red', marker = 'o', label = 'minMaxListing')
if total_event < 100:
    plot1.set_xticks(gInput, minor = True) 
    plot1.set_yticks(randomSeed, minor = True)
    plot1.grid(True, which = 'minor')
else:
    plot1.grid(True, which = 'major')
plt.title(r'$ Poisson \ (mean = {:1.3f},\ deviation = {:1.3f}) $'.format(meanP, deviationP), fontsize = AlvaFontSize)
plt.xlabel(r'$ event-input $', fontsize = AlvaFontSize)
plt.ylabel(r'$ output $', fontsize = AlvaFontSize)
plt.legend(loc = (0, -0.2))
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 

plot2 = figure.add_subplot(1, 2, 2)
plot2.plot(poisson_D, gLevel_int, color = 'blue', marker = 'o', label = 'Poisson Distribution') 
plot2.plot(numberLevel, gLevel, color = 'red', marker = 'o', label = 'category')
if totalLevel < 100:
    plot2.set_xticks(numberLevel, minor = True) 
    plot2.set_yticks(gLevel, minor = True)
    plot2.grid(True, which = 'minor')
else:
    plot2.grid(True, which = 'major')
plt.title(r'$ Poisson \ (events = {:},\ levels = {:}) $'.format(total_event, totalLevel)
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ event/level $', fontsize = AlvaFontSize)
plt.ylabel(r'$ level-range $', fontsize = AlvaFontSize)
plt.legend(loc = (0, -0.2))
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 

figure.tight_layout()
plt.savefig(save_figure, dpi = 300)
plt.show()


# In[ ]:



