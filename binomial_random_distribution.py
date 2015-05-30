# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Probability-insighter
# https://github.com/alvason/probability-insighter
# 
# ### Random distribution --- Binomial distribution (discrete)

# <codecell>

'''
author: Alvason Zhenhua Li
date:   03/19/2015
'''
%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt
import os
dir_path = '/Users/al/Desktop/GitHub/probability-insighter/figure'
file_name = 'binomial-distribution'

import alva_machinery_diffusion as alva

AlvaFontSize = 23
AlvaFigSize = (16, 7)
numberingFig = 0

'''Binomial process --- Binomial distribution'''
figure_name = '-equation'
file_suffix = '.png'
save_figure = os.path.join(dir_path, file_name + figure_name + file_suffix)

numberingFig = numberingFig + 1
plt.figure(numberingFig, figsize=(9, 6))
plt.axis('off')
plt.title(r'$ Binomial-distribution---probability-mass-function $',fontsize = AlvaFontSize)
plt.text(0, 5.0/6, r'$ P_{b}(n|N) = \frac{N!}{n!(N - n)!} p^n (1 - p)^{N - n} $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 4.0/6, r'$ 1-- \ a \ set \ of \ N \ events $', fontsize = AlvaFontSize)
plt.text(0, 3.0/6, r'$ 2-- \ either \ Success \ or \ Failure \ for \ each \ event $', fontsize = AlvaFontSize)
plt.text(0, 2.0/6, r'$ 3-- \ the \ probability-p \ is \ the \ same \ for \ all \ events $', fontsize = AlvaFontSize)
plt.text(0, 1.0/6, r'$ 4-- \ P(n|N) \ is \ the \ probability \ with \ n-success \ events \ in \ a \ total \ of \ N \ events $',
         fontsize = AlvaFontSize)
plt.savefig(save_figure, dpi = 300)
plt.show()

# <codecell>

def AlvaProduct(i):
    if type(i) != np.ndarray:
        i = np.array([i])
    A_product = 0.0*i + 1
    for j in range(np.size(i)):
        for k in range(1, i[j] + 1):        
            A_product[j] = A_product[j]*k
    if np.size(i) == 1:
        A_product = A_product[0]
    return (A_product)

#testing
i = 100
print ('Alva = ', AlvaProduct(i))
print ('NumP = ', np.prod(np.arange(1, i + 1), dtype=np.float64))


def AlvaBinomialD(i, N, p):
    B_distribution = 0.0*i
    B_distribution[:] = AlvaProduct(N)/(AlvaProduct(i[:])*AlvaProduct(N - i[:])) * p**i[:] * (1 - p)**(N - i[:])
    return (B_distribution)


total_event = int(30)
i_event = np.arange(total_event + 1)
totalPoint_Input = i_event.shape[0]
probability_each = 0.5
plt.figure(figsize = (12, 6))
plt.plot(i_event, AlvaBinomialD(i_event, total_event, 0.3), marker ='o', color = 'red', label = 'p=3/10 Distribution')
plt.plot(i_event, AlvaBinomialD(i_event, total_event, probability_each), marker ='o'
         , color = 'green', label = 'p=5/10 Distribution')
plt.plot(i_event, AlvaBinomialD(i_event, total_event, 0.7), marker ='o', color = 'blue', label = 'p=7/10 Distribution')
plt.xlabel(r'$ output-level$', fontsize = AlvaFontSize)
plt.ylabel(r'$ input/level $', fontsize = AlvaFontSize)
plt.title(r'$ Binomial \ distribution $', fontsize = AlvaFontSize)
if totalPoint_Input < 100:
    plt.axes().set_xticks(i_event, minor = True) 
    plt.axes().set_yticks(AlvaBinomialD(i_event, total_event, probability_each), minor = True) 
    plt.grid(True, which = 'minor')
else:
    plt.grid(True, which = 'major')
plt.legend(loc = (1, 0))
plt.show()

# <codecell>

'''Binomial randomness --- Binomial distribution (discrete)'''

totalPoint_Input = int(100)
gInput = np.arange(totalPoint_Input)
output_level = 100
probability_peak = 0.5
randomSeed = np.random.binomial(output_level, probability_peak, totalPoint_Input)

sumI = 0
for i in range(totalPoint_Input):
    sumI = sumI + randomSeed[i]
meanI = sumI/(totalPoint_Input)

totalLevel = int(totalPoint_Input/1)
category = alva.AlvaLevel(randomSeed, totalLevel, False)
gLevel = category[0]
numberLevel = category[1]
print category[2].shape

binomial_D = 100*AlvaBinomialD(np.arange(totalLevel), totalLevel, probability_peak)

figure_name = ''
file_suffix = '.png'
save_figure = os.path.join(dir_path, file_name + figure_name + file_suffix)

numberingFig = numberingFig + 1
figure = plt.figure(numberingFig, figsize = AlvaFigSize)
plot1 = figure.add_subplot(1, 2, 1)
plot1.plot(gInput, randomSeed, color = 'gray', marker = 'o', label = 'data')
plot1.plot(gInput, alva.AlvaMinMax(randomSeed), color = 'red', marker = 'o', label = 'minMaxSorting')
if totalPoint_Input < 100:
    plot1.set_xticks(gInput, minor = True) 
    plot1.set_yticks(randomSeed, minor = True)
    plot1.grid(True, which = 'minor')
else:
    plot1.grid(True, which = 'major')
plt.title(r'$ Binomial \ (total-input = %i,\ mean = %f) $'%(totalPoint_Input, meanI)
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ input $', fontsize = AlvaFontSize)
plt.ylabel(r'$ output $', fontsize = AlvaFontSize)
plt.legend(loc = (0, -0.2))

plot2 = figure.add_subplot(1, 2, 2)
plot2.plot(numberLevel, gLevel, color = 'red', marker = 'o', label = 'category') 
plot2.plot(binomial_D, np.arange(totalLevel), color = 'blue', marker = 'o', label = 'Binomial distribution') 
if totalPoint_Input < 100:
    plot2.set_xticks(numberLevel, minor = True) 
    plot2.set_yticks(gLevel, minor = True)
    plot2.grid(True, which = 'minor')
else:
    plot2.grid(True, which = 'major')
plt.title(r'$ Binomial \ distribution\ (data = %i,\ level = %i) $'%(totalPoint_Input, totalLevel)
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ input/level $', fontsize = AlvaFontSize)
plt.ylabel(r'$ output-level $', fontsize = AlvaFontSize)
plt.legend(loc = (0, -0.2))

figure.tight_layout()
plt.savefig(save_figure, dpi = 300)
plt.show()

# <codecell>


