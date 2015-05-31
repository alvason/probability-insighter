# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Probability-insighter
# https://github.com/alvason/probability-insighter
# 
# Gaussian random distribution (standard normal distribution)

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
file_name = 'gaussian-distribution'

import alva_machinery_probability as alva

AlvaFontSize = 23
AlvaFigSize = (16, 7)
numberingFig = 0

# <codecell>

'''Gaussian randomness --- Gaussian distribution --- Standard normal distribution'''

totalPoint_Input = int(1000)
gInput = np.arange(totalPoint_Input)
randomSeed = np.random.standard_normal(totalPoint_Input)

sumG = 0
for i in range(totalPoint_Input):
    sumG = sumG + randomSeed[i]
meanG = sumG/(totalPoint_Input)

totalLevel = int(totalPoint_Input/1)
category = alva.AlvaLevel(randomSeed, totalLevel, False)
gLevel = category[0]
numberLevel = category[1]
print category[2].shape

# plotting
figure_name = ''
file_suffix = '.png'
save_figure = os.path.join(dir_path, file_name + figure_name + file_suffix)

numberingFig = numberingFig + 1
figure = plt.figure(numberingFig, figsize = AlvaFigSize)
plot1 = figure.add_subplot(1, 2, 1)
plot1.plot(gInput, randomSeed, color = 'gray', marker = 'o', label = 'data')
plot1.plot(gInput, alva.AlvaMinMax(randomSeed), color = 'red', marker = 'o', label = 'minMaxListing')
if totalPoint_Input < 100:
    plot1.set_xticks(gInput, minor = True) 
    plot1.set_yticks(randomSeed, minor = True)
    plot1.grid(True, which = 'minor')
else:
    plot1.grid(True, which = 'major')
plt.title(r'$ Gaussian \ (total-input = %i,\ mean = %f) $'%(totalPoint_Input, meanG)
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ input $', fontsize = AlvaFontSize)
plt.ylabel(r'$ output $', fontsize = AlvaFontSize)
plt.legend(loc = (0, -0.2))

plot2 = figure.add_subplot(1, 2, 2)
plot2.plot(numberLevel, gLevel, color = 'red', marker = 'o', label = 'category')
plot2.plot(np.exp(-gLevel**2)*alva.AlvaMinMax(numberLevel)[-1], gLevel, color = 'blue', marker = 'o', label = 'Gaussian') 
if totalPoint_Input < 100:
    plot2.set_xticks(numberLevel, minor = True) 
    plot2.set_yticks(gLevel, minor = True)
    plot2.grid(True, which = 'minor')
else:
    plot2.grid(True, which = 'major')
plt.title(r'$ Gaussian \ distribution\ (data = %i,\ level = %i) $'%(totalPoint_Input, totalLevel)
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ Number/level $', fontsize = AlvaFontSize)
plt.ylabel(r'$ Output-level $', fontsize = AlvaFontSize)
plt.legend(loc = (0, -0.2))

figure.tight_layout()
plt.savefig(save_figure, dpi = 300)
plt.show()

# <codecell>

def AlvaIntegrateArea(out_i, min_i, max_i, totalGPoint_i):
    spacing_i = np.linspace(min_i, max_i, num = totalGPoint_i, retstep = True)
    grid_i = spacing_i[0]
    dx = spacing_i[1]
    outArea = np.sum(out_i(grid_i[:]))*dx
    return (outArea)

def gaussianA(i):
    inOut = np.exp(-i**2)
    return (inOut)

ggg = AlvaIntegrateArea(gaussianA, -10, 10, 100)
print ggg
ppp = (np.pi)**(1.0/2)
print ppp

# <codecell>


