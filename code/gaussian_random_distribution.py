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

# plotting
figure_name = '-equation'
file_suffix = '.png'
save_figure = os.path.join(dir_path, file_name + figure_name + file_suffix)

numberingFig = numberingFig + 1
plt.figure(numberingFig, figsize=(9, 6))
plt.axis('off')
plt.title(r'$ Gaussian-distribution---probability-mass-function $',fontsize = AlvaFontSize)
plt.text(0, 5.0/6, r'$ P(d, m|n) = \frac{1}{(2\pi)^{1/2}d} exp[-\frac{1}{2}(\frac{x - m}{d})^2] $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 3.0/6, r'$ 1-- \ m \ is \ the \ mean \ number \ of \ events \ per \ interval \ (time \ or \ space) $', fontsize = AlvaFontSize)
plt.text(0, 2.0/6, r'$ 2-- \ d \ is \ the \ deviation \ with \ n-events \ per \ interval $', fontsize = AlvaFontSize)
plt.text(0, 1.0/6, r'$ 3-- \ P(d, m|n) \ is \ the \ probability \ with \ n-events \ per \ interval $',
         fontsize = AlvaFontSize)
plt.savefig(save_figure, dpi = 300)
plt.show()

# <codecell>

'''Gaussian randomness --- Gaussian distribution --- Standard normal distribution'''

total_event = int(1000)
gInput = np.arange(total_event)
randomSeed = np.random.standard_normal(total_event)

sumP = 0
for i in range(total_event):
    sumP = sumP + randomSeed[i]
meanP = sumP/(total_event)

totalLevel = int(total_event/10)
category = alva.AlvaLevel(randomSeed, totalLevel, False)
gLevel = category[0]
numberLevel = category[1]

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
plt.title(r'$ Gaussian \ randomness \ (mean = {:1.6f}) $'.format(meanP), fontsize = AlvaFontSize)
plt.xlabel(r'$ event-input $', fontsize = AlvaFontSize)
plt.ylabel(r'$ output $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 
plt.legend(loc = (0, -0.2))

plot2 = figure.add_subplot(1, 2, 2)
plot2.plot(numberLevel, gLevel, color = 'red', marker = 'o', label = 'category')
plot2.plot(np.exp(-gLevel**2)*alva.AlvaMinMax(numberLevel)[-1], gLevel, color = 'blue', marker = 'o', label = 'Gaussian') 
if total_event < 100:
    plot2.set_xticks(numberLevel, minor = True) 
    plot2.set_yticks(gLevel, minor = True)
    plot2.grid(True, which = 'minor')
else:
    plot2.grid(True, which = 'major')
plt.title(r'$ Gaussian \ distribution \ (events = {ev:},\ levels = {le:}) $'.format(ev = total_event, le = totalLevel)
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ event/level $', fontsize = AlvaFontSize)
plt.ylabel(r'$ level-range $', fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.6)
plt.yticks(fontsize = AlvaFontSize*0.6) 
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


