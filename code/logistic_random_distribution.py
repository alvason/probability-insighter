# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Probability-insighter
# https://github.com/alvason/probability-insighter
# 
# ### Random distribution --- Logistic distribution

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
file_name = 'logistic-distribution'

import alva_machinery_probability as alva

AlvaFontSize = 23
AlvaFigSize = (16, 7)
numberingFig = 0

'''Logistic distribution'''
figure_name = '-equation'
file_suffix = '.png'
save_figure = os.path.join(dir_path, file_name + figure_name + file_suffix)

numberingFig = numberingFig + 1
plt.figure(numberingFig, figsize=(9, 6))
plt.axis('off')
plt.title(r'$ Logistic-distribution---probability-mass-function $',fontsize = AlvaFontSize)
plt.text(0, 5.0/6, r'$ P_{b}(n|N) = \frac{N!}{n!(N - n)!} p^n (1 - p)^{N - n} $', fontsize = 1.2*AlvaFontSize)
plt.text(0, 4.0/6, r'$ 1-- \ a \ set \ of \ N \ events $', fontsize = AlvaFontSize)
plt.text(0, 3.0/6, r'$ 2-- \ either \ Success \ or \ Failure \ for \ each \ event $', fontsize = AlvaFontSize)
plt.text(0, 2.0/6, r'$ 3-- \ the \ probability-p \ is \ the \ same \ for \ all \ events $', fontsize = AlvaFontSize)
plt.text(0, 1.0/6, r'$ 4-- \ P(n|N) \ is \ the \ probability \ with \ n-success \ events \ in \ a \ total \ of \ N \ events $',
         fontsize = AlvaFontSize)
plt.savefig(save_figure, dpi = 300)
plt.show()

# <codecell>

'''Logistic randomness --- Logistic distribution'''
totalPoint_Input = int(100 + 1)
gInput = np.arange(totalPoint_Input)
meanL = totalPoint_Input/2

randomSeed_normal = np.random.standard_normal(totalPoint_Input)
randomSeed = np.random.logistic(0, 3, totalPoint_Input)

totalLevel = int(totalPoint_Input/1)
category = alva.AlvaLevel(randomSeed, totalLevel, False)
gLevel = category[0]
numberLevel = category[1]
print category[2].shape

# calculating the mean
sumL = 0
for i in range(totalPoint_Input):
    sumL = sumL + randomSeed[i]
current_mean = sumL/(totalPoint_Input)
print ('current mean', current_mean)

totalLevel = int(totalPoint_Input/1)
category_normal = alva.AlvaLevel(randomSeed_normal, totalLevel, False)
gLevel_normal = category_normal[0]
numberLevel_normal = category_normal[1]

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
plot1.plot(gInput, alva.AlvaMinMax(randomSeed_normal), label = 'minMax_normal')
if totalPoint_Input < 100:
    plot1.set_xticks(gInput, minor = True) 
    plot1.set_yticks(randomSeed, minor = True)
    plot1.grid(True, which = 'minor')
else:
    plot1.grid(True, which = 'major')
plt.title(r'$ Logistic (total-input = %i,\ mean = %f) $'%(totalPoint_Input, meanL)
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ input-time $', fontsize = AlvaFontSize)
plt.ylabel(r'$ output $', fontsize = AlvaFontSize)
plt.legend(loc = (0, -0.2))

plot2 = figure.add_subplot(1, 2, 2)
plot2.plot(numberLevel_normal, gLevel_normal,  label = 'category_normal')
plot2.plot(numberLevel, gLevel, color = 'red', marker = 'o', label = 'category')
if totalPoint_Input < 100:
    plot2.set_xticks(numberLevel, minor = True) 
    plot2.set_yticks(gLevel, minor = True)
    plot2.grid(True, which = 'minor')
else:
    plot2.grid(True, which = 'major')
plt.title(r'$ Logistic \ distribution\ (data = %i,\ level = %i) $'%(totalPoint_Input, totalLevel)
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ Number/level $', fontsize = AlvaFontSize)
plt.ylabel(r'$ Output-level $', fontsize = AlvaFontSize)
plt.legend(loc = (0, -0.2))

figure.tight_layout()
plt.savefig(save_figure, dpi = 300)
plt.show()

# <codecell>


