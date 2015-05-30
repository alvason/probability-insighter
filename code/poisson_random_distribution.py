# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # Diffusion computation
# https://github.com/alvason/diffusion-computation
# 
# ### Section003 --- Stochastic solution for the diffusion equation
# ##### Random distribution --- Poisson distribution 

# <codecell>

'''
author: Alvason Zhenhua Li
date:   03/19/2015
'''

%matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

import alva_machinery_diffusion as alva

AlvaFontSize = 23
AlvaFigSize = (16, 7)
numberingFig = 0

'''Poisson process --- Poisson distribution'''
numberingFig = numberingFig + 1
plt.figure(numberingFig, figsize=(12, 3))
plt.axis('off')
plt.title(r'$ Poisson-distribution \ equation $',fontsize = AlvaFontSize)
plt.text(0,2.0/3,r'$ P_{p}(n|N) = \frac{N!}{n!(N - n)!} p^n (1 - p)^{N - n} $', fontsize = 1.2*AlvaFontSize)
plt.text(0,1.0/3,r'$ P_{m}(n) = \frac{e^{-m} m^n}{n!}, where \ the \ mean \ (average) \ m \equiv pN  $', fontsize = 1.2*AlvaFontSize)
plt.show()

# <codecell>

def AlvaProduct(i):
    product = 1
    for j in range(1, int(i) + 1):        
        product = product*j
#       print product
    return (product)

def AlvaPoissonProcess(i, N):
    if type(i) == np.ndarray:
        total_Input = np.size(i)
    p = np.zeros(total_Input)
    process = np.zeros(total_Input)
    for xn in range(total_Input):
        p[xn] = 0.5
        process[xn] = AlvaProduct(N)/(AlvaProduct(i[xn])*AlvaProduct(N - i[xn])
                                      ) * p[xn]**i[xn] * (1 - p[xn])**(N - i[xn])
#    print ('probability = ', p)
    return (process)

def AlvaPoissonD(m, i):
    if type(i) == np.ndarray:
        total_Input = np.size(i)
        distribution = np.zeros(total_Input)  
        for xn in range(total_Input):
            distribution[xn] = (m**i[xn])*np.exp(-m)/AlvaProduct(i[xn])
    return (distribution)

N = float(100)
b = 1
mean = N*b/3
rangeN = N*b
print ('mean = ', mean)
aaa =np.arange(1, N)*b
print aaa
plt.figure(figsize = (12, 4))
plt.plot(aaa, AlvaPoissonProcess(aaa, N), marker ='^', color = 'blue', label = 'Process')
plt.plot(aaa, AlvaPoissonD(mean, aaa), marker ='o', color = 'red', label = 'Distribution')
#plt.plot(aaa, np.exp(-((aaa - mean)/rangeN)**2), marker ='+', color = 'red', label = 'Gaussian')
plt.xlabel(r'$ Output-level$', fontsize = AlvaFontSize)
plt.ylabel(r'$ Number/level $', fontsize = AlvaFontSize)
plt.title(r'$ Poisson \ process $', fontsize = AlvaFontSize)
plt.grid(True)
plt.legend(loc = (1, 0))
plt.show()

# <codecell>

'''Poisson randomness --- Poisson distribution'''
totalPoint_Input = int(200 + 1)
gInput = np.arange(totalPoint_Input)
meanP = totalPoint_Input/20
randomSeed_normal = np.random.standard_normal(totalPoint_Input) + meanP
randomSeed = np.random.poisson(meanP, totalPoint_Input)

totalLevel = int(totalPoint_Input/1)
category = alva.AlvaLevel(randomSeed, totalLevel, False)
gLevel = category[0]
numberLevel = category[1]
print category[2].shape

# calculating the mean
sumP = 0
for i in range(totalPoint_Input):
    sumP = sumP + randomSeed[i]
current_mean = sumP/(totalPoint_Input)
print ('current mean', current_mean)

totalLevel = int(totalPoint_Input/1)
category = alva.AlvaLevel(randomSeed, totalLevel, False)
gLevel = category[0]
numberLevel = category[1]
print category[2].shape

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
plt.title(r'$ Poisson\ (total-input = %i,\ mean = %f) $'%(totalPoint_Input, meanP)
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ input $', fontsize = AlvaFontSize)
plt.ylabel(r'$ output $', fontsize = AlvaFontSize)
plt.legend(loc = (0, -0.2))

plot2 = figure.add_subplot(1, 2, 2)
plot2.plot(numberLevel, gLevel, color = 'red', marker = 'o', label = 'category')
plot2.plot(AlvaPoissonD(meanP, gLevel), gLevel, color = 'blue', marker = 'o', label = 'Poisson Distribution') 
plot2.plot(AlvaPoissonProcess(gLevel, 18), gLevel, color = 'green', marker = 'o', label = 'Poisson Process') 
if totalPoint_Input < 100:
    plot2.set_xticks(numberLevel, minor = True) 
    plot2.set_yticks(gLevel, minor = True)
    plot2.grid(True, which = 'minor')
else:
    plot2.grid(True, which = 'major')
plt.title(r'$ Poisson \ distribution\ (data = %i,\ level = %i) $'%(totalPoint_Input, totalLevel)
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ Number/level $', fontsize = AlvaFontSize)
plt.ylabel(r'$ Output-level $', fontsize = AlvaFontSize)
plt.legend(loc = (0, -0.2))

figure.tight_layout()
plt.show()

# <codecell>

i = 100
print ('Alva = ', AlvaProduct(i))
print ('NumP = ', np.prod(np.arange(1, i + 1), dtype=np.float64))

# <codecell>

''' Gaussian Distribution '''
unitD = 1
rangeD = float(100)*unitD
meanD = rangeD/2
varianceD = rangeD
print ('mean = ', mean)
aaa =np.arange(1, rangeD)*b
print aaa
plt.figure(figsize = (12, 4))
#plt.plot(aaa, AlvaPoissonProcess(aaa, N), marker ='^', color = 'blue', label = 'Process')
#plt.plot(aaa, AlvaPoissonD(mean, aaa), marker ='o', color = 'red', label = 'Distribution')
plt.plot(aaa, np.exp(-((aaa - meanD)/varianceD)**2), marker ='+', color = 'red', label = 'Gaussian')
plt.plot(aaa, np.exp(-((aaa - meanD)**2/varianceD)), marker ='+', color = 'blue', label = 'Gaussian')
plt.xlabel(r'$ Output-level$', fontsize = AlvaFontSize)
plt.ylabel(r'$ Number/level $', fontsize = AlvaFontSize)
plt.title(r'$ Poisson \ process $', fontsize = AlvaFontSize)
plt.grid(True)
plt.legend(loc = (1, 0))
plt.show()

# <codecell>


