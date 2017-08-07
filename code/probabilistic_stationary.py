
# coding: utf-8

# In[1]:

'''
author: Alvason Zhenhua Li
date:   09/21/2015
'''

get_ipython().magic('matplotlib inline')

import numpy as np
import matplotlib.pyplot as plt
import os
dir_path = '/Users/azl/Desktop/AlvaCode/probability-insighter/figure'
file_name = 'probabilistic-stationary'

import alva_machinery_statistics as alva

AlvaFontSize = 23
AlvaFigSize = (16, 7)
numberingFig = 0


# In[2]:

# probabilistic eigenstate for Sunny_Rainy
sun_rain_relation = np.array([[0.9, 0.1],
                              [0.3, 0.7]])

print (sun_rain_relation)

total_event = int(14)
current_event = np.identity(2)
total_evolution = np.zeros([total_event, 2, 2])

future_event = current_event
for i in range(total_event):
    total_evolution[i] = future_event
    future_event = np.dot(sun_rain_relation, future_event)
print (i, future_event)

gEvent = np.arange(total_event)

# plotting
figure_name = '-equilibrium'
figure_suffix = '.png'
save_figure = os.path.join(dir_path, file_name + figure_name + figure_suffix)

numberingFig = numberingFig + 1
figure = plt.figure(numberingFig, figsize = (18, 6))

# initial Sunny
initial_state = 0
plot1 = figure.add_subplot(1, 2, 1)
plt.plot(gEvent, total_evolution[:, initial_state, 0], marker = 'o', markersize = 20, color = 'red', alpha = 0.6,
         label = 'Sunny')
plt.plot(gEvent, total_evolution[:, initial_state, 1], marker = 'o', markersize = 20, color = 'green', alpha = 0.6,
         label = 'Rainy')
plt.title(r'$ Probabilistic \ Equilibrium $'.format()
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ Day $'.format(), fontsize = AlvaFontSize)
plt.ylabel(r'$ Probability $'.format(), fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.8)
plt.yticks(fontsize = AlvaFontSize*0.8) 
plt.grid(True)
plt.legend(loc = (0, 1))

# initial rainy
initial_state = 1
plot2 = figure.add_subplot(1, 2, 2)
plt.plot(gEvent, total_evolution[:, initial_state, 0], marker = 'o', markersize = 20, color = 'red', alpha = 0.6,
         label = 'Sunny')
plt.plot(gEvent, total_evolution[:, initial_state, 1], marker = 'o', markersize = 20, color = 'green', alpha = 0.6,
         label = 'Rainy')
plt.title(r'$ Probabilistic \ Equilibrium $'.format()
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ Day $'.format(), fontsize = AlvaFontSize)
plt.ylabel(r'$ Probability $'.format(), fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.8)
plt.yticks(fontsize = AlvaFontSize*0.8) 
plt.grid(True)
plt.legend(loc = (0, 1))

figure.tight_layout()
plt.savefig(save_figure, dpi = 100)
plt.show()


# In[3]:

# probabilistic eigenstate for Sunny_Cloudy_Rainy
sun_cloud_rain_relation = np.array([[0.7, 0.2, 0.1],
                                    [0.2, 0.5, 0.3],
                                    [0.1, 0.1, 0.8]])

print (sun_cloud_rain_relation)

total_event = int(14)
current_event = np.identity(3)
total_evolution = np.zeros([total_event, 3, 3])

future_event = current_event
for i in range(total_event):
    total_evolution[i] = future_event
    future_event = np.dot(sun_cloud_rain_relation, future_event)
print (i, future_event)

gEvent = np.arange(total_event)

# plotting
figure_name = '-equilibrium'
figure_suffix = '.png'
save_figure = os.path.join(dir_path, file_name + figure_name + figure_suffix)

numberingFig = numberingFig + 1
figure = plt.figure(numberingFig, figsize = (18, 6))

# initial Sunny
initial_state = 0
plot1 = figure.add_subplot(1, 3, 1)
plt.plot(gEvent, total_evolution[:, initial_state, 0], marker = 'o', markersize = 20, color = 'red', alpha = 0.6,
         label = 'Sunny')
plt.plot(gEvent, total_evolution[:, initial_state, 1], marker = 'o', markersize = 20, color = 'gray', alpha = 0.6,
         label = 'Cloudy')
plt.plot(gEvent, total_evolution[:, initial_state, 2], marker = 'o', markersize = 20, color = 'green', alpha = 0.6,
         label = 'Rainy')
plt.title(r'$ Probabilistic \ Equilibrium $'.format()
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ Day $'.format(), fontsize = AlvaFontSize)
plt.ylabel(r'$ Probability $'.format(), fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.8)
plt.yticks(fontsize = AlvaFontSize*0.8) 
plt.grid(True)
plt.legend(loc = (0.7, 0.8))

# initial Cloudy
initial_state = 1
plot2 = figure.add_subplot(1, 3, 2)
plt.plot(gEvent, total_evolution[:, initial_state, 0], marker = 'o', markersize = 20, color = 'red', alpha = 0.6,
         label = 'Sunny')
plt.plot(gEvent, total_evolution[:, initial_state, 1], marker = 'o', markersize = 20, color = 'gray', alpha = 0.6,
         label = 'Cloudy')
plt.plot(gEvent, total_evolution[:, initial_state, 2], marker = 'o', markersize = 20, color = 'green', alpha = 0.6,
         label = 'Rainy')
plt.title(r'$ Probabilistic \ Equilibrium $'.format()
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ Day $'.format(), fontsize = AlvaFontSize)
plt.ylabel(r'$ Probability $'.format(), fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.8)
plt.yticks(fontsize = AlvaFontSize*0.8) 
plt.grid(True)
plt.legend(loc = (0.7, 0.8))

# initial Cloudy
initial_state = 2
plot3 = figure.add_subplot(1, 3, 3)
plt.plot(gEvent, total_evolution[:, initial_state, 0], marker = 'o', markersize = 20, color = 'red', alpha = 0.6,
         label = 'Sunny')
plt.plot(gEvent, total_evolution[:, initial_state, 1], marker = 'o', markersize = 20, color = 'gray', alpha = 0.6,
         label = 'Cloudy')
plt.plot(gEvent, total_evolution[:, initial_state, 2], marker = 'o', markersize = 20, color = 'green', alpha = 0.6,
         label = 'Rainy')
plt.title(r'$ Probabilistic \ Equilibrium $'.format()
          , fontsize = AlvaFontSize)
plt.xlabel(r'$ Day $'.format(), fontsize = AlvaFontSize)
plt.ylabel(r'$ Probability $'.format(), fontsize = AlvaFontSize)
plt.xticks(fontsize = AlvaFontSize*0.8)
plt.yticks(fontsize = AlvaFontSize*0.8) 
plt.grid(True)
plt.legend(loc = (0.7, 0.8))

figure.tight_layout()
plt.savefig(save_figure, dpi = 100)
plt.show()


# In[ ]:



