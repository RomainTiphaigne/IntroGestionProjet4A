#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 16:32:33 2023

@author: goulm
"""

from mesh import mesh_from_msh
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt, cos, sin
from solver import solve_one_time_step
from myplot import MyPlot
import csv

x = []
y = []
rho = []
not_first = True

with open('surf.csv', newline='') as csvfile:
    rowreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in rowreader:
        
        if not_first:
            not_first = not not_first
        else:
            tab = row[0].split(",")
            x.append(float(tab[0]))
            y.append(float(tab[1]))
            rho.append(float(tab[2]))

x = np.array(x)
y = np.array(y)
rho = np.array(rho)

sum_ex = 0
num_ex = 0

sum_in = 0
num_in = 0

for i in range(len(x)):
    if y[i] > 0 : # extrados (dessus)
        sum_ex += rho[i]
        num_ex += 1
    else : # extrados (dessus)
        sum_in += rho[i]
        num_in += 1
        
extrado = sum_ex / num_ex
intrado = sum_in / num_in

print("extrado : ", extrado, "intrado : ", intrado, "écart", extrado - intrado)
    


