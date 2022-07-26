#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr

Generates the list of equator crossings from the orbits of Metop-B established by IXION

Created on Fri Mar 25 23:27:36 2022

@author: Bernard Legras
"""
import numpy as np
from datetime import datetime,timedelta
import pickle,gzip
import matplotlib.pyplot as plt
from os.path import join
import codecs

def decode(x):
    aa = x.split('|')
    date = datetime.strptime(aa[0],' %Y %m %d %H:%M:%S  ')
    [lon, _, lat, _, _] = np.genfromtxt(aa[1].split())
    return [date,[lon,lat]]

sel = 3
rr = {0:8723-76-10,1:92230-76-10,2:172883-76-10,3:162803-76-10}
file = 'Metop-B_orbits-'+str(sel)+'.txt'

with codecs.open(file, encoding='ISO-8859-1') as f:
#    for line in f:
#        print(repr(line))
    for i in range(76):
        dumb = f.readline()
    x = f.readline()
    [date,s2] = decode(x)
    print(date)
    print(s2)
    lat_a = s2[1]
    day = datetime(date.year,date.month,date.day)
    xings = {'D':{},'N':{}}
    xings['D'][day]= []
    xings['N'][day]= []
    day_a = day
    print(day.strftime('%d %B %Y'))
    for j in range(rr[sel]):
        x = f.readline()
        [date,s2] = decode(x)
        # if no crossing, continue
        if s2[1]*lat_a >0 :
            lat_a = s2[1]
            continue
        # if crossing, process
        if lat_a < 0: ND='N'
        else: ND = 'D'
        day = datetime(date.year,date.month,date.day)
        if day != day_a:
            xings['D'][day]= []
            xings['N'][day]= []
            day_a = day
            print(day.strftime('%d %B %Y'))
        time = date - day
        xings[ND][day].append([time,s2[0]])
        if ND == 'N': print('Ascending  N ',time,'  {:0.1f}'.format(s2[0]))
        else: print('Descending D ',time,'   {:0.1f}'.format(s2[0]))
        lat_a = s2[1]

with gzip.open('Metop-B_xings-'+str(sel)+'.pkl','wb') as f:
    pickle.dump(xings,f)

# %% Merging of the 3 files

# with gzip.open('Metop-B_xings-0.pkl','rb') as f:
#     xings0 = pickle.load(f)
# with gzip.open('Metop-B_xings-2.pkl','rb') as f:
#     xings1 = pickle.load(f)
# for day in xings0['N']:
#     xings['N'][day] = xings0['N'][day]
# for day in xings0['D']:
#     xings['D'][day] = xings0['D'][day]
# for day in xings1['N']:
#     xings['N'][day] = xings1['N'][day]
# for day in xings1['D']:
#     xings['D'][day] = xings1['D'][day]
with gzip.open('Metop-B_xings.pkl','wb') as f:
    pickle.dump(xings,f)
