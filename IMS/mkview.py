#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr

Plot all the SA and SO2 images for a first look

Created on Tue May 31 09:22:38 2022

@author: Bernard Legras
"""
import IMS
from datetime import date, timedelta
from matplotlib import pyplot as plt
import os

day0 = date(2022,1,13)
day1 = date(2022,4,30)
ND = 'N'
day = day0
while day <= day1:
    try:
        dat = IMS.IMS(day)
        if day < date(2022,3,15): vmax= 0.03
        else: vmax = 0.02
        dat.read('SA',ND)
        dat.show('SA_'+ND,vmin=0.002,vmax=vmax,log=True,cmap=IMS.cmap3,show=False)
        plt.savefig(os.path.join('SA_'+ND,day.strftime('HT_IMS_SA_%y-%m-%d_')+ND+'.png'),dpi=144,bbox_inches='tight')
        plt.close('all')
        print(day)
    except:
        print('no data for this date', day)
    day += timedelta(days=1)
