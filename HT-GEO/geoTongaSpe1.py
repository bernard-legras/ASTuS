#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Series of geo plots of the immediate vicinity of the HongaTonga eruption
on 15 January 2022

HongaTonga coordinates:
    20° 33' S
    175° 21' W

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr
"""

#import pickle,gzip
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import numpy as np
#import matplotlib.colors as colors
import geosat

#%% Ash RGB Image
RGBType = 'Ash'
if RGBType == 'Ash':
    clim0 = (243,303)
    clim1 = (-4,5)
    clim2 = (-4,2)
elif RGBType == 'Dust':
    clim0 = (261,289)
    clim1 = (0,15)
    clim2 = (-4,2)

RGB = {}

dd0 = datetime(2022,1,20,16)
# for test
# dd0 = datetime(2022,1,27,15)

am1 = geosat.MSG1(dd0)
ah = geosat.Himawari(dd0)

band = geosat.GeoGrid('LatBand2')
pm1 = geosat.SatGrid(am1,band)
ph = geosat.SatGrid(ah,band)
am1._mk_Ash()
ah._mk_Ash()
am1.close()
ah.close()
pm1._sat_togrid('Ash')
ph._sat_togrid('Ash')
#%%
subgg = band.subgrid([60,130,-30,0])
patchf = pm1.patch(ph,90,'Ash')
patchf.chart('Ash',txt='Ash composit MSG1 + Hima 8 '+dd0.strftime('%Y-%m-%d %H:%M'),subgrid=subgg,show=False)
plt.savefig('RGB-16h-fig4.png',dpi=300,bbox_inches='tight')