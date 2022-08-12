#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script is doing a mosaic of 4 images that makes
the upper half of fig 3 of the HT plume paper

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
import numpy as np
#import matplotlib.colors as colors
import geosat
#from cartopy import feature
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.crs as ccrs


seq = '24h'
wlon = 90
elon = 200
nlat = -5
slat = -37.5
gg = geosat.GeoGrid('HimFull')

dates = (datetime(2022,1,15,16),datetime(2022,1,16,0),
         datetime(2022,1,16,8),datetime(2022,1,16,15))
# for local test
#dates = (datetime(2022,1,20,11),datetime(2022,1,20,11),
#         datetime(2022,1,20,11),datetime(2022,1,20,11))

#CALIOP trace for 16 Jan, 15_08-09ZN
lons=np.array([170.66814, 169.94249, 169.23776, 168.55098, 167.87825, 167.21664,
       166.56306, 165.91574, 165.271  , 164.62738, 163.98203, 163.33275,
       162.67711, 162.01205, 161.33488, 160.6428 , 159.93211, 159.19855,
       158.43779, 157.64447, 156.81155, 155.93214])
lats=np.array([ 24.998375  ,  21.999462  ,  18.996452  ,  15.993812  ,
        12.987367  ,   9.980883  ,   6.9717884 ,   3.9641902 ,
         0.95440024,  -2.053207  ,  -5.0622354 ,  -8.068334  ,
       -11.074525  , -14.077958  , -17.080713  , -20.079268  ,
       -23.075665  , -26.06836   , -29.057878  , -32.041805  ,
       -35.021427  , -37.99574   ])

wlon = 145.
elon = 195.
slat = -30.
nlat = -15.
ext = [wlon,elon,slat,nlat]
subgg = gg.subgrid(ext)

fig, ax = plt.subplots(nrows=2, ncols=2,figsize=(15,6),
                         subplot_kw={"projection":ccrs.PlateCarree(central_longitude=180)})
ax = ax.flatten()
fig.subplots_adjust(hspace=0,wspace=0.05,top=0.925,left=0.)

ja = 0
lefts = [True,False,True,False]
bottoms = [False,False,True,True]
for date in dates:
    print(date)
    ah = geosat.Himawari(date)
    print('opened image')
    ah._mk_Ash()
    print('ash  processed')
    ah.close()
    ph = geosat.SatGrid(ah,gg)
    ph._sat_togrid('Ash')
    ax[ja] = ph.chart('Ash',txt=date.strftime('%Y-%m-%d %H:%M UTC'),
             axf=ax[ja],subgrid=subgg,show=False,cm_lon=180,
             left=lefts[ja],bottom=bottoms[ja])
    if ja==3:
        ax[ja].plot(lons-180,lats,'k')
        ax[ja].set_ylim((slat,nlat))
    print('chart processed')
    ja += 1

ax[0].annotate('a)',(-0.03,1.04),xycoords='axes fraction',fontsize=16)
ax[1].annotate('b)',(-0.03,1.04),xycoords='axes fraction',fontsize=16)
ax[2].annotate('c)',(-0.03,1.04),xycoords='axes fraction',fontsize=16)
ax[3].annotate('d)',(-0.03,1.04),xycoords='axes fraction',fontsize=16)

plt.savefig('RGB-fig3.png',dpi=300,bbox_inches='tight')