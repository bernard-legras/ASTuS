# -*- coding: utf-8 -*-
"""

This script processes daily lat alt MLS averages after the Hunga Tonga eruption

Created on 28 May 2022

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr
"""

import mapmls
from datetime import datetime,timedelta
import numpy as np
import matplotlib.pyplot as plt
from zISA import zISA
import gzip, pickle

nlev = 19
date1 = datetime(2022,1,27)
date2 = datetime(2022,6,6)
date = date1

combinat = {}
combinat['data'] = {}

show = False

while date <= date2:
    res = mapmls.dailyLatprofMLS(date,'H2O',bypass='True')
    zEdges = zISA(100*np.exp(res['LogPressureEdges']))
    if show:
        plt.pcolormesh(res['LatEdges'],zEdges,res['H2O']*1.e6,vmax=25,cmap='gist_ncar')
        plt.colorbar()
        plt.ylim(20,30)
        plt.xlim(-35,20)
        plt.show()
    combinat['data'][date] = {'meanWP':res['H2O'],'npix':res['count']}
    date += timedelta(days=1)

combinat['attr'] = {'lats':res['Latitude'],'press':100*res['Pressure'],
                    'lats_edge':res['LatEdges'],
                    'logPress_edge':res['LogPressureEdges']+np.log(100)}

with gzip.open('combinat-MLS-daily.pkl','wb') as f:
    pickle.dump(combinat,f,protocol=pickle.HIGHEST_PROTOCOL)
