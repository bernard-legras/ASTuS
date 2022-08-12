# -*- coding: utf-8 -*-
"""
Produce lat,lon list of reduced points (1000 values) for selected orbits, 
to be used in producing charts figuring the orbits. 
 
Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr

"""
import pickle,gzip
from datetime import datetime
from pyhdf.SD import SD, SDC
import os
import numpy as np
import socket

# rootdir and main directories for aerosol profiles and L1 data
if 'gort' == socket.gethostname():
    rootdir = '/dkol/data/STC/ST-HungaTonga'
    dirL1 = '/dkol/data/CALIOP/CAL_LID_L1.v3.41'
elif 'satie' in socket.gethostname():
    rootdir = '/data/STC/ST-HungaTonga'
    dirL1 = '/data/CALIOP/CAL_LID_L1.v3.41'
else:
    rootdir = '/home/b.legras/STC/ST-HungaTonga'
    dirL1 = '/DATA/LIENS/CALIOP/CAL_LID_L1.v3.41'


ND = 'nit'
sel = 3
latmax = 20
latmin = -35

if sel == 2:
    with gzip.open('NCald2.pkl','rb') as f:
        NCald = pickle.load(f)
    listi = range(1,len(NCald[ND])+1)
elif sel == 3:
    with gzip.open('NCald3.pkl','rb') as f:
        NCald = pickle.load(f)
    listi = range(1,len(NCald[ND])+1)
elif sel ==0:
    with gzip.open('NCald0.pkl','rb') as f:
        NCald = pickle.load(f)
    if ND=='nit': listi = [9,24,25,39,40,41,70,71]
    if ND=='day': listi = [17,33,63,64]


# Definition of the catalogue
sektions = {'data':{}}

for i in listi:
#for i in range(94,109):
    print(i)
    sektions['data'][i] = {}
    # if missing, cycle
    if NCald[ND][i]['missing']:
        print(i,' MISSING')
        sektions['data'][i]['missing'] = True
        continue
    sektions['data'][i]['missing'] = False
    # generate date and daily directory name
    day = NCald[ND][i]['date']
    sektions['data'][i]['date'] = datetime.strptime(NCald[ND][i]['fname'][:19],'%Y-%m-%dT%H-%M-%S')
    dirdayL1 = os.path.join(dirL1,day.strftime('%Y/%Y_%m_%d'))
    fileL1 = os.path.join(dirdayL1,'CAL_LID_L1-ValStage1-V3-41.'+NCald[ND][i]['fname']+'.hdf')
    hdf1 = SD(fileL1,SDC.READ)
    lats1 = hdf1.select('Latitude').get().flatten()
    lons1 = hdf1.select('Longitude').get().flatten()
    selL1 = (lats1 <= latmax+5) & (lats1 >= latmin-5)
    if np.all(selL1):
        print(i,' NO PART OF THE ORBIT IN THE LAT RANGE')
        sektions['data'][i]['missing'] = True
        continue
    lats1 = lats1[selL1]
    sektions['data'][i]['lats'] = lats1[0:len(lats1):1000]
    lons1 = lons1[selL1]
    sektions['data'][i]['lons'] = lons1[0:len(lons1):1000]
    
with gzip.open(os.path.join('.','sektions_caliop.'+str(sel)+'_'+ND+'.pkl'),'wb') as f:
        pickle.dump(sektions,f,protocol=pickle.HIGHEST_PROTOCOL)
