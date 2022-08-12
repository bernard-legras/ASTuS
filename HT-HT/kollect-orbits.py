#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploring the calipso database for orbits that can interesect the latitude band where the Hunga Tonga
plume is expected
Version for product 3.41

This script generates a catalog of all available CALIOP orbits during a time interval.
It does not open the files and thus runs very fast. All the information is extracted from
the names of the files. Due to the speed no update mode is necessary

The katalog is organized by dates. It does not contain any ordered lits of the files.
For each date, it contains two sublists for day and night (day & nit).

The segment is defined by the variable name_sel and the period of processing is in date0 and date1
date0 is fixed for each segment. date1 is open for Nsel3 as long as the cloud can be followed with CALIOP.

This script must be run on ICARE

It produces a file 'katalogOrbits-'+name_sel+'.pkl' which is an entry point for subsequant scripts 
(list_missing_orbits.py and korrect_catalog_orbits.py)

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr
"""
import numpy as np
from datetime import datetime,timedelta
import pickle,gzip
#import matplotlib.pyplot as plt
import os
import glob

# rootdir and main directories for aerosol profiles and L1 data
rootdir = '/home/b.legras/STC/HT-HT'
dirL1 = '/DATA/LIENS/CALIOP/CAL_LID_L1.v3.41'

update = True
no0shift = False

katalog = {}
date0 = datetime(2022,1,26)
date1 = datetime(2022,3,27)
name_sel = 'Nsel2'
date0 = datetime(2022,3,31)
date1 = datetime(2022,7,11)
name_sel = 'Nsel3'
#date0 = datetime(2022,1,15)
#date1 = datetime(2022,1,19)
#name_sel = 'Nsel0'

date = date0
while date <= date1:
    # Generate names of daily directories
    dirdayL1 = os.path.join(dirL1,date.strftime('%Y/%Y_%m_%d'))
    print(dirdayL1)
    # List the content of the daily aeorosol directory
    try:
        fic = sorted(glob.glob(dirdayL1+'/CAL_LID_L1-ValStage1-V3-41.*.hdf'))
        print(len(fic))
        fic.reverse()
        katalog[date] = {'day':[],'nit':[]}
        ll = len(fic)
        katalog[date]['number'] = ll
    except:
        print('missing day')
        date += timedelta(days=1)

    for i in range(ll):
        # pop file
        fileL1 = fic.pop()
        print(fileL1)
        fname = os.path.basename(fileL1)
        rootname = fname[27:-4]

        if 'ZD' in fileL1: katalog[date]['day'].append(rootname)
        if 'ZN' in fileL1: katalog[date]['nit'].append(rootname)

    date += timedelta(days=1)

with gzip.open('katalogOrbits-'+name_sel+'.pkl','wb') as f:
    pickle.dump(katalog,f)
