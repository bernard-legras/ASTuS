#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script exploits the catalog created by kollect_orbits.py to find the location of missing orbits

A better procedure would be to use the source catalog of LARC and compare with the catalog

Command to get the list of all files available at LARC for a given month and compare
with ICARE
There might be a more direct usage of opendap within python
lynx -dump -listonly https://opendap.larc.nasa.gov/opendap/CALIPSO/LID_L1-ValStage1-V3-41/2022/03/contents.html | grep -e 'hdf$' | grep -v viewer | awk '{print $2}'

The script writes two output files 
'Caliop-day-OrbitList-Nsel'+str(Nsel)+'.txt' & 'Caliop-nit-OrbitList-Nsel'+str(Nsel)+'.txt'

A good practice is to rename the output of the previous run with a '-bak' extension before the new run
and to compare with the output of the new version to check whether some missing orbits have been restored
in the mean time. Such restoration might imply a partial rerun of plotting and averaging routines t o account for this new data.

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr
"""
import numpy as np
from datetime import datetime,timedelta
import pickle,gzip

Nsel = 3

with gzip.open('katalogOrbits-Nsel'+str(Nsel)+'.pkl','rb') as f:
    katalog = pickle.load(f)

fulldates = {'day':[],'nit':[]}
fnames = {'day':[],'nit':[]}

for date in katalog:
    print(date)
    for ND in ['day','nit']:
        for fname in katalog[date][ND]:
            fnames[ND].append(fname)
            fulldates[ND].append(datetime.strptime(fname[:-2],'%Y-%m-%dT%H-%M-%S'))

deltas = {}
for ND in ['day','nit']:
    deltas[ND] = np.array([(fulldates[ND][i+1] - fulldates[ND][i]).total_seconds() for i in range(len(fulldates[ND])-1)])
delta_int = {'day':deltas['day'].min(),'nit':deltas['nit'].min()}

skip = 0
for ND in ['day','nit']:
    day = fulldates[ND][0].date()
    fo=open('Caliop-'+ND+'-OrbitList-Nsel'+str(Nsel)+'.txt','w')
    print(day)
    fo.write(day.strftime('%Y %m %d\n'))
    j = 1
    ij = 1
    fo.write('{:2d} {:3d}  '.format(j,skip+ij)+fnames[ND][0]+fulldates[ND][0].time().strftime('  %H:%M\n'))
    for i in range(len(deltas[ND])):
        if fulldates[ND][i+1].date() > day:
            day = fulldates[ND][i+1].date()
            print(day)
            fo.write(day.strftime('%Y %m %d\n'))
            j = 0
        if deltas[ND][i] > delta_int[ND] + 1000:
            nb_miss = int(deltas[ND][i] / delta_int[ND] + 0.5)
            for nm in range(nb_miss-1):
                j += 1
                ij += 1
                fo.write('{:2d}   MISSING ORBIT\n'.format(j))
        j += 1
        ij += 1
        fo.write('{:2d} {:3d}  '.format(j,skip+ij)+fnames[ND][i+1]+fulldates[ND][i+1].time().strftime('  %H:%M\n'))
    fo.close()
