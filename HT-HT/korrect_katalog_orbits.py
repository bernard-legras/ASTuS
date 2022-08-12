#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script corrects the catalog from ICARE by inserting the missing orbits at their required location
flagged to be missing (hopefully temporarily)
These orbits are those which are detected as missing by list_missing_orbits.py
with an approximate time taking into account the mean interval of 1h38 between two 
orbits.

In addition, the no_exploit list contains orbits that do exist but do not contain 
any exploitable information in th erequired band of latitude. Such orbits can be detected
by browsing the LARC site or by attempting to process them, thus generating errors.

The list has been created manually from the missing_orbits.txt and by browsing the LARC catalog
For a more systematic usage, the process should be made automatic by a comparison with LARC list.

This piece of code comes after list_missing_orbits.py and assessing new missing 
orbits.

The parameter is sel that takes values 0, 2 or 3.
A better way to input parameters and missing orbit list would be to use a json file.

This script produces a dictionary Nkatalog indexed by day (date). Each day contains 
two items for 'day' and 'nit'. Each item contains variables which are themselves 
lists corresponding to all orbits of that day, including those missing. 
The variables are 'fname' for the name of the orbit, 'missing' which is True 
or False  and 'number' which is the number or orbits for that day.

This is meant to be exploited to generate daily zonal averages.

The script must be run after kollect-orbits.py which generates the katalogOrbits 
file read as input. 

A second dictionary Ncald is produced that contains the orbits as a list indexed 
from 1 to the number of orbits for 'nit' and 'day' separately.
Each item contains 'fname' and 'missing'.

This is meant to be exploited by scripts that process each orbit.

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr
"""
#import numpy as np
from datetime import datetime,date
import pickle,gzip

sel = 3
name_sel = 'Nsel'+str(sel)

with gzip.open('katalogOrbits-'+name_sel+'.pkl','rb') as f:
    katalog = pickle.load(f)

# List of missing orbits between 26 January and 26 March
if sel==2: missing = {'day':[],'nit':[]}
# List of missing orbits between 31 March and ...
# These orbits are also missing at LARC and their timing is only approximate
elif sel==3: missing = {'day':['2022-04-01T22-30-00ZD','2022-04-06T00-30-00ZD','2022-04-14T22-30-00ZD',
                               '2022-05-15T03-30-00ZD','2022-05-15T05-10-00ZD','2022-05-15T06-50-00ZD',
                               '2022-06-05T00-10-00ZD','2022-06-21T23-30-00ZD','2022-06-22T01-00-00ZD'],
                        'nit':['2022-04-01T23-00-00ZN','2022-04-14T22-45-00ZN','2022-04-15T00-30-00ZN',
                               '2022-05-15T02-45-00ZN','2022-05-15T04-30-00ZN','2022-05-15T06-05-00ZN',
                               '2022-06-04T23-30-00ZN','2022-06-21T22-40-00ZN','2022-06-22T00-00-00ZN']}
elif sel==0: missing = {'day':['2022-01-18T01-00-00ZD','2022-01-18T02-30-00ZD','2022-01-18T04-00-00ZD',
                               '2022-01-18T05-30-00ZD','2022-01-18T07-00-00ZD','2022-01-18T08-30-00ZD',
                               '2022-01-18T10-00-00ZD','2022-01-18T11-30-00ZD','2022-01-18T13-00-00ZD',
                               '2022-01-18T14-30-00ZD','2022-01-18T16-00-00ZD','2022-01-18T17-30-00ZD',
                               '2022-01-18T20-00-00ZD','2022-01-18T21-30-00ZD'],
                        'nit':['2022-01-18T01-00-00ZN','2022-01-18T02-30-00ZN','2022-01-18T04-00-00ZN',
                               '2022-01-18T05-30-00ZN','2022-01-18T07-00-00ZN','2022-01-18T08-30-00ZN',
                               '2022-01-18T10-00-00ZN','2022-01-18T11-30-00ZN','2022-01-18T13-00-00ZN',
                               '2022-01-18T14-30-00ZN','2022-01-18T16-00-00ZN','2022-01-18T17-30-00ZN',
                               '2022-01-18T20-00-00ZN','2022-01-18T21-30-00ZN']}
# Nsel2 : Four orbits that are present but contain no exploitable data
# The last two for sel 2 are files which are truncated only on ICARE (to be checked again)
if sel ==2:
	no_exploit = ['2022-02-11T11-46-05ZN','2022-03-25T15-09-53ZN',
                      '2022-03-25T04-26-11ZD','2022-03-24T19-27-50ZN']
elif sel ==3:
       no_exploit = ['2022-04-15T13-33-48ZN','2022-04-16T01-03-20ZN',
                     '2022-04-16T02-41-50ZN','2022-04-16T04-20-20ZN',
                     '2022-04-16T05-58-50ZN','2022-04-18T08-52-32ZN',
                     '2022-04-16T00-10-35ZD','2022-04-16T01-49-05ZD',
                     '2022-04-16T03-27-40ZD','2022-04-16T05-06-10ZD',
                     '2022-06-17T15-09-14ZN','2022-06-24T16-14-05ZN']
elif sel == 0: no_exploit = [] 

# Create New catalog
Nkatalog = {}
for datet in katalog:
    Nkatalog[datet.date()] = {}
    for ND in ['day','nit']:
        Nkatalog[datet.date()][ND] = {}
        Nkatalog[datet.date()][ND]['fname'] = katalog[datet][ND]
        Nkatalog[datet.date()][ND]['missing'] = [False for i in range(len(katalog[datet][ND]))]
    Nkatalog[datet.date()]['number'] = katalog[datet]['number']

# Fill the missing orbits in the new catalog
for ND in ['day','nit']:
    for fmiss in missing[ND]:
        fdat = datetime.strptime(fmiss[:-2],'%Y-%m-%dT%H-%M-%S')
        day = fdat.date()
        j = 0
        for fname in Nkatalog[day][ND]['fname']:
            fulldate = datetime.strptime(fname[:-2],'%Y-%m-%dT%H-%M-%S')
            if fulldate < fdat:
                j += 1
                continue
            else:
                break
        Nkatalog[day][ND]['fname'].insert(j,fmiss)
        Nkatalog[day][ND]['missing'].insert(j,True)
        Nkatalog[day]['number'] += 1

# Flagging orbits with no exploitable data as missing
for fname in no_exploit:
    print('no_exploit',fname)
    if 'ZN' in fname: ND = 'nit'
    else: ND = 'day'
    fdat = datetime.strptime(fname[:-2],'%Y-%m-%dT%H-%M-%S')
    day = fdat.date()
    idx = Nkatalog[day][ND]['fname'].index(fname)
    Nkatalog[day][ND]['missing'][idx] = True

# Create the new Cald for sel 2/3
NCald = {'day':{},'nit':{}}
for ND in ['day','nit']:
    i = 0
    for day in Nkatalog:
        Nkatalog[day][ND]['idx'] = []
        ll = len(Nkatalog[day][ND]['fname'])
        for j in range(ll):
            i += 1
            NCald[ND][i] = {'date':day,'j':j,'fname':Nkatalog[day][ND]['fname'][j],
                           'missing':Nkatalog[day][ND]['missing'][j]}
            Nkatalog[day][ND]['idx'].append(i)

with gzip.open('katalogOrbitsKorrect'+str(sel)+'.pkl','wb') as f:
    pickle.dump(Nkatalog,f)

with gzip.open('NCald'+str(sel)+'.pkl','wb') as f:
    pickle.dump(NCald,f)