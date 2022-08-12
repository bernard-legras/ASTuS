# -*- coding: utf-8 -*-
"""
Merge the files produced by mkZonDailyMean.py into a single history
The list of segments is contained in the segments list 
The resulting file is 'zonDailyMean-all.pkl'

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr

"""
from datetime import date
import pickle,gzip
zonMean = {}
zonMean['mean'] = {}
zonMean['var'] = {}
listvar = ['T','U','PT','W','Z','LPV','ASSWR','ASLWR']

segments = ['Jan-1','Jan-2','Feb','Mar-1','Mar-2','Apr-a1','Apr-a2','Apr-a3','Apr-a4','Apr-a5',
            'May-a1','May-a2','May-a3','May-a4','May-a5','May-d16','May-d17','May-d18','May-d19',
            'May-d20','May-d21','May-d22','May-d23','May-d24','May-d25','May-d26','May-d27',
            'May-d28','Jun-d1','Jun-d5']

for seg in segments:
    print('processsing '+seg)
    with gzip.open('zonDailyMean-'+seg+'.pkl','rb') as f:
        zz = pickle.load(f)
    days = list(zz['mean'].keys())
    for day in days:
        print(day())
        zonMean['mean'][day()] = zz['mean'][day]
        zonMean['var'][day()] = zz['var'][day]

zonMean['attr'] = zz['attr']
with gzip.open('zonDailyMean-all.pkl','wb') as f:
    pickle.dump(zonMean,f,protocol=pickle.HIGHEST_PROTOCOL)
