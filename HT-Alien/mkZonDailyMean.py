# -*- coding: utf-8 -*-
"""
Produce zonal daily means of a number of variables from ERA5 archive.
To be run on a computer hosting this archive and where the package pylib/ECMWF_N is installed.
This script is using EN, DI and VO3 data.
The potentail temperature PT, geopotential Z and the Lait PV are calculated.
The LAit PV is here defined as PV / PT**4. It can be dimensionalized by muliplying to a proper PT0**4 in the applications.

The calculation is made between the two dates provided by day and day1 and the resulting data are stored in a file with name constructed from dd.

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr

"""
from ECMWF_N import ECMWF
from datetime import datetime, timedelta
import numpy as np
import pickle,gzip
import argparse

# Input parameters
parser = argparse.ArgumentParser()
parser.add_argument("-y","--year",type=int,help="year")
parser.add_argument("-m1","--month1",type=int,choices=1+np.arange(12),help="month")
parser.add_argument("-d1","--day1",type=int,choices=1+np.arange(31),help="day")
parser.add_argument("-m2","--month2",type=int,choices=1+np.arange(12),help="month")
parser.add_argument("-d2","--day2",type=int,choices=1+np.arange(31),help="day")

# default parameters
year = 2022
m1 = 6
d1 = 6
#highlev = 15
#lowlev = 73
listvar = ['T','U','PT','W','Z','LPV','ASSWR','ASLWR']

# setting parameters from input
args = parser.parse_args()
if args.year is not None: year=args.year
if args.month1 is not None: m1=args.month1
if args.day1 is not None: d1=args.day1
if args.month2 is not None: m2=args.month2
else: m2 = m1
if args.day2 is not None: d2=args.day2
else: d2 = d1

day1 = datetime(year,m1,d1)
day2 = datetime(year,m2,d2)
day = day1
zonMean = {}
zonMean['mean'] = {}
zonMean['var'] = {}
while day <= day2:
    zonMean['mean'][day.date] = {}
    zonMean['var'][day.date] = {}
    for hours in range(0,22,3):
        ddd = day + timedelta(hours=hours)
        dat = ECMWF('FULL-EA',ddd,exp=['VOZ','DI'])
        dat._get_T()
        dat._get_var('VO')
        dat._mkpscale()
        dat._mkzscale()
        dat._mkp()
        dat._mkz()
        dat._get_U()
        dat._get_V()
        dat._get_var('W')
        dat._mkthet()
        dat._mkpv()
        dat.var['LPV'] = dat.var['PV']/dat.var['PT']**4
        dat._get_var('ASSWR')
        dat._get_var('ASLWR')
        dat.close()
        datr = dat.extract(latRange=(-35,20.5),varss=listvar)
        for var in listvar:
            if hours == 0:
                zonMean['mean'][day.date][var] = np.mean(datr.var[var],axis=2)
                zonMean['var'][day.date][var] = np.var(datr.var[var],axis=2)
            else:
                zonMean['mean'][day.date][var] += np.mean(datr.var[var],axis=2)
                zonMean['var'][day.date][var] += np.var(datr.var[var],axis=2)
    for var in listvar:
        zonMean['mean'][day.date][var] /= 8
        zonMean['var'][day.date][var] /= 8
    print(day.strftime('Completed %d %B'))
    day += timedelta(days=1)

zonMean['attr'] = datr.attr
with gzip.open(dd.strftime('zonDailyMean-%b-d%d.pkl'),'wb') as f:
    pickle.dump(zonMean,f,protocol=pickle.HIGHEST_PROTOCOL)