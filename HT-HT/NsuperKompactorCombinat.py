# -*- coding: utf-8 -*-
"""
Makes daily averages from superCatal_caliop using the indexing 
from katalogOrbitsKorrect

To be run on any machine. Intermediate step between NsuperKompactor.py
and the analysis notebook for zonal averages

This is to be runned for each exploited sel

At the end, a piece of code to merge the two files produced by sel 2 and sel 3

Created: 01/06/2022

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr

"""
import pickle,gzip
import os
from datetime import date
import numpy as np

Qs = 5.167*1.e-31
kbw = 1.0313

# rootdir and main directories for aerosol profiles and L1 data

[alts,alts_edges] = pickle.load(open('L1_alts_edges.pkl','rb'))

ND = 'nit'
sel = 3

with gzip.open('katalogOrbitsKorrect'+str(sel)+'.pkl','rb') as f:
    Nkatalog = pickle.load(f)

with gzip.open(os.path.join('.','superCatal_caliop.'+str(sel)+'_'+ND+'.pkl'),'rb') as f:
    catal = pickle.load(f)

combinat = {}
combinat['data'] = {}

#listi = range(1,len(NCald[ND])+1)
#listi = range(692,len(NCald[ND])+1)

lvar1 = ['T532','SR532','DEPOL','ACR']
lvarAll =  ['T532','SR532','DEPOL','ACR','INT532','TROPOH','P','T']

for day in Nkatalog.keys():
    if day < date(2022,5,25): continue
    # generate accumulators (15001,32-125)
    print('processing',day)
    combinat['data'][day] = {}

    prov = {}
    for var in lvar1:
        prov[var] = np.ma.zeros((15,183,93))
    prov['INT532'] = np.ma.zeros((15,183))
    prov['TROPOH'] = np.ma.zeros((15,183))
    prov['P'] = np.ma.zeros((15,183,8))
    prov['T'] = np.ma.zeros((15,183,8))
    no = 0
    for j in range(len(Nkatalog[day][ND]['idx'])):
        idx = Nkatalog[day][ND]['idx'][j]
        if Nkatalog[day][ND]['missing'][j]:
            print ('Missing orbit',day,j,idx)
            continue
        for var in lvarAll:
            prov[var][no,...] =  catal['data'][idx][var]
        no += 1
    
    for var in lvarAll:
        combinat['data'][day][var] = np.ma.mean(prov[var][:no,...],axis=0)
        
combinat['attr'] = catal['attr']

#with gzip.open(os.path.join('.','superCombi_caliop.'+str(sel)+'_'+ND+'-s10.pkl'),'wb') as f:
#        pickle.dump(combinat,f,protocol=pickle.HIGHEST_PROTOCOL)

# %% merge operation
# To make it activate following switch
merge = True
if merge:
    with gzip.open(os.path.join('.','superCombi_caliop.2_nit.pkl'),'rb') as f:
        combi2 = pickle.load(f)
    with gzip.open(os.path.join('.','superCombi_caliop.3_nit.pkl'),'rb') as f:
        combi3 = pickle.load(f)
    for day in combi3['data']:
        combi2['data'][day] = combi3['data'][day]
    for day in combinat['data']:
        combi2['data'][day] = combinat['data'][day]
    with gzip.open(os.path.join('.','superCombi_caliop.all_nit.pkl'),'wb') as f:
        pickle.dump(combi2,f,protocol=pickle.HIGHEST_PROTOCOL)
