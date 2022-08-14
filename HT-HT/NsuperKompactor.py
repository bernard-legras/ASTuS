# -*- coding: utf-8 -*-
"""
This script averages the CALIOP sections for each orbit in a lat x alt view with 
reduced latitude resolution. It is derived from Ncombinat.py.It performs the same
initial averaging with 5 km resolution using 3600 points between 35S and 20N
(Ncominat uses 3665 but 3660 has better factoring) which is subsequently shrinked 
to 183 values by averaging over 20 points.

See Ncombinat.py for the description of the 5km averaging

The input is the Ncald catalog produced by korrect_catalog_orbits and the listi 
list that determines the orbits being processed.

The output is in a 'superCatal_caliop.'+str(sel)+'_nit' file with a suffix    

This script does not produce any plot unlike Ncombinat.py

The output is a dictionary with two members ['attr'] and ['data']
The 'attr' member is a dictionary providing the latitudes and altitudes of the grid
(centers and edges), and the altitudes of the reduced meteorological data.
The 'data' member is a dictionary with a member for each orbit indexed as in Ncald.
It contains
'lon':    mean longitude of the segment 35S-20N
'fname':  name of the orbit
'P':      pressure (on reduced met grid) (Pa)
'T':      temperature (on reduced met grid) (K)
'TROPOH': tropopause altitude (km)
'SR532':  attenuated 532 nm backscattering ratio
'T532':   total attenuated 532nm backscattering (sr**-1 km**-1)
'DEPOL':  532 nm depolarization ratio (perpendicular/parallel)
'COLOR RATIO': color ratio (total 1064 nm / total 532 nm)
'INT532': integrated T532 above the tropopause                          '')

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr

"""
import pickle,gzip
from datetime import date, timedelta, datetime
from pyhdf.SD import SD, SDC
from pyhdf import HDF, VS, V
import os
import numpy as np
import socket
import scipy.ndimage as nd
from scipy.interpolate import interp1d

Qs = 5.167*1.e-31
kbw = 1.0313

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

def findbndry(t532):
    textr = t532[:105,50]
    epsilon = 1.e-3
    L1 = None
    for j in range(1000):
        if np.any(np.abs((textr[j+1:j+5]-textr[j])/np.abs(textr[j:j+5]).max()) > epsilon): continue
        L1 = j%5
        break
    if L1 is None:
        print('NO block-5 bndry found')
        return [-1,-1,-1]
    #print('block-5 bndry found',j,L1)
    N = t532.shape[0]
    L2 = N - (N-L1-4)%5
    L2 = N - (N-L1)%5
    L3 = int((L2-L1)/5)
    return [L1,L2,L3]

[alts,alts_edges] = pickle.load(open('L1_alts_edges.pkl','rb'))

ND = 'nit'
sel = 3
#figsav = True
#sav2hdf5 = False
medfilter = True
# alts1[32] = 30.2155 km 
# alts1[124] = 18.0007 km
top = 32
bottom = 125
# malts1[4] = 32.012 km
# malts1[11] = 19.078 km
mtop = 4
mbottom = 12
latmax = 20
latmin = -35
if ND == 'nit': widthfilter = 65
elif ND == 'day': widthfilter = 129

Qs = 5.167*1.e-31
kbw = 1.0313

if sel == 2:
    with gzip.open('NCald2.pkl','rb') as f:
        NCald = pickle.load(f)
elif sel == 3:
    with gzip.open('NCald3.pkl','rb') as f:
        NCald = pickle.load(f)
    listi = range(1788,len(NCald[ND])+1)
elif sel ==0:
    with gzip.open('NCald0.pkl','rb') as f:
        NCald = pickle.load(f)
    if ND=='nit': listi = [9,24,25,39,40,41,70,71]
    if ND=='day': listi = [17,33,63,64]
    top = 0
    mtop = 0

#with gzip.open('katalogOrbitsKorrect'+str(sel)+'.pkl','rb') as f:
#    Nkatalog = pickle.load(f)

lats = np.linspace(-35,20,3660)
#lats2 = 0.5*(lats[:-1]+lats[1:])
#lats2 = np.append(np.insert(lats2,0,1.5*lats[0]-0.5*lats[1]),
#                  1.5*lats[-1]-0.5*lats[-2])
interp = interp1d(lats,np.arange(3660),kind='nearest')
nlevs = bottom-top
combinat = {}
combinat['data'] = {}

#listi = range(1,len(NCald[ND])+1)

# Definition of the catalogue
catal = {'data':{},'attr':{}}

for i in listi:
#for i in range(94,109):
    catal['data'][i] = {}
    # if missing, cycle
    if NCald[ND][i]['missing']:
        print(i,' MISSING')
        catal['data'][i]['missing'] = True
        continue
    catal['data'][i]['missing'] = False
    SR = np.zeros((3660,nlevs))
    # generate date and daily directory name
    day = NCald[ND][i]['date']
    catal['data'][i]['date'] = datetime.strptime(NCald[ND][i]['fname'][:19],'%Y-%m-%dT%H-%M-%S')
    dirdayL1 = os.path.join(dirL1,day.strftime('%Y/%Y_%m_%d'))
    fileL1 = os.path.join(dirdayL1,'CAL_LID_L1-ValStage1-V3-41.'+NCald[ND][i]['fname']+'.hdf')
    hdf1 = SD(fileL1,SDC.READ)
    hh1 = HDF.HDF(fileL1,HDF.HC.READ)
    lats1 = hdf1.select('Latitude').get().flatten()
    selL1 = (lats1 <= latmax) & (lats1 >= latmin)
    if np.all(selL1):
        print(i,' NO PART OF THE ORBIT IN THE LAT RANGE')
        catal['data'][i]['missing'] = True
        continue
    lats1 = lats1[selL1]
    t532L1 = hdf1.select('Total_Attenuated_Backscatter_532').get()[selL1,:]
    p532L1 = hdf1.select('Perpendicular_Attenuated_Backscatter_532').get()[selL1,:]
    t1064L1 = hdf1.select('Attenuated_Backscatter_1064').get()[selL1,:]
    tropoL1 = hdf1.select('Tropopause_Height').get()[selL1].flatten()
    TL1 = hdf1.select('Temperature').get()[selL1,:] + 273.15
    pL1 = hdf1.select('Pressure').get()[selL1,:] * 100
    lons1 = hdf1.select('Longitude').get()[selL1].flatten()
    if lons1[0]*lons1[-1] >0:
        lons1 = lons1%360
    elif 180-np.abs(lons1[0])<20:
        lons1 = lons1%360
    catal['data'][i]['lon'] = np.mean(lons1)%360
    catal['data'][i]['fname'] = NCald[ND][i]['fname']
    mnd1 = hdf1.select('Molecular_Number_Density').get()[selL1,:]
    lbeta532_met = np.log(1000 * mnd1 * Qs / (kbw*8*np.pi/3))
    meta1 = hh1.vstart().attach('metadata')
    alts1 = np.array(meta1.read()[0][meta1.field('Lidar_Data_Altitudes')._idx])
    meta1 = hh1.vstart().attach('metadata')
    malts1 = np.array(meta1.read()[0][meta1.field('Met_Data_Altitudes')._idx])
    # calculation of the molecular backscatter
    lbeta532_lid = np.empty(shape=t532L1.shape)
    for jy in range(len(lats1)):
        lbeta532_lid[jy,:] = np.interp(alts1,malts1[::-1],lbeta532_met[jy,::-1])
    
    # Group and reduce
    [L1,L2,L3] = findbndry(t532L1)
    t532meanfull = np.mean(np.reshape(t532L1[L1:L2,:],(L3,5,583)),axis=1)
    t532mean =  t532meanfull[:,top:bottom]
    #t532mean = np.mean(np.reshape(t532L1[L1:L2,top:bottom],(L3,5,bottom-top)),axis=1)
    p532mean = np.mean(np.reshape(p532L1[L1:L2,top:bottom],(L3,5,bottom-top)),axis=1)
    latsmean = np.mean(np.reshape(lats1[L1:L2],(L3,5)),axis=1)
    lbeta532_mean = np.mean(np.reshape(lbeta532_lid[L1:L2,top:bottom],(L3,5,bottom-top)),axis=1)
    t1064mean = np.mean(np.reshape(t1064L1[L1:L2,top:bottom],(L3,5,bottom-top)),axis=1)
    tropomean = np.mean(np.reshape(tropoL1[L1:L2],(L3,5)),axis=1)
    pmean = np.mean(np.reshape(pL1[L1:L2,mtop:mbottom],(L3,5,mbottom-mtop)),axis=1)
    Tmean = np.mean(np.reshape(TL1[L1:L2,mtop:mbottom],(L3,5,mbottom-mtop)),axis=1)
    del t532L1, p532L1, lbeta532_met, lbeta532_lid, t1064L1, tropoL1, pL1, TL1

    # Filtering
    t532full = nd.median_filter(t532meanfull,mode='reflect',size=(widthfilter,1))
    sr532raw = t532mean/np.exp(lbeta532_mean)
    sr532= nd.median_filter(sr532raw,mode='reflect',size=(widthfilter,1))
    t532 = t532full[:,top:bottom]
    #t532 = nd.median_filter(t532mean,mode='reflect',size=(widthfilter,1))
    p532 = nd.median_filter(p532mean,mode='reflect',size=(widthfilter,1))
    t1064 = nd.median_filter(t1064mean,mode='reflect',size=(widthfilter,1))
    press = nd.median_filter(pmean,mode='reflect',size=(widthfilter,1))
    temp = nd.median_filter(Tmean,mode='reflect',size=(widthfilter,1))
    tropoH = nd.median_filter(tropomean,mode='reflect',size=(widthfilter,)) 
    
    # Projection on the reference grid
    idx = interp(latsmean)
    if np.sum(np.isnan(idx))>0:
        print('We have a problem with the interpolation')
        print('Skipping this image')
        continue
    idx = idx.astype(int)
    
    def shrink(sr,nlev,idx):
        SR0 = np.zeros((3660,nlev))
        for jy in range(len(latsmean)):
            for lev in range(nlev):
                SR0[idx[jy],lev] = sr[jy,lev]
        SR1 = np.ma.masked_where(SR0<0,SR0)
        SR2 = np.ma.reshape(SR1,(183,20,nlev))
        return np.ma.mean(SR2,axis=1)
    def shrink0(sr,idx):
        SR0 = np.zeros(3660)
        for jy in range(len(latsmean)):
            SR0[idx[jy]] = sr[jy]
        SR1 = np.ma.masked_where(SR0<0,SR0)
        SR2 = np.ma.reshape(SR1,(183,20))
        return np.ma.mean(SR2,axis=1)
    
    SR532 = catal['data'][i]['SR532'] = shrink(sr532,nlevs,idx)
    T532 = catal['data'][i]['T532'] = shrink(t532,nlevs,idx)
    P532 = shrink(p532,nlevs,idx)
    T1064 = shrink(t1064,nlevs,idx)
    PRESS = shrink(press,8,idx)
    TEMP = shrink(temp,8,idx)
    TROPOH = catal['data'][i]['TROPOH'] = shrink0(tropoH,idx)
    T532full = shrink(t532full,583,idx)
    dz = - alts_edges[1:] + alts_edges[:-1]
    INT532 = np.empty(183)
    for jy in range(183):
        thl = int(np.where(alts>TROPOH[jy])[0][-1])
        INT532[jy] = np.ma.sum(T532full[jy,:thl]*dz[:thl])
  
    print(i,'sr',sr532.min(),sr532.max(),'SR3',SR532 .max())
    hh1.close()
    hdf1.end()
    #catal['data'][i]['SR532'] = SR532
    catal['data'][i]['DEPOL'] = P532/(T532-P532)
    catal['data'][i]['ACR'] = T1064/T532
    catal['data'][i]['P'] = PRESS
    catal['data'][i]['T'] = TEMP
    catal['data'][i]['INT532'] = INT532

lats3 = np.mean(np.reshape(lats,(183,20)),axis=1)
lats3_edge = 0.5*(lats3[:-1]+lats3[1:])
lats3_edge = np.append(np.insert(lats3_edge,0,1.5*lats3[0]-0.5*lats3[1]),
                  1.5*lats3[-1]-0.5*lats3[-2])
catal['attr'] = {'lats':lats3,'alts':alts[top:bottom],'lats_edge':lats3_edge,
                 'alts_edge':alts_edges[top:bottom+1],'malts':malts1[mtop:mbottom]}
with gzip.open(os.path.join('.','superCatal_caliop.'+str(sel)+'_'+ND+'-s15.pkl'),'wb') as f:
        pickle.dump(catal,f,protocol=pickle.HIGHEST_PROTOCOL)
