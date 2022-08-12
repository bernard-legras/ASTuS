# -*- coding: utf-8 -*-
"""
Produce CALIOP plots for the selected expedited orbits data

This script averages the CALIOP sections for each day in a lat x alt view.
The view is produced as an image in the Ncombinat_nit or Ncombinat_day directory
that must exist in the same directory.
It is also genarating a Ncombinat-daily file  which name must be updated manually
for each run at the end of the script. These files are subsenquently combined
itertively within a file Ncombinat-daily_(nit,day).all.pkl which contains all 
the sections from the beginning to the end of the event (that is combines sel=2
and sel=3).

The displayed quantity is the 532nm attenuated scattering ratio defined by using 
the ATBD formulae for the molecular scattering.  

In the present application the levels 32 to 125 are selected in the original 
section (93 levels) and the latitudes are truncated between 35S and 20N.
As the latitude sampling varies from orbit to orbit, the [35S, 20N] interval is
sampled over 3665 values corresponding to 5 km resolution which is nominal 
between 20 and 30 km. The original data are grouped and average by 5. 
The findbndry fuction finds the boundary of blocks where the values are identical.
This should be the case, but not always, hence the exploration by finbndry to 
find the first boundary L1 and the subsequent averaging.
Once this averaging is done, the value in the fixed 3665-pixel range is determined
by nearest neighbour interpolation.

The data are subsequently filtered in latitude using a median filter of width 
widthfilter (65 for nit and 129 for day).

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr
"""
import pickle,gzip
import matplotlib.pyplot as plt
from datetime import date, timedelta
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
    textr = t532[:1005,50]
    print('shape',t532.shape)
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

def show(sr,lats2,top=32,bottom=125,vmax=30,cmap='gist_ncar',title='',show=True,vmin=0):
    alts2 = alts_edges[top:bottom+1]
    plt.pcolormesh(lats2,alts2,sr.T,cmap=cmap,vmin=vmin,vmax=vmax)
    plt.xlabel('Latitude')
    plt.ylabel('Altitude (km)')
    plt.title(title)
    plt.colorbar()
    plt.grid(True)
    if show: plt.show()
    return

ND = 'nit'
sel = 3
figsav = True
sav2hdf5 = False
medfilter = True
ysup = 30
yinf = 18
top = 32
bottom = 125
vmax = 30
GEO = True
latmax = 20
latmin = -35
if ND == 'nit': widthfilter = 65
elif ND == 'day': widthfilter = 129

Qs = 5.167*1.e-31
kbw = 1.0313

with gzip.open('katalogOrbitsKorrect'+str(sel)+'.pkl','rb') as f:
    Nkatalog = pickle.load(f)

lats = np.linspace(-35,20,3665)
lats2 = 0.5*(lats[:-1]+lats[1:])
lats2 = np.append(np.insert(lats2,0,1.5*lats[0]-0.5*lats[1]),
                  1.5*lats[-1]-0.5*lats[-2])
interp = interp1d(lats,np.arange(3665),kind='nearest')
nlevs = bottom-top
combinat = {}
combinat['data'] = {}

day0 = date(2022,1,27)
day1 = date(2022,3,26)
day0 = date(2022,1,30)
day1 = date(2022,1,30)
day0 = date(2022,3,25)
day1 = date(2022,3,26)
day0 = date(2022,3,31)
day1 = date(2022,4,14)
day0 = date(2022,5,4)
day1 = date(2022,5,6)
day0 = date(2022,5,15)
day1 = date(2022,5,19)
day0 = date(2022,6,29)
day1 = date(2022,7,11)
day = day0

while day <= day1:
    # generate accumulators (15001,32-125)
    SR = np.zeros((3665,93))
    npix = np.zeros((3665,93),dtype=int)
    print (day.strftime('Processing %Y-%m-%d'))
    for j in range(len(Nkatalog[day][ND]['fname'])):
        if Nkatalog[day][ND]['missing'][j]:
            print ('Missing image')
            continue
        print ('Image '+str(j)+'  '+Nkatalog[day][ND]['fname'][j])
        # generate date and daily directory name
        dirdayL1 = os.path.join(dirL1,day.strftime('%Y/%Y_%m_%d'))
        fileL1 = os.path.join(dirdayL1,'CAL_LID_L1-ValStage1-V3-41.'+Nkatalog[day][ND]['fname'][j]+'.hdf')
        hdf1 = SD(fileL1,SDC.READ)
        hh1 = HDF.HDF(fileL1,HDF.HC.READ)
        lats1 = hdf1.select('Latitude').get().flatten()
        selL1 = (lats1 <= latmax) & (lats1 >= latmin)
        if np.all(selL1):
            print(j,' NO PART OF THE ORBIT IN THE LAT RANGE')
            continue
        lats1 = lats1[selL1]
        t532L1 = hdf1.select('Total_Attenuated_Backscatter_532').get()[selL1,:]
        if ND == 'day':
            lons1 = hdf1.select('Longitude').get()[selL1].flatten()
            if lons1[0]*lons1[-1] >0:
                lons1 = lons1%360
            elif 180-np.abs(lons1[0])<20:
                lons1 = lons1%360
            ml = np.mean(lons1)%360
            if (ml > 280) & (ml < 325):
                print("Achtung SAA")
                continue
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
        t532mean = np.mean(np.reshape(t532L1[L1:L2,top:bottom],(L3,5,bottom-top)),axis=1)
        latsmean = np.mean(np.reshape(lats1[L1:L2],(L3,5)),axis=1)
        lbeta532_mean = np.mean(np.reshape(lbeta532_lid[L1:L2,top:bottom],(L3,5,bottom-top)),axis=1)

        # Filtering
        if medfilter:
            sr532raw = t532mean/np.exp(lbeta532_mean)
            sr532= nd.median_filter(sr532raw,mode='reflect',size=(widthfilter,1))
        else:
            sr532 = t532mean/np.exp(lbeta532_mean)
        #sr532c = np.clip(sr532,0,1000)

        # Projection on the reference grid
        idx = interp(latsmean)
        if np.sum(np.isnan(idx))>0:
            print('We have a problem with the interpolation')
            print('Jumping this image')
            continue
        idx = idx.astype(int)
        for jy in range(len(latsmean)):
            for lev in range(nlevs):
                if sr532[jy,lev] >=0:
                    SR[idx[jy],lev] += sr532[jy,lev]
                    npix[idx[jy],lev] += 1
        print('sr',sr532.min(),sr532.max(),'SR',SR.max())
        hh1.close()
        hdf1.end()

    # Doing the average on SR
    print('Output step')
    npixr = np.clip(npix,1,1000)
    SR = SR/npixr
    combinat['data'][day] = {}
    combinat['data'][day]['SR'] = SR
    combinat['data'][day]['npix'] = npix
    fig = plt.figure(figsize=(7,6))
    #norm1 = colors.LogNorm(vmin=0.00001,vmax=0.1)
    show(SR,lats2,show=False,vmax=8,
         title='composite SR 532  '+day.strftime('%Y-%m-%d'))
    plt.savefig(os.path.join('NCombinat_'+ND,str(day.month)+'.Combinat-'+str(sel)+'_'+ND+day.strftime('-%d%B.png')),dpi=300,bbox_inches='tight')

    day += timedelta(days=1)
    plt.close()

    combinat['attr'] = {'lats':lats,'alts':alts[top:bottom],'lats_edge':lats2,'alts_edge':alts_edges[top:bottom+1]}
    with gzip.open(os.path.join('.','Ncombinat-daily-'+str(sel)+'_'+ND+'-s13.pkl'),'wb') as f:
        pickle.dump(combinat,f,protocol=pickle.HIGHEST_PROTOCOL)
