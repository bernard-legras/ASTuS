---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: p39N
    language: python
    name: p39n
---

<!-- #region tags=[] -->
# Processing of OMPS / CALIOP / MLS data for Hunga Tonga
<!-- #endregion -->

Copyright or © or Copr.  Bernard Legras & Clair Duchamp (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr

clair.duchamp@lmd.ipsl.fr


## Initializations

```python
from netCDF4 import Dataset
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.dates as mdates
from datetime import datetime, timedelta, date, time
import glob
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline
import gzip, pickle
import socket
import constants as cst
from scipy.interpolate import Akima1DInterpolator as aki
from scipy.ndimage import gaussian_filter
import scipy.signal as sps 

# Locate the data
hostname = socket.gethostname()
if hostname == 'satie':
    dirOMPS = '/data/OMPS/v2.1/2022'
elif hostname == 'Mentat':
    dirOMPS = "C:\\cygwin64\\home\\berna\\data\\OMPS\\v2.1\\2022"
    #dirOMPS = "C:\\cygwin64/home/berna/data/OMPS/2022"
    #dirOMPS = os.path.join('..','..','OMPS','v2.1',2022')
elif hostname == 'gort':
    dirOMPS = '/dkol/data/OMPS/v2.1/2022'

def flatten(t):
    return [item for sublist in t for item in sublist]
```

<!-- #region tags=[] -->
## Making the projected daily sections in lat alt for the raw extinction
<!-- #endregion -->

Quality flag tested

<!-- #region tags=[] -->
### Definitions
<!-- #endregion -->

OMPS fixed latitude grid with 50 bins between -35 and 20, that is with a latitude resolution of 1.1 degree that approximates the actual resolution of OMPS

```python
extended = False
if extended:
    latmin = -70.2
    latmax = 53
else:
    latmin = -35
    latmax = 20
dl = 1.1
latsRefEdge = np.arange(latmin,latmax+0.1,dl)
latsRef = np.arange(latmin+0.5*dl,latmax+0.1,dl)
altsRef = np.linspace(0.5,40.5,41)
altsRefEdge = np.linspace(0,41,42)
ny = nlat = len(latsRef)
nalt = 41
# Interpolation trick to manage the full interval [latmin, latmax]
interp = interp1d(np.arange(latmin-0.5*dl,latmax+dl,dl),np.arange(-1,nlat+1),kind='nearest')
combinat = {}
combinat['data'] = {}
combinat['attr'] = {'lats':latsRef,'alts':altsRef,
                    'lats_edge':latsRefEdge,
                    'alts_edge':altsRefEdge}
day0 = date(2022,1,1)
day1 = date(2022,7,26)
day = day0
# Option to filter all latitudes in the SAA range
SAA_filt = True
# Initialization of the molecular extinction
lamb = [510., 600., 675., 745., 869., 997.]
# This is formula 4.17 of the CALIOP L1 ATBD PS-SCI-201.v1.0
QS = 4.5102e-31*(lamb[3]/550)**(-4.025-0.05627*(lamb[3]/550)**(-1.647)) # in m**2 
NA = cst.Na
RA = cst.Ra
```

<!-- #region tags=[] -->
### Performing the action
<!-- #endregion -->

```python jupyter={"source_hidden": true} tags=[]
day = day0
while day <= day1 :
    print(day)
    file = day.strftime('OMPS-NPP_LP-L2-AER-DAILY_v2.1_%Ym%m%d_*.h5')
    combinat['data'][day] = {}
    # Exception for 11 and 13 May
    if day in [date(2022,5,11),date(2022,5,13)]:
        file = day.strftime('OMPS-NPP_LP-L2-AER-DAILY_v2.0_%Ym%m%d_*.h5')
    search = os.path.join(dirOMPS,file)
    
    # Open the file and read the needed field 
    try:
        fname = glob.glob(search)[0]
        ncid = Dataset(fname)
    except: 
        print('missing day',day)
        combinat['data'][day]['missing'] = True
        combinat['data'][day]['npix'] = np.ma.empty_like(npixMa)
        combinat['data'][day]['npix'][...] = np.ma.masked
        for var in ['maxExt','meanExt','stdevExt''maxExtratio','meanExtRatio','stdevExtRatio']:
            combinat['data'][day][var] = np.ma.empty_like(meanExtMa)
            combinat['data'][day][var][...] = np.ma.masked 
        day += timedelta(days=1)
        continue
        
    combinat['data'][day]['missing'] = False
    alt = ncid.groups['ProfileFields']['Altitude'][:]
    lats=ncid.groups['GeolocationFields']['Latitude'][:]
    lons=ncid.groups['GeolocationFields']['Longitude'][:]                                                  
    ext=ncid.groups['ProfileFields']['RetrievedExtCoeff'][:]
    seconds = ncid.groups['GeolocationFields']['SecondsInDay'][:]
    quals = ncid.groups['GeolocationFields']['SwathLevelQualityFlags'][:]
    press = ncid.groups['AncillaryData']['Pressure'][:]
    temp = ncid.groups['AncillaryData']['Temperature'][:]
    
    orbit1 = ncid.OrbitNumberStart
    orbit2 = ncid.OrbitNumberStop
    
    # Define accumulators for the date
    meanExt = np.zeros(shape=(len(latsRef),41))
    npix = np.zeros(shape=(len(latsRef),41),dtype=int)
    maxExt = np.zeros(shape=(len(latsRef),41))
    varExt = np.zeros(shape=(len(latsRef),41))
    meanExtRatio = np.zeros(shape=(len(latsRef),41))
    maxExtRatio = np.zeros(shape=(len(latsRef),41))
    varExtRatio = np.zeros(shape=(len(latsRef),41))
    
    # Loop on the orbit for that day
    for orbit in range(orbit1,orbit2+1):
        # Select the orbit number
        selec=(ncid.groups['GeolocationFields']['OrbitNumber'][:] == orbit)
        if np.sum(selec) == 0:
            print('Missing orbit',date,orbit)
            continue
        xalt = np.arange(0,42)
        quals2 = quals[selec]
        # Select middle slit and 745 nm channel for the orbit
        lats2 = lats[selec,1]
        ext2 = ext[selec,1,3,:]
        press2 = press[selec,1,:]
        temp2 = temp[selec,1,:]
        # Select latitude band
        selec2 = (lats2<latmax) & (lats2>latmin)
        lats3 = lats2[selec2]
        if len(lats3) != len(lats3.compressed()):
            # Should not happen
            print('Masked lat on this orbit')
            continue
        quals3 = quals2[selec2]
        ext3 = ext2[selec2,:]
        # Notice that temp & press are occasionally masked
        # Pressures are in hPa
        press3 = press2[selec2,:]*100
        temp3 = temp2[selec2,:]
        # Molecular extinction in km-1
        extRef = (NA*QS*1000/RA)*(press3/temp3)
        # Separate and test quality flag
        saa = quals3 & 3
        moon = (quals3 & 0xC) >> 2
        planets = (quals3 & 0x60) >> 5
        nonom = (quals3 & 0x80) >> 7
        if np.any(nonom == 1):
            print('non nominal attitude on orbit ',orbit)
            #sel3 = (nonom == 1)
            #print(lats3[sel3])
        if np.any((moon == 2) | planets == 2):
            print('moon or planets on orbit ',orbit)
            continue
        if np.any(saa >= 1):
            if np.any(saa >= 2):
                print('disabled by SAA ',orbit)
            else:
                print('polluted by SAA ',orbit)
            if SAA_filt:
                sel3 = (saa >= 1)
                ext3[sel3,:] = np.ma.masked      
        idx = interp(lats3.compressed())
        if np.sum(np.isnan(idx))>0:
            print('We have a problem with the interpolation')
            continue
        idx = np.clip(idx,0,nlat-1).astype(int)
        rat = ext3/extRef
        for lev in range(nalt):
            for jy in range(ext3.shape[0]):
                if (not ext3.mask[jy,lev]) & (not extRef.mask[jy,lev]):
                    meanExt[idx[jy],lev] += ext3[jy,lev]
                    maxExt[idx[jy],lev] = max(maxExt[idx[jy],lev],ext3[jy,lev])
                    varExt[idx[jy],lev] += ext3[jy,lev]**2
                    meanExtRatio[idx[jy],lev] += rat[jy,lev]
                    maxExtRatio[idx[jy],lev] = max(maxExtRatio[idx[jy],lev],rat[jy,lev])
                    varExtRatio[idx[jy],lev] += rat[jy,lev]**2
                    npix[idx[jy],lev] += 1
    npixMa = np.ma.masked_equal(npix,0)
    maxExtMa = np.ma.masked_equal(maxExt,0)
    meanExtMa = np.ma.masked_equal(meanExt,0)
    varExtMa = np.ma.masked_equal(varExt,0)
    meanExtMa /= npixMa
    stdevExtMa = np.ma.sqrt(varExtMa/npixMa - meanExtMa**2)
    maxExtRatioMa = np.ma.masked_equal(maxExtRatio,0)
    meanExtRatioMa = np.ma.masked_equal(meanExtRatio,0)
    varExtRatioMa = np.ma.masked_equal(varExtRatio,0)
    meanExtRatioMa /= npixMa
    stdevExtRatioMa = np.ma.sqrt(varExtRatioMa/npixMa - meanExtRatioMa**2)
    
    combinat['data'][day]['maxExt'] = maxExtMa
    combinat['data'][day]['meanExt'] = meanExtMa
    combinat['data'][day]['stdevExt'] = stdevExtMa
    combinat['data'][day]['maxExtratio'] = maxExtRatioMa
    combinat['data'][day]['meanExtRatio'] = meanExtRatioMa
    combinat['data'][day]['stdevExtRatio'] = stdevExtRatioMa
    combinat['data'][day]['npix'] = npixMa
    
    day += timedelta(days=1)
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
#### Some tests exploring the error flag 
<!-- #endregion -->

Here a test that show that the SAA can lead to non exploitable retrieval when the expected error is larger than 40%

```python
date = datetime(2022,3,11)
file = date.strftime('OMPS-NPP_LP-L2-AER-DAILY_v2.0_%Ym%m%d_*.h5')
search = os.path.join(dirOMPS,file)
fname = glob.glob(search)[0]
ncid = Dataset(fname)
alt = ncid.groups['ProfileFields']['Altitude'][:]
lats=ncid.groups['GeolocationFields']['Latitude'][:]
quals = ncid.groups['GeolocationFields']['SwathLevelQualityFlags'][:]
ext=ncid.groups['ProfileFields']['RetrievedExtCoeff'][:]
selec=(ncid.groups['GeolocationFields']['OrbitNumber'][:] == 53733)
lats2 = lats[selec,1]
quals2 = quals[selec]
ext2 = ext[selec,1,3,:]
#plt.plot(lats2,(quals2  & 0x80)>> 7)
plt.plot(lats2,quals2 & 3)
```

```python
plt.pcolormesh(lats2,alt,ext2.T,vmin=0,vmax=0.004)
plt.ylim(15,30)with gzip.open('combinat-daily.pkl','wb') as f:
        pickle.dump(combinat,f,protocol=pickle.HIGHEST_PROTOCOL
```

```python
ncid.groups['ProfileFields']['Wavelength'][:]
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
#### ncid header
<!-- #endregion -->

```python
ncid
```

### Storing the result


The saved version was v2.1 for the initial submission. It changes to v2.2 with the revised version using same filters but extended in time with the 'missing' variable.
DO NOT RUN THIS CELL IF THE PROCESSING HAS NOT BEEN DONE BEFORE, IT ERASES DATA ON DISK

```python
name = 'combinat-daily'
if extended: name = 'combinat-daily-extended'
if SAA_filt:
    with gzip.open(name+'-SAAfilt.v2.2.pkl','wb') as f:
        pickle.dump(combinat,f,protocol=pickle.HIGHEST_PROTOCOL)
else:
    with gzip.open(name+'.v2.2.pkl','wb') as f:
        pickle.dump(combinat,f,protocol=pickle.HIGHEST_PROTOCOL)
```

### Loading the result


The saved version was v2.1 for the initial submission. It changes to v2.2 with the revised version using same filters but extended in time with the 'missing' variable.

```python
with gzip.open('combinat-daily-SAAfilt.v2.2.pkl','rb') as f:
    combinat = pickle.load(f)
```

<!-- #region tags=[] -->
## Building and plotting extinction ratios
<!-- #endregion -->

<!-- #region tags=[] -->
### Reference extinction from the first 12 days of 2022
<!-- #endregion -->

This is obsolete.
It replaces extRef calculated in the loop above for each day by an empirical mean over the first 12 days of 2022.

```python
extRefEmpir = np.ma.zeros(shape=(combinat['data'][date(2022,1,1)]['meanExt'].shape))
npixRefEmpir = np.ma.zeros(shape=(combinat['data'][date(2022,1,1)]['meanExt'].shape))
day = date(2022,1,1)
while day < date(2022,1,11):
    extRefEmpir += combinat['data'][day]['meanExt']
    npixRefEmpir += 1 - combinat['data'][day]['meanExt'].mask.astype(int)
    day += timedelta(days=1)
extRefEmpir /= npixRefEmpir
backExt = {}
backExt['attr'] = combinat['attr']
backExt['data'] = extRefEmpir
# with gzip.open('backgroundExtinction.pkl','wb') as f:
#    pickle.dump(backExt,f)
```

<!-- #region tags=[] -->
### Plotting ext ratio
<!-- #endregion -->

#### Common elements

```python
def plot1(day,ylim=(18,30),vmax=None,ax=None,txt=None,cmap='jet',ratio=True, empir=False, 
          xlabel=True,showlat=True,showalt=True,annot=False):
    if ratio:
        if empir: extRatio = combinat['data'][day]['meanExt']/extRefEmpir
        else: extRatio = combinat['data'][day]['meanExtRatio']
        if vmax == None: vmax = 20
        if txt == None: txt = day.strftime('OMPS 745nm extinction ratio %d %b %Y')
    else:
        extRatio = combinat['data'][day]['meanExt']
        if vmax == None: vmax = 0.004
        if txt == None: txt = day.strftime('OMPS 745nm extinction %d %b %Y')    
    latsEdge = combinat['attr']['lats_edge']
    altsEdge = combinat['attr']['alts_edge']
    if ax == None: fig,axe = plt.subplots(nrows=1,ncols=1)
    else: axe = ax
    im = axe.pcolormesh(latsEdge,altsEdge,extRatio.T,cmap=cmap,vmin=0,vmax=vmax)
    if ax == None: plt.colorbar(im)
    axe.set_ylim(ylim[0],ylim[1])
    axe.set_xlim(-35,20)
    if annot: axe.annotate("OMPS-LP",(-9,28.5), fontsize=9.5, color="white")
    if showlat==True: axe.set_xlabel('Latitude') 
    else : ax.set_xticklabels([])    
    if showalt==True: axe.set_ylabel('Altitude')
    else : ax.set_yticklabels([])
    axe.grid(True)
    axe.set_title(txt)
    # plt.show()
    return im
```

<!-- #region jupyter={"source_hidden": true} tags=[] -->
#### Plotting for a few dates and testing
<!-- #endregion -->

Comparing the extinction ratio using the empirical estimate and the "theoretical estimate" of molecular Rayleigh extinction.
Note that there is an apparent factor 2 between the empirical and the calculated background extinction (instrumental factor?).

```python
# With the empirical estimate of the molecular extinction
plot1(date(2022,1,30),ylim=(15,40),empir=True,vmax=15,cmap='gist_ncar')
plt.show()
# With the theoretical estimate of the molecular extinction
plot1(date(2022,1,30),ylim=(15,40),empir=False,vmax=30,cmap='gist_ncar')
plt.show()
# Plot the pure extinction
plot1(date(2022,1,30),ylim=(15,40),ratio=False,cmap='gist_ncar')
plt.show()
```

Comparing the ratios with SAA (v2.0) and no SAA filtering (v2.1)

```python
with gzip.open('combinat-daily.pkl','rb') as f:
    combinat = pickle.load(f)
plot1(date(2022,3,11),ylim=(15,40),empir=False,vmax=30,cmap='gist_ncar')
plt.show()
with gzip.open('combinat-daily-SAAfilt.v2.2.pkl','rb') as f:
    combinat = pickle.load(f)
plot1(date(2022,3,11),ylim=(15,40),empir=False,vmax=30,cmap='gist_ncar')
plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
#### Plots of the initial days
<!-- #endregion -->

```python
j = 0
fig, axes = plt.subplots(figsize=(21,16),ncols=5,nrows=3,sharex=True,sharey=True)
axes = flatten(axes)
for i in range(16,31):
    day = date(2022,1,i)
    im = plot1(day,ylim=(18,34),vmax=20,ax=axes[j],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar')
    j += 1
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.03])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('OMPS 745 nm daily zonal average extinction ratio',y=0.9,fontsize=24)
plt.savefig('EarlyOMPS.v2.1.png',dpi=300,bbox_inches='tight')
plt.savefig('EarlyOMPS.v2.1.pdf',dpi=300,bbox_inches='tight')
plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
#### Plots from 27 January onward
<!-- #endregion -->

```python
figsave = True
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = date(2022,1,27) + timedelta(days=2*j)
    im = plot1(day,vmax=40,ax=axes[j],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar')
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('OMPS 745 nm daily zonal average extinction ratio',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-1-OMPS.v2.1.png',dpi=300,bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = date(2022,4,21) + timedelta(days=2*j)
    im = plot1(day,vmax=40,ax=axes[j],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar')
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('OMPS 745 nm daily zonal average extinction ratio',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-2-OMPS.v2.1.png',dpi=300,bbox_inches='tight')
plt.show()

#fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
#axes = flatten(axes)
#for j in range(28):
#    day = date(2022,6,2) + timedelta(days=2*j)
#    im = plot1(day,vmax=40,ax=axes[j],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar')
#fig.subplots_adjust(top=0.8)
#cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
#fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
#fig.suptitle('OMPS 745 nm daily zonal average extinction ratio',y=0.9,fontsize=24)
#if figsave: plt.savefig('HT-3-OMPS.v2.1.png',dpi=300,bbox_inches='tight')
#plt.show()
```

<!-- #region tags=[] -->
## Plots from CALIOP combined files
<!-- #endregion -->

### Read the CALIOP data


For the sake of speed, we use the compact superCombi version with reduced latitude resolution.

```python
with gzip.open(os.path.join('..','HT-HT','superCombi_caliop.all_nit.pkl'),'rb') as f:
    combinat_CALIOP = pickle.load(f)
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Dedicated plot routine for CALIOP SR
<!-- #endregion -->

```python
def plot2(day,var='SR532',ylim=(18,30),vmax=8,ax=None,txt=None,cmap='jet',
          showlat=True,showalt=True,annot=False):
    SR = combinat_CALIOP['data'][day][var]
    latsEdge = combinat_CALIOP['attr']['lats_edge']
    altsEdge = combinat_CALIOP['attr']['alts_edge']
    if ax == None: fig,axe = plt.subplots(nrows=1,ncols=1)
    else: axe = ax
    im = axe.pcolormesh(latsEdge,altsEdge,SR.T,cmap=cmap,vmin=0,vmax=vmax)
    if ax == None: plt.colorbar(im)
    axe.set_ylim(ylim[0],ylim[1])
    axe.set_xlim(-35,20)
    if annot: axe.annotate("CALIOP",(-9,28.5), fontsize=9.5, color="white")
    if showlat==True: axe.set_xlabel('Latitude') 
    else : ax.set_xticklabels([])    
    if showalt==True: axe.set_ylabel('Altitude')
    else : ax.set_yticklabels([])
    axe.grid(True)
    if txt == None: axe.set_title(day.strftime('CALIOP 532 nm attenuated scattering ratio %d %b %Y'))
    else: axe.set_title(txt)
    return im
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
### Example with a single frame
<!-- #endregion -->

```python
plot2(date(2022,2,27),cmap='gist_ncar')
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
### Multiframe plot
<!-- #endregion -->

```python
figsave = True
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = date(2022,1,27) + timedelta(days=2*j)
    if day not in combinat_CALIOP['data']: continue
    im = plot2(day,ax=axes[j],txt=day.strftime('%d %b %Y'),cmap='gist_ncar')
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('CALIOP daily zonal average 532 nm attenuated scattering ratio',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-1-CALIOP.png',dpi=300,bbox_inches='tight')
plt.show()
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = date(2022,4,21) + timedelta(days=2*j)
    if day not in combinat_CALIOP['data']: continue
    im = plot2(day,ax=axes[j],txt=day.strftime('%d %b %Y'),cmap='gist_ncar')
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('CALIOP daily zonal average 532 nm attenuated scattering ratio',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-2-CALIOP.png',dpi=300,bbox_inches='tight')
plt.show()
#fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
#axes = flatten(axes)
#for j in range(40):
#    day = date(2022,4,21) + timedelta(days=j)
#    if day not in combinat_CALIOP['data']: continue
#    im = plot2(day,ax=axes[j],txt=day.strftime('%d %b %Y'),cmap='gist_ncar')
#fig.subplots_adjust(top=0.8)
#cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
#fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
#fig.suptitle('CALIOP daily zonal average 532 nm attenuated scattering ratio',y=0.9,fontsize=24)
#if figsave: plt.savefig('HT-3-CALIOP.png',dpi=300,bbox_inches='tight')
#plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
### Composite image with both OMPS and CALIOP
<!-- #endregion -->

```python
figsave = True
fig, axes = plt.subplots(figsize=(21,60),ncols=6,nrows=14,sharex=True,sharey=True)
#axes = flatten(axes)
for j in range(42):
    ix = j%6
    jy = int((j - j%6)/6)
    day = date(2022,1,27) + timedelta(days=2*j)
    try: im1 = plot1(day,vmax=40,ax=axes[2*jy,ix],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar')
    except: pass
    try: im2 = plot2(day,ax=axes[2*jy+1,ix],txt=day.strftime('%d %b %Y'),cmap='gist_ncar')
    except: pass
fig.subplots_adjust(top=0.8)
cbar_ax1 = fig.add_axes([0.20, 0.84, 0.6, 0.01])
fig.colorbar(im1, cax=cbar_ax1,orientation='horizontal')
cbar_ax2 = fig.add_axes([0.20, 0.82, 0.6, 0.01])
fig.colorbar(im2, cax=cbar_ax2,orientation='horizontal')
fig.suptitle('OMPS 745 nm daily zonal average extinction ratio\n'+ \
            'CALIOP daily zonal average 532 nm attenuated scattering ratio', \
             y=0.88,fontsize=24)
if figsave: plt.savefig('HT-1-Combined_OMPS-LP_CALIOP.png',dpi=300,bbox_inches='tight')
# much too long
#plt.savefig('Combined_OMPS-LP_CALIOP.pdf',dpi=300,bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(figsize=(21,60),ncols=6,nrows=14,sharex=True,sharey=True)
#axes = flatten(axes)
for j in range(42):
    ix = j%6
    jy = int((j - j%6)/6)
    day = date(2022,4,21) + timedelta(days=2*j)
    try: im1 = plot1(day,vmax=40,ax=axes[2*jy,ix],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar')
    except: pass
    try: im2 = plot2(day,ax=axes[2*jy+1,ix],txt=day.strftime('%d %b %Y'),cmap='gist_ncar')
    except: pass
fig.subplots_adjust(top=0.8)
cbar_ax1 = fig.add_axes([0.20, 0.84, 0.6, 0.01])
fig.colorbar(im1, cax=cbar_ax1,orientation='horizontal')
cbar_ax2 = fig.add_axes([0.20, 0.82, 0.6, 0.01])
fig.colorbar(im2, cax=cbar_ax2,orientation='horizontal')
fig.suptitle('OMPS 745 nm daily zonal average extinction ratio\n'+ \
            'CALIOP daily zonal average 532 nm attenuated scattering ratio', \
             y=0.88,fontsize=24)
if figsave: plt.savefig('HT-2-Combined_OMPS-LP_CALIOP.png',dpi=300,bbox_inches='tight')
# much too long
#plt.savefig('Combined_OMPS-LP_CALIOP.pdf',dpi=300,bbox_inches='tight')
plt.show()
```

#### Selected one row composite

```python
len(combinat['data'])
```

```python
figsave = True
fig, axes = plt.subplots(figsize=(21*8/7,7),ncols=8,nrows=2,sharex=True,sharey=True)
#axes = flatten(axes)
shift = range(0,108,14)
for j in range(8):
    day = date(2022,1,28) + timedelta(days=shift[j])
    im1 = plot1(day,vmax=40,ax=axes[0,j],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar',showlat=False)
    im2 = plot2(day,ax=axes[1,j],txt=day.strftime('%d %b %Y'),cmap='gist_ncar')
fig.subplots_adjust(top=0.7,bottom=0.1)
cbar_ax1 = fig.add_axes([0.36, 0.77, 0.3, 0.02])
fig.colorbar(im1, cax=cbar_ax1,orientation='horizontal')
cbar_ax2 = fig.add_axes([0.36, 0.01, 0.3, 0.02])
fig.colorbar(im2, cax=cbar_ax2,orientation='horizontal')
fig.suptitle('OMPS 745 nm daily zonal average extinction ratio\n'+ \
            'CALIOP daily zonal average 532 nm attenuated scattering ratio', \
             y=0.88,fontsize=20)
if figsave: plt.savefig('HT-spe-1raw-Combined_OMPS-LPv2.1_CALIOP.png',dpi=300,bbox_inches='tight')
# much too long
#plt.savefig('Combined_OMPS-LP_CALIOP-1row.pdf',dpi=300,bbox_inches='tight')
plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
## Ratio of CALIOP scattering to OMPS extinction
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Dedicated plot routine for the ratio of scattering to extinction
<!-- #endregion -->

```python
def plot4(day,ylim=(18,30),vmax=None,vmin=None,ax=None,txt=None,cmap='jet'):
    # Interpolation of OMPS extinction to the CALIOP grid
    interp = RectBivariateSpline(combinat['attr']['lats'],combinat['attr']['alts'],combinat['data'][day]['meanExt'])
    TT = combinat_CALIOP['data'][day]['T532']
    SR = combinat_CALIOP['data'][day]['SR532']
    smooth = np.zeros(shape=SR.shape)
    for kz in range(len(combinat_CALIOP['attr']['alts'])):
        smooth[:,kz] = np.reshape(interp(combinat_CALIOP['attr']['lats'],combinat_CALIOP['attr']['alts'][kz]),183)
    # Calculation of the scattering/extinction ratio masked by CALIOP SR
    ratio = smooth / TT
    ratio = np.ma.masked_where(SR<1,ratio)
    # Axes for pcolormesh and contour
    latsEdge = combinat_CALIOP['attr']['lats_edge']
    altsEdge = combinat_CALIOP['attr']['alts_edge']
    lats = combinat_CALIOP['attr']['lats']
    alts = combinat_CALIOP['attr']['alts']
    if ax == None: fig,axe = plt.subplots(nrows=1,ncols=1)
    else: axe = ax
    # Plot of the ratio
    im = axe.pcolormesh(latsEdge,altsEdge,ratio.T,cmap=cmap,norm=colors.LogNorm(vmax=vmax,vmin=vmin))
    if ax == None: plt.colorbar(im)
    # Plot of the SR contour 1.5
    axe.contour(lats,alts,SR.T,levels=(1.5,))
    axe.set_ylim(ylim[0],ylim[1])
    axe.set_xlim(-35,10)
    axe.set_xlabel('Latitude')
    axe.set_ylabel('Altitude')
    axe.grid(True)
    if txt == None: axe.set_title(day.strftime('Scattering to extinction ratio %d %b %Y'))
    else: axe.set_title(txt)
    return im
```

<!-- #region tags=[] -->
#### Examples 
<!-- #endregion -->

```python
plot4(date(2022,1,31),vmax=100,vmin=1); plt.show()
plot4(date(2022,3,1),vmax=100,vmin=1); plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Multiframe plot
<!-- #endregion -->

```python
figsave = True
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = date(2022,1,27) + timedelta(days=2*j)
    try: im = plot4(day,ax=axes[j],txt=day.strftime('%d %b %Y'),vmax=100,vmin=1)
    except: pass
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('745 nm OMPS extinction ratio to 532nm CALIOP attenuated scattering ratio (sr)',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-1-Extinction_to_scattering.png',dpi=300,bbox_inches='tight')
plt.show()
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = date(2022,4,21) + timedelta(days=2*j)
    try: im = plot4(day,ax=axes[j],txt=day.strftime('%d %b %Y'),vmax=100,vmin=1)
    except: continue
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('745 nm OMPS extinction ratio to 532nm CALIOP attenuated scattering ratio (sr)',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-2-Extinction_to_scattering.png',dpi=300,bbox_inches='tight')
plt.show()
#fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
#axes = flatten(axes)
#for j in range(36):
#    day = date(2022,4,21) + timedelta(days=2*j)
#    if day not in combinat_CALIOP['data']: continue
#    im = plot4(day,ax=axes[j],txt=day.strftime('%d %b %Y'),vmax=100,vmin=1)
#fig.subplots_adjust(top=0.8)
#cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
#fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
#fig.suptitle('745 nm OMPS extinction ratio to 532nm CALIOP attenuated scattering ratio (sr)',y=0.9,fontsize=24)
#if figsave: plt.savefig('HT-3-Extinction_to_scattering.png',dpi=300,bbox_inches='tight')
#plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
## Ratio of variance to the mean OMPS Extinction
<!-- #endregion -->

### Dedicated plot function

```python
def plot3(day,ylim=(18,30),vmax=3,ax=None,txt=None,cmap='gist_ncar',offset=2):
    extRatio = combinat['data'][day]['meanExtRatio']
    varRatio = combinat['data'][day]['stdevExtRatio']/combinat['data'][day]['meanExtRatio']
    varRatio[extRatio<3] = np.ma.masked
    if txt == None: txt = day.strftime('OMPS 745 nm std dev / mean ratio %d %b %Y')
    latsEdge = combinat['attr']['lats_edge']
    altsEdge = combinat['attr']['alts_edge']
    lats = combinat['attr']['lats']
    alts = combinat['attr']['alts']
    if ax == None: fig,axe = plt.subplots(nrows=1,ncols=1)
    else: axe = ax
    im = axe.pcolormesh(latsEdge,altsEdge,varRatio.T,cmap=cmap,vmin=0,vmax=vmax)
    axe.contour(lats,alts,extRatio.T,levels=(4,8,12,16))
    if ax == None: plt.colorbar(im)
    axe.set_ylim(ylim[0],ylim[1])
    axe.set_xlim(-35,20)
    axe.set_xlabel('Latitude')
    axe.set_ylabel('Altitude')
    axe.grid(True)
    axe.set_title(txt)
    # plt.show()
    return im
```

<!-- #region tags=[] -->
### Example of a plot
<!-- #endregion -->

```python
plot3(date(2022,2,27),vmax=3)
plt.show()
```

<!-- #region tags=[] -->
### Plots from 27 January onward
<!-- #endregion -->

```python
figsave = True
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = date(2022,1,27) + timedelta(days=2*j)
    try: im = plot3(day,ax=axes[j],txt=day.strftime('%d %b %Y'),cmap='gist_ncar',vmax=3)
    except: pass
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('OMPS 745 nm daily zonal std dev / mean extinction',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-1-OMPS-dilution.png',dpi=300,bbox_inches='tight')
plt.show()
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = date(2022,3,10) + timedelta(days=2*j)
    try: im = plot3(day,ax=axes[j],txt=day.strftime('%d %b %Y'),cmap='gist_ncar',vmax=3)
    except: pass
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('OMPS 745 nm daily zonal std dev / mean extinction',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-2-OMPS-dilution.png',dpi=300,bbox_inches='tight')
plt.show()
```

<!-- #region tags=[] -->
## Processing of MLS
<!-- #endregion -->

#### Preparation


##### Load both MLS and the zonal means from ERA5


MLS is restricted to the range 35S-20N after loading 

```python
with gzip.open('../HT-MLS/combinat-MLS-daily.pkl','rb') as f:
    MLS0 = pickle.load(f)
with gzip.open('../HT-Alien/zonDailyMean-all.pkl','rb') as f:
    zonal = pickle.load(f)
MLS = {}
MLS['attr'] = MLS0['attr']
MLS['data'] = {}
jy1 = np.where(MLS0['attr']['lats'] >= -35)[0][0]
jy2 = np.where(MLS0['attr']['lats'] <= 20)[0][-1]+1
MLS['attr']['lats'] = MLS0['attr']['lats'][jy1:jy2]
MLS['attr']['lats_edge'] = MLS0['attr']['lats_edge'][jy1:jy2+1]
for dd in MLS0['data']:
    MLS['data'][dd] = {}
    MLS['data'][dd]['meanWP'] = MLS0['data'][dd]['meanWP'][:,jy1:jy2]
    MLS['data'][dd]['npix'] = MLS0['data'][dd]['npix'][jy1:jy2]
del MLS0
```

##### Interpolate pressure to oversampling altitudes


The oversampling is made with a vertical resolution of 100 m using Akima interpolation in log pressure to provide a smooth vertical representation. No smoothing is applied in the temporal direction.
In the previous version, corresponding to the submitted paper, the interpolation was made to 1 km vertical resolution. As a result, the vertical motion of the water vapour plume, as detected from mas, mean or median was displaying jumps in the vertical that, in tun, generated bumps in the determined air velocity. See archived version 2.1 of this notebook (retrievable from tag v2.1 from github as md file).
In order to assess the equilibrium sulfate/water, we calculate two additional fields for the interpolated temperature and log pressure on the same grid as the interpolated MLS.

```python
# Latitude indexing and interpolation coefficient
# This works because the step in lat is 1° for ERA5
jl1 = []
jl2 = []
cl1 = []
cl2 = []
lats_mls = MLS['attr']['lats']
lats_mls_e = MLS['attr']['lats_edge']
logp_mls = np.log(MLS['attr']['press'])
for lat in lats_mls:
    jy = np.where(zonal['attr']['lats'] <= lat)[0][-1]
    jl1.append(jy)
    cl1.append(zonal['attr']['lats'][jy+1]-lat)
jl2 = [jl +1 for jl in jl1]
cl2 = [1 - cl for cl in cl1]
# Definition of the altitude grid with step 100m
# This generous oversampling is meant to smooth the vertical motion of MLS water vapour
# Unit is km
z_mls = MLS['attr']['alts_z'] = np.arange(18.05,30.,0.1)
ze_mls = MLS['attr']['alts_z_edge'] = np.arange(18.,30.05,0.1)
# Interpolation
for dd in MLS['data']:   
    day = date(dd.year,dd.month,dd.day)
    if day > date(2022,8,6): continue
    MLS['data'][dd]['WPZ'] = np.empty(shape=(len(lats_mls),len(z_mls)))
    MLS['data'][dd]['TZ'] = np.empty(shape=(len(lats_mls),len(z_mls)))
    MLS['data'][dd]['LPZ'] = np.empty(shape=(len(lats_mls),len(z_mls)))
    try:      
        for jy in range(len(lats_mls)):
            # Horizontal interpolation of the ERA5 data onto MLS grid
            # ERA5 vertical resolution is kept
            Z = cl1[jy] * zonal['mean'][day]['Z'][:,jl1[jy]] + cl2[jy] * zonal['mean'][day]['Z'][:,jl2[jy]] 
            θ = cl1[jy] * zonal['mean'][day]['PT'][:,jl1[jy]] + cl2[jy] * zonal['mean'][day]['PT'][:,jl2[jy]]
            T = cl1[jy] * zonal['mean'][day]['T'][:,jl1[jy]] + cl2[jy] * zonal['mean'][day]['T'][:,jl2[jy]]
            # Calculation of Log p
            logP = np.log(cst.p0) + np.log(T/θ)/cst.kappa
            # Vertical interpolation of -log p as a function of -Z in ERA5 
            # (-Z because Z ordered from top to bottom)
            inter1 = aki(-Z,-logP)
            # Vertical interpolation of MLS water vapour as a function of -log p 
            inter2 = aki(-logp_mls,MLS['data'][dd]['meanWP'][:,jy])
            # Using the two interpolations to determine the value of water vapour
            # in the extended MLS Z grid
            MLS['data'][dd]['WPZ'][jy,:] = inter2(inter1(-z_mls*1000))
            # Interpolation of T as a function of -Z
            interT = aki(-Z,T)
            # Using interT to determine T on the extended MLS grid
            MLS['data'][dd]['TZ'][jy,:] = interT(-z_mls*1000)
            # Using inter1 to determine log p on the extended MLS grid
            MLS['data'][dd]['LPZ'][jy,:] = - inter1(-z_mls*1000)
    except:
        print(day.strftime('missed %d %m'))
        continue
```

<!-- #region tags=[] -->
#### Dedicated plot function
<!-- #endregion -->

```python
def plotmls(dd,vmax=25,vmin=0,ax=None,txt=None,cmap='gist_ncar',ylim=(18,30),
           showlat=True,showalt=True,annot=False):
    if txt == None: txt = dd.strftime('MLS %d %b %Y (ppmv)')
    if ax == None: fig,axe = plt.subplots(nrows=1,ncols=1)
    else: axe = ax
    im = axe.pcolormesh(lats_mls_e,ze_mls,MLS['data'][dd]['WPZ'].T*1.e6,cmap=cmap,vmin=vmin,vmax=vmax)
    #axe.contour(lats_lms,z_mls,MLS['data'][dd]['WPZ']*1.e6,levels=(4,8,12,16))
    if ax == None: plt.colorbar(im)
    axe.set_ylim(ylim[0],ylim[1])
    axe.set_xlim(-35,20)
    if annot: axe.annotate("MLS",(-9,28.5), fontsize=9.5, color="black")
    if showlat==True: axe.set_xlabel('Latitude') 
    else : ax.set_xticklabels([])    
    if showalt==True: axe.set_ylabel('Altitude')
    else : ax.set_yticklabels([])
    axe.grid(True,color='dimgray')
    axe.set_title(txt)
    # plt.show()
    return im
```

<!-- #region tags=[] -->
#### Plot a few tests
<!-- #endregion -->

```python
plotmls(datetime(2022,1,31));plt.show()
plotmls(datetime(2022,5,25));plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
#### Multiple plots
<!-- #endregion -->

```python tags=[]
figsave = True
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = datetime(2022,1,27) + timedelta(days=2*j)
    im = plotmls(day,vmax=25,ax=axes[j],txt=day.strftime('%d %b %Y'))
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('MLS H20 daily zonal average mixing ratio (ppmv)',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-1-MLS.v4.png',dpi=300,bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = datetime(2022,4,21) + timedelta(days=2*j)
    im = plotmls(day,vmax=25,ax=axes[j],txt=day.strftime('%d %b %Y'))
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('MLS H20 daily zonal average mixing ratio (ppmv)',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-2-MLS.v4.png',dpi=300,bbox_inches='tight')
plt.show()

#fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
#axes = flatten(axes)
#for j in range(36):
#    day = datetime(2022,4,21) + timedelta(days=j)
#    im = plotmls(day,vmax=25,ax=axes[j],txt=day.strftime('%d %b %Y'))
#fig.subplots_adjust(top=0.8)
#cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
#fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
#fig.suptitle('MLS H20 daily zonal average mixing ratio (ppmv)',y=0.9,fontsize=24)
#if figsave: plt.savefig('HT-3-MLS.v4.png',dpi=300,bbox_inches='tight')
#plt.show()
```

<!-- #region tags=[] -->
### Selective row composite
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
#### Initial version until 20 May
<!-- #endregion -->

```python
figsave = True
fig, axes = plt.subplots(figsize=(21*8/7,11),ncols=8,nrows=3,sharex=True,sharey=True)
#axes = flatten(axes)
shift = range(0,128,16)
for j in range(8):
    day = date(2022,1,28) + timedelta(days=shift[j])
    dd = datetime(2022,1,28) + timedelta(days=shift[j])
    im1 = plot1(day,vmax=40,ax=axes[0,j],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar',showlat=False)
    im2 = plot2(day,ax=axes[1,j],txt=day.strftime('%d %b %Y'),cmap='gist_ncar',showlat=False)
    im3 = plotmls(dd,ax=axes[2,j],txt=day.strftime('%d %b %Y'))
fig.subplots_adjust(top=0.75,right=0.9)
cbar_ax1 = fig.add_axes([0.91, 0.565, 0.02, 0.185])
fig.colorbar(im1, cax=cbar_ax1,orientation='vertical')
cbar_ax2 = fig.add_axes([0.91, 0.345, 0.02, 0.185])
fig.colorbar(im2, cax=cbar_ax2,orientation='vertical')
cbar_ax3 = fig.add_axes([0.91, 0.125, 0.02, 0.185])
fig.colorbar(im3, cax=cbar_ax3,orientation='vertical')
fig.suptitle('OMPS 745 nm daily zonal average extinction ratio\n'+ \
             'CALIOP daily zonal average 532 nm attenuated scattering ratio\n'+ \
             'MLS H20 daily zonal average mixing ration (ppmv)',\
             y=0.88,fontsize=20)
if figsave: plt.savefig('HT-spe-1row-Combined_OMPS_CALIOP_MLS.png',dpi=300,bbox_inches='tight')
# much too long
#plt.savefig('Combined_OMPS-LP_CALIOP-1row.pdf',dpi=300,bbox_inches='tight')
plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
#### Extended version until 22 July
<!-- #endregion -->

```python
figsave = False
fig, axes = plt.subplots(figsize=(21*8/7,11),ncols=8,nrows=3,sharex=True,sharey=True)
#axes = flatten(axes)
shift = range(0,196,25)
for j in range(8):
    day = date(2022,1,28) + timedelta(days=shift[j])
    dd = datetime(2022,1,28) + timedelta(days=shift[j])
    im1 = plot1(day,vmax=40,ax=axes[0,j],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar',showlat=False)
    im2 = plot2(day,ax=axes[1,j],txt=day.strftime('%d %b %Y'),cmap='gist_ncar',showlat=False)
    im3 = plotmls(dd,ax=axes[2,j],txt=day.strftime('%d %b %Y'))
fig.subplots_adjust(top=0.75,right=0.9)
cbar_ax1 = fig.add_axes([0.91, 0.565, 0.02, 0.185])
fig.colorbar(im1, cax=cbar_ax1,orientation='vertical')
cbar_ax2 = fig.add_axes([0.91, 0.345, 0.02, 0.185])
fig.colorbar(im2, cax=cbar_ax2,orientation='vertical')
cbar_ax3 = fig.add_axes([0.91, 0.125, 0.02, 0.185])
fig.colorbar(im3, cax=cbar_ax3,orientation='vertical')
fig.suptitle('OMPS 745 nm daily zonal average extinction ratio\n'+ \
             'CALIOP daily zonal average 532 nm attenuated scattering ratio\n'+ \
             'MLS H20 daily zonal average mixing ration (ppmv)',\
             y=0.88,fontsize=20)
if figsave: plt.savefig('HT-spe-1row-Combined_OMPS_CALIOP_MLS.png',dpi=300,bbox_inches='tight')
# much too long
#plt.savefig('Combined_OMPS-LP_CALIOP-1row.pdf',dpi=300,bbox_inches='tight')
plt.show()
```

#### Revised version using CD code

```python
figsave = False

fig = plt.figure(constrained_layout=True,figsize=(12,16))
#axes = flatten(axes)
shift = range(0,192,19) #shift = range(0,140,14)
gs = fig.add_gridspec(7,5,height_ratios=(1,1,1,0.3,1,1,1))

for j in range(5):
    day = date(2022,1,28) + timedelta(days=shift[j])
    dd = datetime(2022,1,28) + timedelta(days=shift[j])
    if j == 0:
        ax = fig.add_subplot(gs[0,j])
        im1 = plot1(day,vmax=40,ax=ax,txt=day.strftime('%d %b %Y'),empir=False,
                    cmap='gist_ncar',showlat=False,showalt=True,annot=True)
        ax = fig.add_subplot(gs[1,j])
        im2 = plot2(day,ax=ax,cmap='gist_ncar',showlat=False,showalt=True,
                    annot=True,txt=' ')
        ax = fig.add_subplot(gs[2,j])
        im3 = plotmls(dd,ax=ax,showalt = True,annot=True,txt=' ')
    else :
        ax = fig.add_subplot(gs[0,j])
        im1 = plot1(day,vmax=40,ax=ax,txt=day.strftime('%d %b %Y'),
                    empir=False,cmap='gist_ncar',showlat=False,showalt=False)
        ax = fig.add_subplot(gs[1,j])
        im2 = plot2(day,ax=ax,cmap='gist_ncar',showlat=False,showalt=False,txt=' ')
        ax = fig.add_subplot(gs[2,j])
        im3 = plotmls(dd,ax=ax,txt=' ',showalt=False)

for j in range(5):
    day = date(2022,1,28) + timedelta(days=shift[j+5])
    dd = datetime(2022,1,28) + timedelta(days=shift[j+5])
    if j == 0 :
        ax = fig.add_subplot(gs[4,j])
        im1 = plot1(day,vmax=40,ax=ax,txt=day.strftime('%d %b %Y'),empir=False,
                    cmap='gist_ncar',showlat=False,showalt=True,annot=True)
        ax = fig.add_subplot(gs[5,j])
        im2 = plot2(day,ax=ax,cmap='gist_ncar',showlat=False,showalt=True,
                    annot=True,txt=' ')
        ax = fig.add_subplot(gs[6,j])
        im3 = plotmls(dd,ax=ax,showalt=True,annot=True,txt=' ')
    else :
        ax = fig.add_subplot(gs[4,j])
        im1 = plot1(day,vmax=40,ax=ax,txt=day.strftime('%d %b %Y'),
                    empir=False,cmap='gist_ncar',showlat=False,showalt=False)
        ax = fig.add_subplot(gs[5,j])
        im2 = plot2(day,ax=ax,cmap='gist_ncar',showlat=False,txt=' ',showalt=False)
        ax = fig.add_subplot(gs[6,j])
        im3 = plotmls(dd,ax=ax,txt=' ',showalt=False)
fig.subplots_adjust(top=0.81,right=0.9)
cbar_ax1 = fig.add_axes([0.91, 0.715, 0.02, 0.0965])
cb1 = fig.colorbar(im1, cax=cbar_ax1,orientation='vertical')
#cb1.ax.get_yaxis().labelpad = 10
cb1.ax.set_ylabel('745 nm aerosol\n extinction-ratio', rotation=90)
cbar_ax2 = fig.add_axes([0.91, 0.6055, 0.02, 0.0965])
cb2 = fig.colorbar(im2, cax=cbar_ax2,orientation='vertical')
cb2.ax.set_ylabel('532 nm aerosol\n backscatter-ratio', rotation=90)
cbar_ax3 = fig.add_axes([0.91, 0.496, 0.02, 0.0965])
cb3 = fig.colorbar(im3, cax=cbar_ax3,orientation='vertical')
cb3.ax.set_ylabel('water vapour (ppmv)', rotation=90)
cbar_ax4 = fig.add_axes([0.91, 0.342, 0.02, 0.0965])
cb4 = fig.colorbar(im1, cax=cbar_ax4,orientation='vertical')
cb4.ax.set_ylabel('745 nm aerosol\n extinction-ratio', rotation=90)
cbar_ax5 = fig.add_axes([0.91, 0.2325, 0.02, 0.0965])
cb5 = fig.colorbar(im2, cax=cbar_ax5,orientation='vertical')
cb5.ax.set_ylabel('532 nm aerosol\n backscatter-ratio', rotation=90)
cbar_ax6 = fig.add_axes([0.91, 0.123, 0.02, 0.0965])
cb6 = fig.colorbar(im3, cax=cbar_ax6,orientation='vertical')
cb6.ax.set_ylabel('water vapour (ppmv)', rotation=90)
fig.suptitle('OMPS-LP 745 nm daily zonal average aerosol extinction ratio\n'+ \
             'CALIOP daily zonal average 532 nm attenuated aerosol scattering ratio\n'+ \
             'MLS water vapour daily zonal average mixing ratio (ppmv)',\
             y=0.88,fontsize=16)
if figsave: plt.savefig('HT-spe-1raw-Combined_OMPS_CALIOP_MLSv3.png',dpi=300,bbox_inches='tight')
plt.show()
```

<!-- #region tags=[] -->
## Latitude average plots in the 35°S-20°N domain
<!-- #endregion -->

### OMPS


Here we generate the column of the extinction ratio for OMPS as an average in latitude between 35S and 20N. The product is stored in column_OMPS0

```python
days = []
for day in combinat['data']:
    if day < date(2022,1,27): continue
    days.append(day)
nd = len(days)
days_e = [datetime.combine(day-timedelta(days=1),time(12)) for day in days]
days_e.append(days_e[-1]+timedelta(hours=12))
column_OMPS0 = np.ma.asarray(np.full((nd,41),999999.))
cosfac = np.cos(np.deg2rad(combinat['attr']['lats']))
cosfac = cosfac/np.sum(cosfac)
jd = 0
for day in combinat['data']:
    if day < date(2022,1,27): continue
    column_OMPS0[jd,:] = np.ma.sum(cosfac[:,np.newaxis]*combinat['data'][day]['meanExtRatio'],axis=0)
    jd += 1
```

```python
fig, ax = plt.subplots()
xlims = mdates.date2num([days[0],days[-1]])
xx_e = mdates.date2num(days_e)
alts_edge = combinat['attr']['alts_edge']
#im=ax.imshow(column.T,cmap='gist_ncar',extent=(xlims[0],xlims[-1],0.,41),origin='lower',aspect=4)
im = ax.pcolormesh(xx_e,alts_edge,column_OMPS0.T,cmap='gist_ncar')
ax.set_ylim(18,30)
ax.xaxis_date()
date_format = mdates.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(date_format)
ax.set_ylabel('Altitude (km)') 
ax.set_xlabel('Day (in 2022)')
ax.set_title('OMPS 745 nm mean extinction ratio (35S-20N)')
plt.colorbar(im)
fig.autofmt_xdate()
```

### CALIOP


Here we generate the column of the scattering ratio for CLIOP as an average in latitude between 35S and 20N. The product is storeed in column_CALIOP0

```python
day0 = date(2022,1,27)
day1 = date(2022,8,9)
day = day0
day_e = datetime.combine(day0-timedelta(days=1),time(12))
nd = (day1-day0).days+1
nz = len(combinat_CALIOP['attr']['alts'])
cosfac = np.cos(np.deg2rad(combinat_CALIOP['attr']['lats']))
cosfac = cosfac/np.sum(cosfac)
column_CALIOP0 = np.ma.asarray(np.full((nd,nz),999999.))
jd = 0
days_e = []
while day <= day1:
    days_e.append(day_e)
    if day not in combinat_CALIOP['data']: 
        column_CALIOP0[jd,:] = np.ma.masked
    else:
        column_CALIOP0[jd,:] = np.ma.sum(cosfac[:,np.newaxis]*combinat_CALIOP['data'][day]['SR532'],axis=0)
    jd += 1
    day += timedelta(days=1)
    day_e += timedelta(days=1)
days_e.append(day_e)
```

```python
fig, ax = plt.subplots()
xx_e = mdates.date2num(days_e)
alts_edge = combinat_CALIOP['attr']['alts_edge']
im=ax.pcolormesh(xx_e,alts_edge,column_CALIOP0.T,cmap='gist_ncar',vmax=3)
ax.set_ylim(18,30)
ax.xaxis_date()
date_format = mdates.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(date_format)
ax.set_ylabel('Altitude (km)') 
ax.set_xlabel('Day (in 2022)')
ax.set_title('CALIOP 532 nm mean scattering ratio (35S-20N)')
plt.colorbar(im)
fig.autofmt_xdate()
```

<!-- #region tags=[] -->
## Total column plots
<!-- #endregion -->

Here we sum the column in the 18-30 km


### OMPS


The product is contained in column_OMPS and is a nondimensional AOD

```python
days = []
for day in combinat['data']:
    if day < date(2022,1,27): continue
    days.append(day)
nd = len(days)
# edge dates
days_oe = [datetime.combine(day-timedelta(days=1),time(12)) for day in days]
days_oe.append(days_oe[-1]+timedelta(hours=12))
# centerd dates 
days_o = days
column_OMPS = np.ma.asarray(np.full((nd,nlat),999999.))
jd = 0
for day in combinat['data']:
    if day < date(2022,1,27): continue
    # The sum can be performed in this way because the vertical resolution is 1 km and 
    # the extinction is in km**-1    
    column_OMPS[jd,:] = np.ma.sum(combinat['data'][day]['meanExt'][:,18:30],axis=1)
    if day == date(2022,7,5): 
        column_OMPS[jd,:] = np.ma.masked
        print(jd)
    jd += 1
```

```python
fig, ax = plt.subplots()
xlims = mdates.date2num([days[0],days[-1]])
xx_oe = mdates.date2num(days_oe)
lats_edge = combinat['attr']['lats_edge']
#im=ax.imshow(column.T,cmap='gist_ncar',extent=(xlims[0],xlims[-1],0.,41),origin='lower',aspect=4)
im = ax.pcolormesh(xx_oe,lats_edge,column_OMPS.T,cmap='gist_ncar',vmax=0.04)
ax.xaxis_date()
date_format = mdates.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(date_format)
ax.set_ylabel('Latitude') 
ax.set_xlabel('Day (in 2022)')
ax.set_title('OMPS 745 nm total extinction (18-30km)')
plt.colorbar(im)
fig.autofmt_xdate()
```

Latitude average extinction

```python
coslat = np.cos(np.deg2rad(combinat['attr']['lats']))
weights = coslat/np.sum(coslat)
OMOD = np.ma.sum(column_OMPS*weights[np.newaxis,:],axis=1)
xxo = mdates.date2num(days_o)
fig, ax = plt.subplots()
ax.plot(xxo,OMOD)
ax.xaxis_date()
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
ax.set_xlabel('Day (in 2022)')
ax.set_title('Mean OMPS-LP OD 18-30 km 20°N-35°S')
ax.set_ylabel('Optical depth')
```

### CALIOP


The product is CALIOP_column witch is the integral of the attenuated scattering with dimension sr**-1

```python
day0 = date(2022,1,27)
day1 = date(2022,8,9)
day = day0
day_e = datetime.combine(day0-timedelta(days=1),time(12))
nd = (day1-day0).days+1
ny = len(combinat_CALIOP['attr']['lats'])
cosfac = np.cos(np.deg2rad(combinat_CALIOP['attr']['lats']))
cosfac = cosfac/np.sum(cosfac)
column_CALIOP = np.ma.asarray(np.full((nd,ny),999999.))
jd = 0
# Centered and edge dates
days_c = []
days_ce = []
hh = combinat_CALIOP['attr']['alts_edge']
dz = hh[:-1]-hh[1:]
while day <= day1:
    days_c.append(day)
    days_ce.append(day_e)
    if day not in combinat_CALIOP['data']: 
        column_CALIOP[jd,:] = np.ma.masked
    else:
        column_CALIOP[jd,:] = np.ma.sum(combinat_CALIOP['data'][day]['T532']*dz[np.newaxis,:],axis=1)
    jd += 1
    day += timedelta(days=1)
    day_e += timedelta(days=1)
days_ce.append(day_e)
```

```python
fig, ax = plt.subplots()
xx_ce = mdates.date2num(days_ce)
lats_edge = combinat_CALIOP['attr']['lats_edge']
im=ax.pcolormesh(xx_ce,lats_edge,column_CALIOP.T,cmap='gist_ncar')
ax.xaxis_date()
date_format = mdates.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(date_format)
ax.set_ylabel('Latitude') 
ax.set_xlabel('Day (in 2022)')
ax.set_title('CALIOP 532 nm total attenuated scattering (18-30 km)')
plt.colorbar(im)
fig.autofmt_xdate()
```

### Ratio total extinction to total backscatter


We assume here that the shortest time series is provided by OMPS

```python
# First the OMPS column is interpolated in latitude to match the CALIOP resolution
smooth_OMPS = np.ma.zeros(shape=(column_OMPS.shape[0],column_CALIOP.shape[1]))
nbdays_OMPS = smooth_OMPS.shape[0]
for jd in range(smooth_OMPS.shape[0]):
    interp = interp1d(combinat['attr']['lats'],column_OMPS[jd,:],kind='slinear',fill_value='extrapolate')
    smooth_OMPS[jd,:] = np.reshape(interp(combinat_CALIOP['attr']['lats']),183)
    if jd == 159: smooth_OMPS[jd,:] = np.ma.masked
fig, ax = plt.subplots()
xxoe = mdates.date2num(days_oe)
lats_edge = combinat_CALIOP['attr']['lats_edge']
im=ax.pcolormesh(xxoe,lats_edge,(smooth_OMPS/column_CALIOP[:nbdays_OMPS,:]).T,cmap='gist_ncar',vmax=40)
ax.xaxis_date()
date_format = mdates.DateFormatter('%b-%d')
ax.xaxis.set_major_formatter(date_format)
ax.set_ylabel('Latitude') 
ax.set_xlabel('Day (in 2022)')
ax.set_title('ratio total 745nm OMPS extinction to total 532nm CALIOP attenuated scattering (18-30 km) (str)')
plt.colorbar(im)
fig.autofmt_xdate()
plt.show()

# save the two fields for usage in composite figure with IMS
attribs = {'lats':combinat_CALIOP['attr']['lats'],'lats_edge':lats_edge,
           'days':days_c,'days_edge':days_ce}
with gzip.open('integrated_CALIOP_OMPS.pkl','wb') as f:
    pickle.dump([column_CALIOP,smooth_OMPS,attribs],f,protocol=5)
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
##### Obsolete (and not updates to August)
<!-- #endregion -->

```python
# Obsolete
# Mean in 15S-25S & 05S-15S & 
# No cosine weighting here
lats = combinat_CALIOP['attr']['lats']
fig, axs = plt.subplots(figsize=(16,3.5),nrows=1,ncols=3)
xx_c = mdates.date2num(days_c)
date_format = mdates.DateFormatter('%b-%d')
jy1 = np.where(lats >= -25)[0][0]
jy2 = np.where(lats > -15)[0][0]
axs[1].plot(xx_c,np.mean(smooth_OMPS[:,jy1:jy2],axis=1))
axs[0].plot(xx_c,np.mean(column_CALIOP[:,jy1:jy2],axis=1))
axs[2].plot(xx_c,np.mean(smooth_OMPS[:,jy1:jy2],axis=1)/np.mean(column_CALIOP[:,jy1:jy2],axis=1))
jy1 = np.where(lats >= -15)[0][0]
jy2 = np.where(lats > -5)[0][0]
axs[1].plot(xx_c,np.mean(smooth_OMPS[:,jy1:jy2],axis=1))
axs[0].plot(xx_c,np.mean(column_CALIOP[:,jy1:jy2],axis=1))
axs[2].plot(xx_c,np.mean(smooth_OMPS[:,jy1:jy2],axis=1)/np.mean(column_CALIOP[:,jy1:jy2],axis=1))
jy1 = 0
jy2 = column_CALIOP.shape[1]
axs[1].plot(xx_c,np.mean(smooth_OMPS[:,jy1:jy2],axis=1))
axs[0].plot(xx_c,np.mean(column_CALIOP[:,jy1:jy2],axis=1))
axs[2].plot(xx_c,np.mean(smooth_OMPS[:,jy1:jy2],axis=1)/np.mean(column_CALIOP[:,jy1:jy2],axis=1))
for i in range(3):
    axs[i].xaxis_date()
    axs[i].xaxis.set_major_formatter(date_format)
axs[1].set_ylabel('AOD')
axs[1].set_title('OMPS 745nm AOD')
axs[0].set_ylabel('Total BS (str**-1)')
axs[0].set_title('CALIOP integrated attenuated 532nm backscatter')
axs[2].set_ylabel('AOD / Total BS (str)')
axs[2].set_title('"lidar ratio" from OMPS 745 nm & CALIOP 532nm')
fig.autofmt_xdate()
fig.subplots_adjust(top=0.83)
fig.suptitle('15S-25S (blue), 05S-15S (orange) and 35S-20N (green) average of the column integrated quantities between 18 and 30 km')
fig.savefig('lidarRatio.png',dpi=144,bbox_inches='tight')
plt.show()
```

#### Correct version

```python
# Mean in 15S-25S & 05S-15S & 
# Same with cosine weighting (does not seem to change anything visible)
figsave = False
lats = combinat_CALIOP['attr']['lats']
cosfac = np.cos(np.deg2rad(lats))
fig, axs = plt.subplots(figsize=(16,3.5),nrows=1,ncols=3)
xx_o = mdates.date2num(days_o)
date_format = mdates.DateFormatter('%b-%d')
jy1 = np.where(lats >= -25)[0][0]
jy2 = np.where(lats > -15)[0][0]
factor = cosfac[jy1:jy2]/np.sum(cosfac[jy1:jy2])
axs[1].plot(xx_o,np.ma.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1))
axs[0].plot(xx_o,np.ma.sum(column_CALIOP[:nbdays_OMPS,jy1:jy2]*factor[np.newaxis,:],axis=1))
axs[2].plot(xx_o,np.ma.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1)/
            np.ma.sum(column_CALIOP[:nbdays_OMPS,jy1:jy2]*factor[np.newaxis,:],axis=1))
jy1 = np.where(lats >= -15)[0][0]
jy2 = np.where(lats > -5)[0][0]
factor = cosfac[jy1:jy2]/np.sum(cosfac[jy1:jy2])
axs[1].plot(xx_o,np.ma.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1))
axs[0].plot(xx_o,np.ma.sum(column_CALIOP[:nbdays_OMPS,jy1:jy2]*factor[np.newaxis,:],axis=1))
axs[2].plot(xx_o,np.ma.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1)/
            np.ma.sum(column_CALIOP[:nbdays_OMPS,jy1:jy2]*factor[np.newaxis,:],axis=1))
jy1 = 0
jy2 = column_CALIOP.shape[1]
factor = cosfac[jy1:jy2]/np.sum(cosfac[jy1:jy2])
axs[1].plot(xx_o,np.ma.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1))
axs[0].plot(xx_o,np.ma.sum(column_CALIOP[:nbdays_OMPS,jy1:jy2]*factor[np.newaxis,:],axis=1))
axs[2].plot(xx_o,np.ma.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1)/
            np.ma.sum(column_CALIOP[:nbdays_OMPS,jy1:jy2]*factor[np.newaxis,:],axis=1))
for i in range(3):
    axs[i].xaxis_date()
    axs[i].xaxis.set_major_formatter(date_format)
axs[1].set_ylabel('AOD')
axs[1].set_title('OMPS 745nm AOD')
axs[0].set_ylabel('Total BS (str**-1)')
axs[0].set_title('CALIOP integrated attenuated 532nm backscatter')
axs[2].set_ylabel('AOD / Total BS (str)')
axs[2].set_title('"lidar ratio" from OMPS 745 nm & CALIOP 532nm')
fig.autofmt_xdate()
fig.subplots_adjust(top=0.83)
fig.suptitle('15S-25S (blue), 05S-15S (orange) and 35S-20N (green) average of the column integrated quantities between 18 and 30 km')
if figsave: fig.savefig('lidarRatio.png',dpi=144,bbox_inches='tight')
plt.show()
```

CHECK the problem in July and remove this point if spurious

<!-- #region tags=[] -->
## Composite section plot of CALIOP with MLS
<!-- #endregion -->

Products are in column-CALIOP1 and column_MLS1

```python
# Definitions of latitude sections
secs = {'all','15-25','05-15'}
lats_CALIOP = combinat_CALIOP['attr']['lats']
lats_MLS = MLS['attr']['lats']
jy1 = {'CALIOP':{'all':0,'15-25':np.where(lats_CALIOP >= -25)[0][0],'05-15':np.where(lats_CALIOP >= -15)[0][0]},
       'MLS':{'all':0,'15-25':np.where(lats_MLS >= -25)[0][0],'05-15':np.where(lats_MLS >= -15)[0][0]}}
jy2 = {'CALIOP':{'all':len(lats_CALIOP),'15-25':np.where(lats_CALIOP > -15)[0][0],'05-15':np.where(lats_CALIOP > -5)[0][0]},
       'MLS':{'all':len(lats_MLS),'15-25':np.where(lats_MLS >= -15)[0][0],'05-15':np.where(lats_MLS >= -5)[0][0]}}
# Generation of the temporal vector
day0 = date(2022,1,27)
day1 = date(2022,8,6)
day = day0
day_e = datetime.combine(day0-timedelta(days=1),time(12))
nd = (day1-day0).days+1
days_e = []
days = []
while day <= day1:
    days.append(day)
    days_e.append(day_e)
    day += timedelta(days=1)
    day_e += timedelta(days=1)
days_e.append(day_e)

# Processing of CALIOP data
nz = len(combinat_CALIOP['attr']['alts'])
column_CALIOP1 = {}
for sec in secs:
    column_CALIOP1[sec] = np.ma.asarray(np.full((nd,nz),999999.))
jd = 0
day = day0
while day <= day1:
    if day not in combinat_CALIOP['data']:
        for sec in secs:
            column_CALIOP1[sec][jd,:] = np.ma.masked
    else:
        for sec in secs:
            j1 = jy1['CALIOP'][sec]
            j2 = jy2['CALIOP'][sec]
            cosfac = np.cos(np.deg2rad(combinat_CALIOP['attr']['lats'][j1:j2]))
            cosfac = cosfac/np.sum(cosfac)           
            column_CALIOP1[sec][jd,:] = np.ma.sum(cosfac[:,np.newaxis]*combinat_CALIOP['data'][day]['SR532'][j1:j2,:],axis=0)
    jd += 1
    day += timedelta(days=1)
     
# Processing of MLS data
nz = len(MLS['attr']['alts_z'])
column_MLS1 = {}
for sec in secs:
    column_MLS1[sec] = np.ma.asarray(np.full((nd,nz),999999.))
jd = 0
day = day0
while day <= day1:
    dd = datetime(day.year,day.month,day.day)
    if dd not in MLS['data']:
        for sec in secs:
            column_MLS1[sec][jd,:] = np.ma.masked
    else:
        for sec in secs:
            j1 = jy1['MLS'][sec]
            j2 = jy2['MLS'][sec]
            cosfac = np.cos(np.deg2rad(MLS['attr']['lats'][j1:j2]))
            cosfac = cosfac/np.sum(cosfac)
            column_MLS1[sec][jd,:] = np.ma.sum(cosfac[:,np.newaxis]*MLS['data'][dd]['WPZ'][j1:j2,:],axis=0)
    jd += 1
    day += timedelta(days=1)
```

```python
from scipy.ndimage import gaussian_filter
vmax = {'all':3,'15-25':6,'05-15':6}
for sec in secs:
    fig, ax = plt.subplots(figsize=(12,4))
    xxe = mdates.date2num(days_e)
    alts_edge = combinat_CALIOP['attr']['alts_edge']
    im=ax.pcolormesh(xxe,alts_edge,column_CALIOP1[sec].T,cmap='gist_ncar',vmax=vmax[sec],vmin=0)
    xx = mdates.date2num(days)
    alts_mls = MLS['attr']['alts_z']
    CS = ax.contour(xx,alts_mls,gaussian_filter(column_MLS1[sec],1).T*1.e6,linewidths=3,colors='pink')
    #CS = ax.contour(xx,alts_mls,column_MLS1[sec].T*1.e6,linewidths=3,colors='pink')
    ax.clabel(CS,inline=1,fontsize=12)
    ax.set_ylim(20,28)
    ax.xaxis_date()
    date_format = mdates.DateFormatter('%d-%b')
    ax.xaxis.set_major_formatter(date_format)
    ax.set_ylabel('Altitude (km)') 
    ax.set_xlabel('Day in 2022')
    ax.grid(True)
    ax.set_title('CALIOP 532 nm scattering ratio + MLS water vapour in zonal band '+sec+'S')
    plt.colorbar(im)
    fig.autofmt_xdate()
    fig.savefig('scattering-MLS-profile-'+sec+'.png',dpi=144,bbox_inches='tight')
    plt.show()
attribs1 = {'alts_CALIOP':combinat_CALIOP['attr']['alts'],'alts_edge_CALIOP':alts_edge,
           'alts_MLS':alts_mls,'alts_edge_MLS':MLS['attr']['alts_z_edge'],
           'days':days,'days_edge':days_e,'numdays':xx,'numdays-edge':xxe}
with gzip.open('colonnes.pkl','wb') as f:
    pickle.dump([column_CALIOP1,column_MLS1,attribs1],f,protocol=pickle.HIGHEST_PROTOCOL)
```

<!-- #region tags=[] -->
### Calculation of a fit for CALIOP sections
<!-- #endregion -->

Calculation of the mean, max & median. Use of column_CALIOP1

```python
center_CALIOP={}
alts = combinat_CALIOP['attr']['alts']
# Blacklisted dates (because too less orbits)
blacklist = [date(2022,3,31),]
secs = ('05-15','15-25','all')
offset = {'15-25':1.5, '05-15':1.5, 'all':1}
days = []
for sec in secs:
    print(sec)
    center_CALIOP[sec] = {"max":np.ma.zeros(column_CALIOP1['15-25'].shape[0]),
                         "median":np.ma.zeros(column_CALIOP1['15-25'].shape[0]),
                          "mean":np.ma.zeros(column_CALIOP1['15-25'].shape[0])}
    jd = 0
    day0 = date(2022,1,27)
    day1 = date(2022,8,6)
    day = day0
    while day <= day1:
        if sec == 'all': days.append(day)
        col = column_CALIOP1[sec][jd,:].copy()
        col [col<offset[sec]] = np.ma.masked
        if all(col.mask) | (day in blacklist):
            print('masked ',day,sec)
            center_CALIOP[sec]["max"][jd] = np.ma.masked
            center_CALIOP[sec]["median"][jd] = np.ma.masked
            center_CALIOP[sec]["mean"][jd] = np.ma.masked
            jd += 1
            day += timedelta(days=1)
            continue
        center_CALIOP[sec]["max"][jd] = alts[np.ma.argmax(col)]
        center_CALIOP[sec]["mean"][jd] = np.ma.sum(col * alts)/np.ma.sum(col)
        cs = np.ma.cumsum(col)/np.ma.sum(col)
        idx = np.ma.where(cs>0.5)[0][0]
        # fix for pb if cs[idx-1] not defined (met once)
        dec=1
        while cs.mask[idx-dec]: dec += 1
        if dec>1:
            print(day,jd,sec,idx,dec)
        p = (cs[idx]-0.5)/(cs[idx]-cs[idx-dec])
        center_CALIOP[sec]["median"][jd] = (1-p)*alts[idx] + p*alts[idx-dec]
        jd += 1
        day += timedelta(days=1)

fig = plt.figure(figsize=(15,3))
xx = mdates.date2num(days)
jg = 0
date_format = mdates.DateFormatter('%d-%b')    
for sec in secs:
    jg += 1
    ax = plt.subplot(1,3,jg)
    CS = ax.contour(xx,combinat_CALIOP['attr']['alts'],column_CALIOP1[sec].T,linewidths=1,colors='black')
    ax.plot(xx,center_CALIOP[sec]["max"],xx,center_CALIOP[sec]["mean"],xx,center_CALIOP[sec]["median"])
    ax.set_title('CALIOP '+sec)
    ax.xaxis_date()   
    ax.xaxis.set_major_formatter(date_format)
    ax.legend(('max','mean','median'))
    ax.set_ylim(20.,26.5)
plt.show
fig.autofmt_xdate()
```

### Sections of CALIOP at a series of dates 

```python
alts_c = combinat_CALIOP['attr']['alts']
dz = alts_c[1:]-alts_c[:-1]
```

```python
fig = plt.figure(figsize=(15,3))
xx = mdates.date2num(days)
jg = 0
date_format = mdates.DateFormatter('%b-%d')    
for sec in secs:
    jg += 1
    ax = plt.subplot(1,3,jg)
    for d in range(20,len(xx),20):
        ax.plot(alts_c,column_CALIOP1[sec][d,:])
    #CS = ax.contour(xx,combinat_CALIOP['attr']['alts'],column_CALIOP1[sec].T,linewidths=1,colors='black')
    #ax.plot(xx,center_CALIOP[sec]["max"],xx,center_CALIOP[sec]["mean"],xx,center_CALIOP[sec]["median"])
    ax.set_title('CALIOP '+sec)
    #ax.xaxis_date()   
    #ax.xaxis.set_major_formatter(date_format)
    #ax.set_ylim(20.,26.5)
plt.show
```

### Calculation of descent rates from the median using Savitsky-Golay filter


Check where to make the cut


#### Make the fit for the three sections


##### Corrected version

```python
wlist = [11,21,31,41]
fo = 2
mode = 'interp'
l1 = 60
l2 = 64
ff0 = {}
ff1 = {}
#ff1b = {}
for wl in wlist:
    ff0[wl] = {}
    ff1[wl] = {}
    #ff1b[wl] = {}
center = "mean"

center_CALIOP_corr = {}
for sec in secs:
    center_CALIOP_corr[sec] = {}
    center_CALIOP_corr[sec][center] = np.zeros(len(center_CALIOP[sec][center]))
    center_CALIOP_corr[sec][center][:l1] = center_CALIOP[sec][center][:l1].copy()
    center_CALIOP_corr[sec][center][l2:] = center_CALIOP[sec][center][l2:].copy()
    for l in range(l1,l2):
        p = (l-l1+1)/(l2-l1+1)
        center_CALIOP_corr[sec][center][l] = (1-p) * center_CALIOP[sec][center][l2] + p * center_CALIOP[sec][center][l1-1]

for wl in wlist:
    for sec in secs:
        # ff0 fit to the function and ff1 to its derivative
        ff0[wl][sec] = np.ma.zeros(len(center_CALIOP_corr[sec][center]))
        ff1[wl][sec] = np.ma.zeros(len(center_CALIOP_corr[sec][center]))
        #ff1b[sec] = np.ma.zeros(len(center_CALIOP_corr[sec][center]))
        ff0[wl][sec] = sps.savgol_filter(center_CALIOP_corr[sec][center],wl,fo,deriv=0,mode=mode)
        ff1[wl][sec] = sps.savgol_filter(center_CALIOP_corr[sec][center],wl,fo,deriv=1,mode=mode)
        #ff1b[sec][1:-1] = 0.5*(ff0[sec][2:] - ff0[sec][:-2])
        #ff1b[sec][0] = ff0[sec][1]-ff0[sec][0]
        #ff1b[sec][-1] = ff0[sec][-1]-ff0[sec][-2]
        # Conversion of the derivative m/day
        ff1[wl][sec] *= 1000
        #ff1b[sec] *= 1000
    

fig = plt.figure(figsize=(15,6))
xx = mdates.date2num(days)
jg = 0
date_format = mdates.DateFormatter('%d-%b')    
for sec in secs:
    jg += 1
    ax0 = plt.subplot(2,3,jg)
    ax1 = plt.subplot(2,3,jg+3)
    ax0.plot(xx,center_CALIOP_corr[sec][center])
    ax1.plot(xx,ff1[11][sec])
    for wl in wlist:
        ax0.plot(xx,ff0[wl][sec])
        ax1.plot(xx,ff1[wl][sec])
    ax0.set_title('CALIOP '+sec)
    ax0.xaxis_date()   
    ax0.xaxis.set_major_formatter(date_format)   
    ax1.xaxis_date()   
    ax1.xaxis.set_major_formatter(date_format)
    ax0.grid(True)
    ax1.grid(True)
    ax0.set_ylim(22,26.5)
    ax1.set_ylim(-50,50)
    ax0.legend(['mean center','wl = 11','wl = 21','wl = 31','wl = 41'])
    ax1.legend(['','wl = 11','wl = 21','wl = 31','wl = 41'])
fig.autofmt_xdate()
```

```python
[vals,nbins,_]=plt.hist(np.reshape(column_CALIOP['15-25'],93*120),bins=100)
```

#### Ambiant heating rate compensation and combining

```python
with gzip.open(os.path.join('..','HT-Alien','sections_ERA5.pkl'),'rb') as f:
    [sections,dd] = pickle.load(f)
xx_era5 = mdates.date2num(dd['days'])
```

```python
sections[sec].keys()
```

```python
# Achtung: this uses only the 31-day filter for ff0

wdiab = {}
wadiab = {}
wl = 31
for sec in secs:
    wdiab[sec] = np.ma.zeros(shape=ff0[wl][sec].shape)
    wadiab[sec] = np.ma.zeros(shape=ff0[wl][sec].shape)
    jd = 0
    day = day0
    blacklist = [date(2022,3,31),]
    while day <= day1:
        #if ff0[sec].mask[jd] | (day in blacklist):
        #    wdiab[sec][jd] = np.ma.masked
        #    wadiab[sec][jd] = np.ma.masked
        #    jd += 1
        #    day += timedelta(days=1)
        id = np.where(mdates.date2num(day) == xx_era5)[0][0]
        Z = sections[sec]['Z'][id,:]
        DZDt = gaussian_filter(sections[sec]['DZDt'][id,25:50],3)
        DZDtAdiab =  gaussian_filter(sections[sec]['DZDtAdiab2'][id,25:50],3)
        wdiab[sec][jd] = np.interp(ff0[wl][sec][jd],Z[25:50],DZDt)
        wadiab[sec][jd] = np.interp(ff0[wl][sec][jd],Z[25:50],DZDtAdiab)
        jd += 1
        day += timedelta(days=1)  
```

### Figures from CALIOP and MLS


ACHTUNG ACHTUNG: MLS slope calculated in the next subsection must be available for this plot

```python
# Version grand public
fig = plt.figure(figsize=(16,5))
xx = mdates.date2num(days)
jg = 0
wl = 31
date_format = mdates.DateFormatter('%d-%b')    
for sec in secs:
    jg += 1
    ax0 = plt.subplot(1,3,jg)
    ax0.plot(xx,ff1[wl][sec],xx,wdiab[sec],xx,wdiab[sec]+wadiab[sec],
             xx,ff1[wl][sec]-wdiab[sec]-wadiab[sec],
             xx,ff1_mls[wl][sec],xx,ff1_mls[wl][sec]-wdiab[sec]-wadiab[sec],linewidth=3)
    ax0.set_title('CALIOP vertical motion '+sec+'S')
    ax0.set_ylabel('w (m per day)')
    ax0.xaxis_date()   
    ax0.xaxis.set_major_formatter(date_format)
    ax0.legend(['$w_{CALIOP}$','$w_R$','$w_{ERA5}$','$w_{CALIOP}^{air}$',
                '$w_{MLS}$','$w_{MLS}^{air}$'],fontsize=12)
    ax0.set_ylim(-130,40)
    ax0.grid(True)
fig.autofmt_xdate()
plt.show()
```

```python
Aerosol radius from the fall speed 
```

```python
rada = {}
mu = 1.45e-5
rho = 1000
Cc = 1
for wl in wlist:
    rada[wl] = {}
    for sec in secs:
        rada[wl][sec] = np.sqrt(np.clip(-18*mu*(ff1[wl][sec]-wdiab[sec]-wadiab[sec])/(86400*cst.g*Cc*rho),0,1000))*1e6/2
        rada[wl][sec] = np.ma.array(rada[wl][sec])
        rada[wl][sec][rada[wl][sec]<=0] = np.ma.masked
```

```python
fig,ax = plt.subplots(figsize=(16,4),ncols=4)
for n in range(4):
    ax[n].plot(xx,rada[wlist[n]]['15-25'],xx,rada[wlist[n]]['05-15'],linewidth=3)
    ax[n].xaxis_date()   
    ax[n].xaxis.set_major_formatter(date_format)
    ax[n].set_ylabel('Aerosol radius (µm)')
    ax[n].legend(('15-25 S','05-15 S'),fontsize=12)
    ax[n].grid(True)
    ax[n].set_ylim(0,3.5)
    fig.autofmt_xdate()
```

### MLS filtering

```python
center_MLS={}
alts_mls = MLS['attr']['alts_z']
secs = ('05-15','15-25','all')
offset = {'15-25':6.e-6, '05-15':6.e-6, 'all':5.e-6}
days = []
for sec in secs:
    print(sec)
    center_MLS[sec] = {"max":np.ma.zeros(column_MLS1['15-25'].shape[0]),
                       "median":np.ma.zeros(column_MLS1['15-25'].shape[0]),
                       "mean":np.ma.zeros(column_MLS1['15-25'].shape[0])}
    jd = 0
    # The days must be the same as for the calculation of column_MLS1 above
    day0 = date(2022,1,27)
    day1 = date(2022,8,6)
    day = day0
    # It does not make a big diff to smooth the data
    #buf = np.ma.array(gaussian_filter(column_MLS1[sec],1))
    buf = column_MLS1[sec]
    while day <= day1:
        if sec == 'all': days.append(day)
        col = buf[jd,:].copy()
        
        col [col<offset[sec]] = np.ma.masked
        if all(col.mask):
            print('masked ',day,sec)
            center_MLS[sec]["max"][jd] = np.ma.masked
            center_MLS[sec]["median"][jd] = np.ma.masked
            center_MLS[sec]["mean"][jd] = np.ma.masked
            jd += 1
            day += timedelta(days=1)
            continue
        center_MLS[sec]["max"][jd] = alts_mls[np.ma.argmax(col)]
        center_MLS[sec]["mean"][jd] = np.ma.sum(col * alts_mls)/np.ma.sum(col)
        cs = np.ma.cumsum(col)/np.ma.sum(col)
        idx = np.ma.where(cs>0.5)[0][0]
        # fix for pb if cs[idx-1] not defined (met once)
        dec=1
        while cs.mask[idx-dec]: dec += 1
        if dec>1:
            print(day,jd,sec,idx,dec)
        p = (cs[idx]-0.5)/(cs[idx]-cs[idx-dec])
        center_MLS[sec]["median"][jd] = (1-p)*alts[idx] + p*alts[idx-dec]
        jd += 1
        day += timedelta(days=1)

fig = plt.figure(figsize=(15,3))
xx = mdates.date2num(days)
jg = 0
date_format = mdates.DateFormatter('%d-%b')
for sec in secs:
    buf = column_MLS1[sec]
    jg += 1
    ax = plt.subplot(1,3,jg)
    CS = ax.contour(xx,alts_mls,buf.T*1.e6,linewidths=1,colors='black')
    ax.plot(xx,center_MLS[sec]["max"],xx,center_MLS[sec]["mean"],xx,center_MLS[sec]["median"])
    ax.set_title('MLS '+sec)
    ax.xaxis_date()   
    ax.xaxis.set_major_formatter(date_format)
    ax.legend(('max','mean','median'))
    ax.set_ylim(23,28)
plt.show
fig.autofmt_xdate()
```

```python
wlist = [11,21,31,41]
fo = 2
mode = 'interp'
l1 = 60
l2 = 64
ff0_mls = {}
ff1_mls = {}
#ff1b_mls = {}
for wl in wlist:
    ff0_mls[wl] = {}
    ff1_mls[wl] = {}
    #ff1b_mls[wl] = {}
center = "mean"

for wl in wlist:
    for sec in secs:
        # ff0 fit to the function and ff1 to its derivative
        ff0_mls[wl][sec] = np.ma.zeros(len(center_MLS[sec][center]))
        ff1_mls[wl][sec] = np.ma.zeros(len(center_MLS[sec][center]))
        #ff1b_mls[wl][sec] = np.ma.zeros(len(center_MLS[sec][center]))
        ff0_mls[wl][sec] = sps.savgol_filter(center_MLS[sec][center],wl,fo,deriv=0,mode=mode)
        ff1_mls[wl][sec] = sps.savgol_filter(center_MLS[sec][center],wl,fo,deriv=1,mode=mode)
        #ff1b_mls[wl][sec][1:-1] = 0.5*(ff0_mls[wl][sec][2:] - ff0_mls[wl][sec][:-2])
        #ff1b_mls[wl][sec][0] = ff0_mls[wl][sec][1]-ff0_mls[wl][sec][0]
        #ff1b_mls[wl][sec][-1] = ff0_mls[wl][sec][-1]-ff0_mls[wl][sec][-2]
        # Conversion of the derivative m/day
        ff1_mls[wl][sec] *= 1000
        #ff1b_mls[wl][sec] *= 1000
    
fig = plt.figure(figsize=(15,6))
xx = mdates.date2num(days)
jg = 0
date_format = mdates.DateFormatter('%d-%b')    
for sec in secs:
    jg += 1
    ax0 = plt.subplot(2,3,jg)
    ax1 = plt.subplot(2,3,jg+3)
    ax0.plot(xx,center_MLS[sec][center])
    ax1.plot(xx,wdiab[sec]+wadiab[sec])
    for wl in wlist:
        ax0.plot(xx,ff0_mls[wl][sec])       
        ax1.plot(xx,ff1_mls[wl][sec])
    ax0.set_title('MLS '+sec)
    ax0.xaxis_date()   
    ax0.xaxis.set_major_formatter(date_format)
    ax1.xaxis_date()   
    ax1.xaxis.set_major_formatter(date_format)
    ax0.set_ylim(24,26.5)
    ax1.set_ylim(-50,30)
    ax0.grid(True)
    ax1.grid(True)
    ax0.legend(['mean center','wl = 11','wl = 21','wl = 31','wl = 41'])
    ax1.legend(['wERA5','wl = 11','wl = 21','wl = 31','wl = 41'])
fig.autofmt_xdate()
```

### Figure 2 komposit


ACHTUNG: This composit requires the preparation scripts of all the components to have run previously
The time series are truncated on the end date 

```python
savefig = True
enddate = date(2022,7,26)
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig = plt.figure(constrained_layout=True,figsize=(16,12))
fs = 14
fsl = 14
wl = 31
gs0 = fig.add_gridspec(3,1)
gs1 = gs0[0].subgridspec(1,3,width_ratios=[9,0.3,5])
gs2 = gs0[1].subgridspec(1,3,width_ratios=[9,0.3,5])
gs3 = gs0[2].subgridspec(1,5,width_ratios=[3,0.2,3,0.2,3])
# find the index of the enddate
end = np.where([enddate < day for day in days])[0][0]
xxe = mdates.date2num(days_e[:end+1])
xx = mdates.date2num(days[:end])
xx_c = xx_o
# Date of the transition between the two regimes
xx_tr = mdates.date2num(date(2022,2,20))
alts_edge = combinat_CALIOP['attr']['alts_edge']
alts_mls = MLS['attr']['alts_z']
vmax1 = {'all':3,'15-25':6,'05-15':6}
date_format = mdates.DateFormatter('%d-%b')
ax0 = fig.add_subplot(gs1[0])
sec = '05-15'
im0 = ax0.pcolormesh(xxe,alts_edge,column_CALIOP1[sec][:end,:].T,cmap='gist_ncar',vmax=vmax1[sec],vmin=0)
CS0 = ax0.contour(xx,alts_mls,gaussian_filter(column_MLS1[sec][:end,:],1).T*1.e6,linewidths=3,colors='pink')
ax0.clabel(CS0,inline=1,fontsize=fs)
ax0.plot([xx_tr,xx_tr],[20.5,27.7],color='magenta',alpha=0.5,linewidth=6)
ax0.set_ylim(20,28)
ax0.xaxis_date()
ax0.xaxis.set_major_formatter(date_format)
#ax0.set_xlabel('Day in 2022')
ax0.set_ylabel('Altitude (km)',fontsize=fs) 
ax0.grid(True)
ax0.set_title('CALIOP 532 nm scattering ratio (colour) + MLS water vapour (ppmv) in zonal band 15°S-5°S',fontsize=fs)
divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='3%', pad=0.1) # higher pad shrinks in width (??)
fig.colorbar(im0, cax=cax, orientation='vertical')

ax1 = fig.add_subplot(gs1[2])
ax1.plot(xx,ff1[wl][sec][:end],xx,ff1_mls[wl][sec][:end],
         xx,wdiab[sec][:end],xx,wdiab[sec][:end]+wadiab[sec][:end],
         xx,ff1[wl][sec][:end]-wdiab[sec][:end]-wadiab[sec][:end],
         linewidth=3)
ax1.plot([xx_tr,xx_tr],[-90,20],color='magenta',alpha=0.5,linewidth=6)
ax1.set_title('Vertical motion in 5°S-15°S',fontsize=fs)
ax1.set_ylabel('w (m per day)',fontsize=fs)
ax1.xaxis_date()   
ax1.xaxis.set_major_formatter(date_format)
ax1.legend(['$w_{CALIOP}$','$w_{MLS}$','$w_{diab}$','$w_{ERA5}$','$w_S = w_{CALIOP} - w_{ERA5}$'],fontsize=fs)
ax1.set_ylim(-100,30)
ax1.grid(True)
ax1.tick_params(axis='x', rotation=30) 

ax2 = fig.add_subplot(gs2[0])
sec = '15-25'
im2 = ax2.pcolormesh(xxe,alts_edge,column_CALIOP1[sec][:end,:].T,cmap='gist_ncar',vmax=vmax1[sec],vmin=0)
CS2 = ax2.contour(xx,alts_mls,gaussian_filter(column_MLS1[sec][:end,:],1).T*1.e6,linewidths=3,colors='pink')
ax2.clabel(CS2,inline=1,fontsize=fs)
ax2.plot([xx_tr,xx_tr],[20.5,27.7],color='magenta',alpha=0.5,linewidth=6)
ax2.set_ylim(20,28)
ax2.xaxis_date()
ax2.xaxis.set_major_formatter(date_format)
ax2.set_xlabel('Day in 2022',fontsize=fs)
ax2.set_ylabel('Altitude (km)',fontsize=fs) 
ax2.grid(True)
ax2.set_title('CALIOP 532 nm scattering ratio (colour) + MLS water vapour (ppmv) in zonal band 25°S-15°S',fontsize=fs)
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='3%', pad=0.1) # higher pad shrinks in width (??)
fig.colorbar(im2, cax=cax, orientation='vertical')

ax3 = fig.add_subplot(gs2[2])
ax3.plot(xx,ff1[wl][sec][:end],xx,ff1_mls[wl][sec][:end],
         xx,wdiab[sec][:end],xx,wdiab[sec][:end]+wadiab[sec][:end],
         xx,ff1[wl][sec][:end]-wdiab[sec][:end]-wadiab[sec][:end],
         linewidth=3)
ax3.plot([xx_tr,xx_tr],[-90,20],color='magenta',alpha=0.5,linewidth=6)
ax3.set_title('Vertical motion in 25°S-15°S',fontsize=fs)
ax3.set_ylabel('w (m per day)',fontsize=fs)
ax3.xaxis_date()   
ax3.xaxis.set_major_formatter(date_format)
ax3.legend(['$w_{CALIOP}$','$w_{MLS}$','$w_{diab}$','$w_{ERA5}$','$w_S = w_{CALIOP} - w_{ERA5}$'],fontsize=fs)
ax3.set_ylim(-100,30)
ax3.grid(True)
ax3.tick_params(axis='x', rotation=30) 

ax4 = fig.add_subplot(gs3[0])
ax4.plot(xx,rada[31]['05-15'][:end],xx,rada[31]['15-25'][:end],linewidth=3)
ax4.xaxis_date()   
ax4.xaxis.set_major_formatter(date_format)
ax4.set_ylabel('Aerosol radius (µm)',fontsize=fs)
ax4.set_title('Aerosol radius from fall speed',fontsize=fs)
ax4.legend(('15°S-5°S','25°S-15°S'),fontsize=fs)
ax4.plot([xx_tr,xx_tr],[0.25,3.25],linewidth=6,alpha=0.5,color='magenta')
ax4.grid(True)
#ax4.set_aspect(22)
ax4.tick_params(axis='x', rotation=30) 

ax6 = fig.add_subplot(gs3[4])
lats = combinat_CALIOP['attr']['lats']
cosfac = np.cos(np.deg2rad(combinat_CALIOP['attr']['lats']))
jy1 = np.where(lats >= -15)[0][0]
jy2 = np.where(lats > -5)[0][0]
factor = cosfac[jy1:jy2]/np.sum(cosfac[jy1:jy2])
ax6.plot(xx_c[:end],np.sum(smooth_OMPS[:end,jy1:jy2]*factor[np.newaxis,:],axis=1)/
         np.sum(column_CALIOP[:end,jy1:jy2]*factor[np.newaxis,:],axis=1),lw=3)
jy1 = np.where(lats >= -25)[0][0]
jy2 = np.where(lats > -15)[0][0]
factor = cosfac[jy1:jy2]/np.sum(cosfac[jy1:jy2])
ax6.plot(xx_c[:end],np.sum(smooth_OMPS[:end,jy1:jy2]*factor[np.newaxis,:],axis=1)/
         np.sum(column_CALIOP[:end,jy1:jy2]*factor[np.newaxis,:],axis=1),lw=3)
jy1 = 0
jy2 = column_CALIOP.shape[1]
factor = cosfac[jy1:jy2]/np.sum(cosfac[jy1:jy2])
ax6.plot(xx_c[:end],np.sum(smooth_OMPS[:end,jy1:jy2]*factor[np.newaxis,:],axis=1)/
         np.sum(column_CALIOP[:end,jy1:jy2]*factor[np.newaxis,:],axis=1),lw=3)
ax6.xaxis_date()
ax6.xaxis.set_major_formatter(date_format)
ax6.grid(True)
ax6.legend(('15°S-5°S','25°S-15°S','35°S-20°N'),fontsize=fsl)
ax6.set_ylabel('AOD / Total backscatter (steradian)',fontsize=fs)
ax6.set_title('"lidar ratio" from OMPS 745 nm & CALIOP 532nm')
#ax6.set_aspect(3.2)
ax6.tick_params(axis='x', rotation=30) 

#  Adding the Mie data from Pasquale
ax5 = fig.add_subplot(gs3[2])
#import matplotlib.image as image
#bsc = image.imread(os.path.join('..','kollect','bsca_ext_ratio.png'))
#ax5.imshow(rgb,aspect=1)
#ax5.axis('off')
file1 = os.path.join('..','kollect','Pasquale','r_m.txt')
f1 = open(file1,'r')
aa1 = np.array([np.genfromtxt(x.rstrip(']\n').split()) for x in f1.readlines()]).reshape(300)
f1.close()
file2 = os.path.join('..','kollect','Pasquale','LR.txt')
f2 = open(file2,'r')
f2.readline()
f2.readline()
aa2 = np.array([np.genfromtxt(x.lstrip(' \[').rstrip('\]\n').split()) for x in f2.readlines()])
ax5.plot(aa1[1:],aa2[1:,0],lw=3)
ax5.plot(aa1[1:],aa2[1:,1],lw=3)
ax5.plot(aa1[1:],aa2[1:,2],lw=3)
ax5.legend((u'$\sigma$=1.86',u'$\sigma$=1.70',u'$\sigma$=1.60'),fontsize=fsl)
ax5.set_xlim(0.4,3.)
ax5.set_ylim(3,6)
ax5.grid(True)
ax5.set_xlabel('Mean radius (µm)',fontsize=fs)
ax5.set_ylabel('Extinction / backscatter (str)',fontsize=fs)
ax5.set_title('Mie calculation from lognormal distribution',fontsize=fs)

ax1.annotate('b)',(-0.01,1.04),xycoords='axes fraction',fontsize=14)
ax0.annotate('a)',(-0.05,1.04),xycoords='axes fraction',fontsize=14)
ax3.annotate('d)',(-0.01,1.04),xycoords='axes fraction',fontsize=14)
ax2.annotate('c)',(-0.05,1.04),xycoords='axes fraction',fontsize=14)
ax5.annotate('f)',(-0.05,1.04),xycoords='axes fraction',fontsize=14)
ax4.annotate('e)',(-0.05,1.04),xycoords='axes fraction',fontsize=14)
ax6.annotate('g)',(-0.05,1.04),xycoords='axes fraction',fontsize=14)

plt.savefig('kompozit-fig2.png',dpi=300,bbox_inches='tight')

plt.show()
```
<!-- #region tags=[] -->
## Microphysical properties from MLS
<!-- #endregion -->

<!-- #region tags=[] -->
#### Formula giving the supercooled saturated water log pressure
<!-- #endregion -->

From Tabazadeh et al., 1997
The pressure is given as a function of temperature with unit Pa

```python
pSC = lambda T: 100 * np.exp(18.452406985 - 3505.1578807 / T  - 330918.55082 / T**2 \
    + 12725068.262 / T**3)
```

#### Tabulation of the mass proportion of H2SO4 as a function of the activity


From Tabazadeh et al., 1997
The formula is extrapolated below 0.01 even if the authors say it is then invalid.

```python
def ws(aw,T):
#    if aw < 0.01:
#        y1C = [0., 1., 0., 0.]
#        y2C = [0., 1., 0., 0.]
    if aw < 0.05:
        y1C = [1.2372089320e+1, -1.6125516114e-1, -3.0490657554e+1, -2.1133114241e+0]
        y2C = [1.3455394705e+1, -1.9213122550e-1, -3.4285174607e+1, -1.7620073078e+0]
    elif aw < 0.85:
        y1C = [1.1820654354e+1, -2.0786404244e-1, -4.8073063730e+0, -5.1727540348e+0]
        y2C = [1.2891938068e+1, -2.3233847708e-1, -6.4261237757e+0, -4.9005471319e+0]
    elif aw > 0.85:
        y1C = [-1.8006541028e+2, -3.8601102592e-1, -9.3317846778e+1, 2.7388132245e+2]
        y2C = [-1.7695814097e+2, -3.6257048154e-1, -9.0469744201e+1, 2.6745509988e+2]
    y1 = y1C[0]*aw**y1C[1] + y1C[2] * aw + y1C[3]
    y2 = y2C[0]*aw**y2C[1] + y2C[2] * aw + y2C[3]
    ms = y1 + (T-190)*(y2-y1)/70
    ws = 98 * ms / (98 * ms + 1000)
    return(ws)

wsv = np.vectorize(ws)

```

#### Determination of the activity from MLS  


We use MLS water vapour volume mixing ratio and the pressure form ERA5

```python
awe = lambda r,T,lnP: r*np.exp(lnP) / pSC(T)
awev = np.vectorize(awe)
```

```python
for dd in MLS['data']:   
    day = date(dd.year,dd.month,dd.day)
    if day > date(2022,8,6): continue
    MLS['data'][dd]['AWZ'] = np.empty(shape=(len(lats_mls),len(z_mls)))
    MLS['data'][dd]['WSZ'] = np.empty(shape=(len(lats_mls),len(z_mls)))
    for jy in range(len(lats_mls)):
         MLS['data'][dd]['AWZ'][jy,:] = awev(MLS['data'][dd]['WPZ'][jy,:],MLS['data'][dd]['TZ'][jy,:],MLS['data'][dd]['LPZ'][jy,:])
         MLS['data'][dd]['WSZ'][jy,:] = wsv(MLS['data'][dd]['AWZ'][jy,:],MLS['data'][dd]['TZ'][jy,:])
```

### Supplement to latitude sections

```python
# Definitions of latitude sections
secs = {'all','15-25','05-15'}
lats_MLS = MLS['attr']['lats']
jy1 = {'CALIOP':{'all':0,'15-25':np.where(lats_CALIOP >= -25)[0][0],'05-15':np.where(lats_CALIOP >= -15)[0][0]},
       'MLS':{'all':0,'15-25':np.where(lats_MLS >= -25)[0][0],'05-15':np.where(lats_MLS >= -15)[0][0]}}
jy2 = {'CALIOP':{'all':len(lats_CALIOP),'15-25':np.where(lats_CALIOP > -15)[0][0],'05-15':np.where(lats_CALIOP > -5)[0][0]},
       'MLS':{'all':len(lats_MLS),'15-25':np.where(lats_MLS >= -15)[0][0],'05-15':np.where(lats_MLS >= -5)[0][0]}}
# Generation of the temporal vector
day0 = date(2022,1,27)
day1 = date(2022,8,6)
day = day0
day_e = datetime.combine(day0-timedelta(days=1),time(12))
nd = (day1-day0).days+1
days_e = []
days = []
while day <= day1:
    days.append(day)
    days_e.append(day_e)
    day += timedelta(days=1)
    day_e += timedelta(days=1)
days_e.append(day_e)
     
# Additional processing of MLS data
nz = len(MLS['attr']['alts_z'])
column_MLSE = {}
varlist = ['TZ','AWZ','WSZ']
for var in varlist:
    column_MLSE[var] = {}
    for sec in secs:
        column_MLSE[var][sec] = np.ma.asarray(np.full((nd,nz),999999.))
jd = 0
day = day0
while day <= day1:
    dd = datetime(day.year,day.month,day.day)
    if dd not in MLS['data']:
        for var in varlist:
            for sec in secs:
                column_MLSE[var][sec][jd,:] = np.ma.masked
    else:
        for var in varlist:
            for sec in secs:
                j1 = jy1['MLS'][sec]
                j2 = jy2['MLS'][sec]
                cosfac = np.cos(np.deg2rad(MLS['attr']['lats'][j1:j2]))
                cosfac = cosfac/np.sum(cosfac)
                column_MLSE[var][sec][jd,:] = np.ma.sum(cosfac[:,np.newaxis]*MLS['data'][dd][var][j1:j2,:],axis=0)
    jd += 1
    day += timedelta(days=1)
```

### Superposition of CALIOP with the ws from MLS

```python
from scipy.ndimage import gaussian_filter
vmax = {'all':3,'15-25':6,'05-15':6}
for sec in secs:
    fig, ax = plt.subplots(figsize=(12,4))
    xxe = mdates.date2num(days_e)
    alts_edge = combinat_CALIOP['attr']['alts_edge']
    im = ax.pcolormesh(xxe,alts_edge,column_CALIOP1[sec].T,cmap='gist_ncar',vmax=vmax[sec],vmin=0)
    xx = mdates.date2num(days)
    alts_mls = MLS['attr']['alts_z']
    #CS = ax.contour(xx,alts_mls,gaussian_filter(column_MLS1[sec]*1.e6,1).T,linewidths=3,colors='pink')
    CS1 = ax.contour(xx,alts_mls,gaussian_filter(column_MLSE['WSZ'][sec],1).T,linewidths=3,colors='pink')
    CS2 = ax.contour(xx,alts_mls,gaussian_filter(np.log10(column_MLSE['AWZ'][sec]),1).T,linewidths=3,colors='magenta')    
    ax.clabel(CS1,inline=1,fontsize=12)
    ax.clabel(CS2,inline=1,fontsize=12)
    ax.set_ylim(20,28)
    ax.xaxis_date()
    date_format = mdates.DateFormatter('%b-%d')
    ax.xaxis.set_major_formatter(date_format)
    ax.set_ylabel('Altitude (km)') 
    ax.set_xlabel('Day in 2022')
    ax.grid(True)
    ax.set_title('CALIOP 532 nm scattering ratio + MLS water vapour in zonal band '+sec+'S')
    plt.colorbar(im)
    fig.autofmt_xdate()
    #fig.savefig('scattering-MLS-profile-'+sec+'.png',dpi=144,bbox_inches='tight')
    plt.show()
#attribs1 = {'alts_CALIOP':combinat_CALIOP['attr']['alts'],'alts_edge_CALIOP':alts_edge,
#           'alts_MLS':alts_mls,'alts_edge_MLS':MLS['attr']['alts_z_edge'],
#           'days':days,'days_edge':days_e,'numdays':xx,'numdays-edge':xxe}
#with gzip.open('colonnes.pkl','wb') as f:
#    pickle.dump([column_CALIOP1,column_MLS1,attribs1],f,protocol=pickle.HIGHEST_PROTOCOL)
```

```python
Study MLS profile at 20S on May 1
```

```python
lats_mls
```

```python
dd = datetime(2022,7,1)
# choice of lat: 9 for 20S, 16 for 10S
jy = 9
fig = plt.figure(figsize=(14,6))
plt.subplot(241)
plt.plot(MLS['data'][dd]['WPZ'][jy,:]*1.e6,alts_mls)
plt.grid(True)
plt.title('Water vapour (ppmv)')
plt.subplot(242)
plt.plot(MLS['data'][dd]['TZ'][jy,:],alts_mls)
plt.grid(True)
plt.title('Temperature (K)')
plt.subplot(243)
plt.plot(np.exp(MLS['data'][dd]['LPZ'][jy,:]),alts_mls)
plt.grid(True)
plt.title('Pressure (Pa)')
plt.subplot(244)
plt.plot(pSC(MLS['data'][dd]['TZ'][jy,:]),alts_mls)
plt.grid(True)
plt.title('Supercooled sat p (Pa)')
plt.subplot(245)
plt.plot(MLS['data'][dd]['WPZ'][jy,:]*np.exp(MLS['data'][dd]['LPZ'][jy,:]),alts_mls)
plt.grid(True)
plt.title('Water vapour pressure (Pa)')
plt.subplot(246)
aw = MLS['data'][dd]['WPZ'][jy,:]*np.exp(MLS['data'][dd]['LPZ'][jy,:])/pSC(MLS['data'][dd]['TZ'][jy,:])
plt.grid(True)
plt.plot(aw,alts_mls)
plt.plot([0.01,0.01],[18,30])
plt.grid(True)
plt.title('water activity')
plt.subplot(247)
plt.plot(100*wsv(aw,MLS['data'][dd]['TZ'][jy,:]),alts_mls)
plt.grid(True)
plt.title('ws')
```

```python
Plot of water activity with time until June
```

```python
figsave = True
fig = plt.figure(figsize=(10,8))
plt.subplot(221)
jy = 9
for mm in [2,3,4,5,6,7]:
    dd = datetime(2022,mm,1)
    aw = MLS['data'][dd]['WPZ'][jy,:]*np.exp(MLS['data'][dd]['LPZ'][jy,:])/pSC(MLS['data'][dd]['TZ'][jy,:])
    plt.plot(aw,alts_mls)
plt.plot([0.01,0.01],[18,30])
plt.legend(['1 Feb','1 Mar','1 Apr','1 May','1 June','1 Jul'])
plt.ylabel('Altitude (km)')
plt.xlabel('Water activity at 20°S')
plt.xlim(0,0.10)
plt.subplot(222)
jy = 16
for mm in [2,3,4,5,6,7]:
    dd = datetime(2022,mm,1)
    aw = MLS['data'][dd]['WPZ'][jy,:]*np.exp(MLS['data'][dd]['LPZ'][jy,:])/pSC(MLS['data'][dd]['TZ'][jy,:])
    plt.plot(aw,alts_mls)
plt.xlim(0,0.10)
plt.plot([0.01,0.01],[18,30])
plt.legend(['1 Feb','1 Mar','1 Apr','1 May','1 June','1 Jul'])
plt.xlabel('Water activity at 10°S')
plt.subplot(223)
jy = 9
for mm in [2,3,4,5,6,7]:
    dd = datetime(2022,mm,1)
    aw = MLS['data'][dd]['WPZ'][jy,:]*np.exp(MLS['data'][dd]['LPZ'][jy,:])/pSC(MLS['data'][dd]['TZ'][jy,:])
    plt.plot(100*wsv(aw,MLS['data'][dd]['TZ'][jy,:]),alts_mls)
plt.legend(['1 Feb','1 Mar','1 Apr','1 May','1 June','1 Jul'])
plt.xlim([55,80])
plt.xlabel('H2SO4 weight percentage at 20°S')
plt.subplot(224)
jy = 16
for mm in [2,3,4,5,6,7]:
    dd = datetime(2022,mm,1)
    aw = MLS['data'][dd]['WPZ'][jy,:]*np.exp(MLS['data'][dd]['LPZ'][jy,:])/pSC(MLS['data'][dd]['TZ'][jy,:])
    plt.plot(100*wsv(aw,MLS['data'][dd]['TZ'][jy,:]),alts_mls)
plt.legend(['1 Feb','1 Mar','1 Apr','1 May','1 June','1 Jul'])
plt.xlim([55,80])
plt.xlabel('H2SO4 weight percentage at 10°S')
if figsave: fig.savefig('waterActivity.png',dpi=144,bbox_inches='tight')
plt.show()


```

```python

```
