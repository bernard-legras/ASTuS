---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: p39
    language: python
    name: p39
---

<!-- #region tags=[] -->
# Processing of OMPS / CALIOP / MLS data for Hunga Tonga
<!-- #endregion -->

```python
Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr
```

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

OMPS fixed latitude grid with 50 bins between -35 and 20 

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
day1 = date(2022,6,9)
day = day0
# Option to filter all latitudes in the SAA range
SAA_filt = True
# Initialization of the molecular extinction ratio
lamb = [510., 600., 675., 745., 869., 997.]
QS = 4.5102e-31*(lamb[3]/550)**(-4.025-0.05627*(lamb[3]/550)**(-1.647)) # en m2 
NA = cst.Na
RA = cst.Ra
```

<!-- #region tags=[] -->
### Performing the action
<!-- #endregion -->

```python tags=[]
day = day0
while day <= day1 :
    print(day)
    file = day.strftime('OMPS-NPP_LP-L2-AER-DAILY_v2.1_%Ym%m%d_*.h5')
    # Exception for 11 and 13 May
    if day in [date(2022,5,11),date(2022,5,13)]:
        file = day.strftime('OMPS-NPP_LP-L2-AER-DAILY_v2.0_%Ym%m%d_*.h5')
    search = os.path.join(dirOMPS,file)
    fname = glob.glob(search)[0]
    combinat['data'][day] = {}
    
    # Open the file and read the needed field 
    ncid = Dataset(fname)
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

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
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

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
#### ncid header
<!-- #endregion -->

```python
ncid
```

### Storing the result

```python
name = 'combinat-daily'
if extended: name = 'combinat-daily-extended'
if SAA_filt:
    with gzip.open(name+'-SAAfilt.v2.1.pkl','wb') as f:
        pickle.dump(combinat,f,protocol=pickle.HIGHEST_PROTOCOL)
else:
    with gzip.open(name+'.v2.1.pkl','wb') as f:
        pickle.dump(combinat,f,protocol=pickle.HIGHEST_PROTOCOL)
```

```python
combinat['data'].keys()
```

### Loading the result

```python
with gzip.open('combinat-daily-SAAfilt.v2.1.pkl','rb') as f:
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
def plot1(day,ylim=(18,30),vmax=None,ax=None,txt=None,cmap='jet',ratio=True, empir=False, xlabel=True,showlat=True):
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
    if showlat: axe.set_xlabel('Latitude')
    axe.set_ylabel('Altitude')
    axe.grid(True)
    axe.set_title(txt)
    # plt.show()
    return im
```

<!-- #region tags=[] -->
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
with gzip.open('combinat-daily-SAAfilt.v2.1.pkl','rb') as f:
    combinat = pickle.load(f)
plot1(date(2022,3,11),ylim=(15,40),empir=False,vmax=30,cmap='gist_ncar')
plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
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

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
#### Plots from 27 January onward
<!-- #endregion -->

```python
figsave = True
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = date(2022,1,27) + timedelta(days=j)
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
    day = date(2022,3,10) + timedelta(days=j)
    im = plot1(day,vmax=40,ax=axes[j],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar')
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('OMPS 745 nm daily zonal average extinction ratio',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-2-OMPS.v2.1.png',dpi=300,bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(41):
    day = date(2022,4,21) + timedelta(days=j)
    im = plot1(day,vmax=40,ax=axes[j],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar')
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('OMPS 745 nm daily zonal average extinction ratio',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-3-OMPS.v2.1.png',dpi=300,bbox_inches='tight')
plt.show()
```

<!-- #region tags=[] -->
## Plots from CALIOP combined files
<!-- #endregion -->

### Read the CALIOP data

```python
with gzip.open(os.path.join('..','HT-HT','superCombi_caliop.all_nit.pkl'),'rb') as f:
    combinat_CALIOP = pickle.load(f)
```

```python
combinat_CALIOP['data'][date(2022,6,6)]['TROPOH']
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Dedicated plot routine for CALIOP SR
<!-- #endregion -->

```python
def plot2(day,var='SR532',ylim=(18,30),vmax=8,ax=None,txt=None,cmap='jet',showlat=True):
    SR = combinat_CALIOP['data'][day][var]
    latsEdge = combinat_CALIOP['attr']['lats_edge']
    altsEdge = combinat_CALIOP['attr']['alts_edge']
    if ax == None: fig,axe = plt.subplots(nrows=1,ncols=1)
    else: axe = ax
    im = axe.pcolormesh(latsEdge,altsEdge,SR.T,cmap=cmap,vmin=0,vmax=vmax)
    if ax == None: plt.colorbar(im)
    axe.set_ylim(ylim[0],ylim[1])
    axe.set_xlim(-35,20)
    if showlat: axe.set_xlabel('Latitude')
    axe.set_ylabel('Altitude')
    axe.grid(True)
    if txt == None: axe.set_title(day.strftime('CALIOP 532 nm attenuated scattering ratio %d %b %Y'))
    else: axe.set_title(txt)
    return im
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Example with a single frame
<!-- #endregion -->

```python
plot2(date(2022,2,27),cmap='gist_ncar')
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Multiframe plot
<!-- #endregion -->

```python
figsave = True
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = date(2022,1,27) + timedelta(days=j)
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
    day = date(2022,3,10) + timedelta(days=j)
    if day not in combinat_CALIOP['data']: continue
    im = plot2(day,ax=axes[j],txt=day.strftime('%d %b %Y'),cmap='gist_ncar')
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('CALIOP daily zonal average 532 nm attenuated scattering ratio',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-2-CALIOP.png',dpi=300,bbox_inches='tight')
plt.show()
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(40):
    day = date(2022,4,21) + timedelta(days=j)
    if day not in combinat_CALIOP['data']: continue
    im = plot2(day,ax=axes[j],txt=day.strftime('%d %b %Y'),cmap='gist_ncar')
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('CALIOP daily zonal average 532 nm attenuated scattering ratio',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-3-CALIOP.png',dpi=300,bbox_inches='tight')
plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
### Composite image with both OMPS and CALIOP
<!-- #endregion -->

```python
figsave = True
fig, axes = plt.subplots(figsize=(21,60),ncols=6,nrows=14,sharex=True,sharey=True)
#axes = flatten(axes)
for j in range(42):
    ix = j%6
    jy = int((j - j%6)/6)
    day = date(2022,1,27) + timedelta(days=j)
    im1 = plot1(day,vmax=40,ax=axes[2*jy,ix],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar')
    im2 = plot2(day,ax=axes[2*jy+1,ix],txt=day.strftime('%d %b %Y'),cmap='gist_ncar')
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
    day = date(2022,3,10) + timedelta(days=j)
    im1 = plot1(day,vmax=40,ax=axes[2*jy,ix],txt=day.strftime('%d %b %Y'),empir=False,cmap='gist_ncar')
    if day not in combinat_CALIOP['data']: continue
    im2 = plot2(day,ax=axes[2*jy+1,ix],txt=day.strftime('%d %b %Y'),cmap='gist_ncar')
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

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
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
    day = date(2022,1,27) + timedelta(days=j)
    if day not in combinat_CALIOP['data']: continue
    im = plot4(day,ax=axes[j],txt=day.strftime('%d %b %Y'),vmax=100,vmin=1)
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('745 nm OMPS extinction ratio to 532nm CALIOP attenuated scattering ratio (sr)',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-1-Extinction_to_scattering.png',dpi=300,bbox_inches='tight')
plt.show()
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = date(2022,3,10) + timedelta(days=j)
    if day not in combinat_CALIOP['data']: continue
    im = plot4(day,ax=axes[j],txt=day.strftime('%d %b %Y'),vmax=100,vmin=1)
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('745 nm OMPS extinction ratio to 532nm CALIOP attenuated scattering ratio (sr)',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-2-Extinction_to_scattering.png',dpi=300,bbox_inches='tight')
plt.show()
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(36):
    day = date(2022,4,21) + timedelta(days=j)
    if day not in combinat_CALIOP['data']: continue
    im = plot4(day,ax=axes[j],txt=day.strftime('%d %b %Y'),vmax=100,vmin=1)
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('745 nm OMPS extinction ratio to 532nm CALIOP attenuated scattering ratio (sr)',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-3-Extinction_to_scattering.png',dpi=300,bbox_inches='tight')
plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
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
    day = date(2022,1,27) + timedelta(days=j)
    im = plot3(day,ax=axes[j],txt=day.strftime('%d %b %Y'),cmap='gist_ncar',vmax=3)
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('OMPS 745 nm daily zonal std dev / mean extinction',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-1-OMPS-dilution.png',dpi=300,bbox_inches='tight')
plt.show()
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = date(2022,3,10) + timedelta(days=j)
    im = plot3(day,ax=axes[j],txt=day.strftime('%d %b %Y'),cmap='gist_ncar',vmax=3)
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
# Definition of the altitde grid
z_mls = MLS['attr']['alts_z'] = np.arange(18.5,29.6,1)
ze_mls = MLS['attr']['alts_z_edge'] = np.arange(18,30.5,1)
# Interpolation
for dd in MLS['data']:
    day = date(dd.year,dd.month,dd.day)
    MLS['data'][dd]['WPZ'] = np.empty(shape=(len(lats_mls),len(z_mls)))
    try:      
        for jy in range(len(lats_mls)):
            Z = cl1[jy] * zonal['mean'][day]['Z'][:,jl1[jy]] + cl2[jy] * zonal['mean'][day]['Z'][:,jl2[jy]] 
            θ = cl1[jy] * zonal['mean'][day]['PT'][:,jl1[jy]] + cl2[jy] * zonal['mean'][day]['PT'][:,jl2[jy]]
            T = cl1[jy] * zonal['mean'][day]['T'][:,jl1[jy]] + cl2[jy] * zonal['mean'][day]['T'][:,jl2[jy]]
            logP = np.log(cst.p0) + np.log(T/θ)/cst.kappa
            inter1 = aki(-Z,-logP)
            inter2 = aki(-logp_mls,MLS['data'][dd]['meanWP'][:,jy])
            MLS['data'][dd]['WPZ'][jy,:] = inter2(inter1(-z_mls*1000))
            #print(dd)
    except:
        print(day.strftime('missed %d %m'))
        continue
```

<!-- #region tags=[] -->
#### Dedicated plot function
<!-- #endregion -->

```python
def plotmls(dd,vmax=25,vmin=0,ax=None,txt=None,cmap='gist_ncar',ylim=(18,30)):
    if txt == None: txt = dd.strftime('MLS %d %b %Y (ppmv)')
    if ax == None: fig,axe = plt.subplots(nrows=1,ncols=1)
    else: axe = ax
    im = axe.pcolormesh(lats_mls_e,ze_mls,MLS['data'][dd]['WPZ'].T*1.e6,cmap=cmap,vmin=vmin,vmax=vmax)
    #axe.contour(lats_lms,z_mls,MLS['data'][dd]['WPZ']*1.e6,levels=(4,8,12,16))
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
#### Plot a few tests
<!-- #endregion -->

```python
plotmls(datetime(2022,1,31));plt.show()
plotmls(datetime(2022,5,25));plt.show()
```

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
#### Multiple plots
<!-- #endregion -->

```python tags=[]
figsave = True
fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(42):
    day = datetime(2022,1,27) + timedelta(days=j)
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
    day = datetime(2022,3,10) + timedelta(days=j)
    im = plotmls(day,vmax=25,ax=axes[j],txt=day.strftime('%d %b %Y'))
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('MLS H20 daily zonal average mixing ratio (ppmv)',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-2-MLS.v4.png',dpi=300,bbox_inches='tight')
plt.show()

fig, axes = plt.subplots(figsize=(21,29.7),ncols=6,nrows=7,sharex=True,sharey=True)
axes = flatten(axes)
for j in range(36):
    day = datetime(2022,4,21) + timedelta(days=j)
    im = plotmls(day,vmax=25,ax=axes[j],txt=day.strftime('%d %b %Y'))
fig.subplots_adjust(top=0.8)
cbar_ax = fig.add_axes([0.20, 0.84, 0.6, 0.02])
fig.colorbar(im, cax=cbar_ax,orientation='horizontal')
fig.suptitle('MLS H20 daily zonal average mixing ratio (ppmv)',y=0.9,fontsize=24)
if figsave: plt.savefig('HT-3-MLS.v4.png',dpi=300,bbox_inches='tight')
plt.show()
```

<!-- #region tags=[] -->
### Selective row composite
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

<!-- #region tags=[] -->
## Latitude average plots in the 35°S-20°N domain
<!-- #endregion -->

### OMPS

```python
Here we generate the column of the extinction ratio for OMPS as an average in latitude
```

```python
days = []
for day in combinat['data']:
    if day < date(2022,1,27): continue
    days.append(day)
nd = len(days)
days_e = [datetime.combine(day-timedelta(days=1),time(12)) for day in days]
days_e.append(days_e[-1]+timedelta(hours=12))
column = np.ma.asarray(np.full((nd,41),999999.))
cosfac = np.cos(np.deg2rad(combinat['attr']['lats']))
cosfac = cosfac/np.sum(cosfac)
jd = 0
for day in combinat['data']:
    if day < date(2022,1,27): continue
    column[jd,:] = np.ma.sum(cosfac[:,np.newaxis]*combinat['data'][day]['meanExtRatio'],axis=0)
    jd += 1
```

```python
fig, ax = plt.subplots()
xlims = mdates.date2num([days[0],days[-1]])
xx_e = mdates.date2num(days_e)
alts_edge = combinat['attr']['alts_edge']
#im=ax.imshow(column.T,cmap='gist_ncar',extent=(xlims[0],xlims[-1],0.,41),origin='lower',aspect=4)
im = ax.pcolormesh(xx_e,alts_edge,column.T,cmap='gist_ncar')
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

```python
day0 = date(2022,1,27)
day1 = date(2022,5,30)
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

### OMPS

```python
days = []
for day in combinat['data']:
    if day < date(2022,1,27): continue
    days.append(day)
nd = len(days)
days_o = [datetime.combine(day-timedelta(days=1),time(12)) for day in days]
days_o.append(days_o[-1]+timedelta(hours=12))
column_OMPS = np.ma.asarray(np.full((nd,nlat),999999.))
jd = 0
for day in combinat['data']:
    if day < date(2022,1,27): continue
    column_OMPS[jd,:] = np.ma.sum(combinat['data'][day]['meanExt'][:,18:30],axis=1)
    jd += 1
```

```python
fig, ax = plt.subplots()
xlims = mdates.date2num([days[0],days[-1]])
xx_o = mdates.date2num(days_o)
lats_edge = combinat['attr']['lats_edge']
#im=ax.imshow(column.T,cmap='gist_ncar',extent=(xlims[0],xlims[-1],0.,41),origin='lower',aspect=4)
im = ax.pcolormesh(xx_o,lats_edge,column_OMPS.T,cmap='gist_ncar',vmax=0.04)
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
OMOD = np.sum(column_OMPS*weights[np.newaxis,:],axis=1)
xx = mdates.date2num(days)
fig, ax = plt.subplots()
ax.plot(xx,OMOD)
ax.xaxis_date()
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
ax.set_xlabel('Day (in 2022)')
ax.set_title('Mean OMPS-LP OD 18-30 km 20°N-35°S')
ax.set_ylabel('Optical depth')
```

### CALIOP

```python
day0 = date(2022,1,27)
day1 = date(2022,6,8)
day = day0
day_e = datetime.combine(day0-timedelta(days=1),time(12))
nd = (day1-day0).days+1
ny = len(combinat_CALIOP['attr']['lats'])
cosfac = np.cos(np.deg2rad(combinat_CALIOP['attr']['lats']))
cosfac = cosfac/np.sum(cosfac)
column_CALIOP = np.ma.asarray(np.full((nd,ny),999999.))
jd = 0
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

```python
smooth_OMPS = np.zeros(shape=column_CALIOP.shape)
for jd in range(smooth_OMPS.shape[0]):
    interp = interp1d(combinat['attr']['lats'],column_OMPS[jd,:],kind='slinear',fill_value='extrapolate')
    smooth_OMPS[jd,:] = np.reshape(interp(combinat_CALIOP['attr']['lats']),183)
fig, ax = plt.subplots()
xxe = mdates.date2num(days_ce)
lats_edge = combinat_CALIOP['attr']['lats_edge']
im=ax.pcolormesh(xxe,lats_edge,(smooth_OMPS/column_CALIOP).T,cmap='gist_ncar',vmax=40)
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

```python
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

```python
# Mean in 15S-25S & 05S-15S & 
# Same with cosine weighting (does not seem to change anything visible)
lats = combinat_CALIOP['attr']['lats']
cosfac = np.cos(np.deg2rad(lats))
fig, axs = plt.subplots(figsize=(16,3.5),nrows=1,ncols=3)
xx_c = mdates.date2num(days_c)
date_format = mdates.DateFormatter('%b-%d')
jy1 = np.where(lats >= -25)[0][0]
jy2 = np.where(lats > -15)[0][0]
factor = cosfac[jy1:jy2]/np.sum(cosfac[jy1:jy2])
axs[1].plot(xx_c,np.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1))
axs[0].plot(xx_c,np.sum(column_CALIOP[:,jy1:jy2]*factor[np.newaxis,:],axis=1))
axs[2].plot(xx_c,np.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1)/
            np.sum(column_CALIOP[:,jy1:jy2]*factor[np.newaxis,:],axis=1))
jy1 = np.where(lats >= -15)[0][0]
jy2 = np.where(lats > -5)[0][0]
factor = cosfac[jy1:jy2]/np.sum(cosfac[jy1:jy2])
axs[1].plot(xx_c,np.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1))
axs[0].plot(xx_c,np.sum(column_CALIOP[:,jy1:jy2]*factor[np.newaxis,:],axis=1))
axs[2].plot(xx_c,np.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1)/
            np.sum(column_CALIOP[:,jy1:jy2]*factor[np.newaxis,:],axis=1))
jy1 = 0
jy2 = column_CALIOP.shape[1]
factor = cosfac[jy1:jy2]/np.sum(cosfac[jy1:jy2])
axs[1].plot(xx_c,np.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1))
axs[0].plot(xx_c,np.sum(column_CALIOP[:,jy1:jy2]*factor[np.newaxis,:],axis=1))
axs[2].plot(xx_c,np.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1)/
            np.sum(column_CALIOP[:,jy1:jy2]*factor[np.newaxis,:],axis=1))
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

<!-- #region tags=[] -->
## Composite section plot of CALIOP with MLS
<!-- #endregion -->

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
day1 = date(2022,6,6)
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
    ax.clabel(CS,inline=1,fontsize=12)
    ax.set_ylim(22,28)
    ax.xaxis_date()
    date_format = mdates.DateFormatter('%b-%d')
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

Calculation of the mean, max & median

```python
center_CALIOP={}
alts = combinat_CALIOP['attr']['alts']
# Blacklisted dates (because too less orbits)
blacklist = [date(2022,3,31),]
secs = ('05-15','15-25','all')
offset = {'15-25':2, '05-15':2, 'all':1}
days = []
for sec in secs:
    print(sec)
    center_CALIOP[sec] = {"max":np.ma.zeros(column_CALIOP1['15-25'].shape[0]),
                         "median":np.ma.zeros(column_CALIOP1['15-25'].shape[0]),
                          "mean":np.ma.zeros(column_CALIOP1['15-25'].shape[0])}
    jd = 0
    day0 = date(2022,1,27)
    day1 = date(2022,6,6)
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
date_format = mdates.DateFormatter('%b-%d')    
for sec in secs:
    jg += 1
    ax = plt.subplot(1,3,jg)
    ax.plot(xx,center_CALIOP[sec]["max"],xx,center_CALIOP[sec]["mean"],xx,center_CALIOP[sec]["median"])
    ax.set_title('CALIOP '+sec)
    ax.xaxis_date()   
    ax.xaxis.set_major_formatter(date_format)
    ax.legend(('max','mean','median'))
    ax.set_ylim(23,26.5)
plt.show
fig.autofmt_xdate()
```

### Calculation of descent rates from the median using Savitsky-Golay filter


Check where to make the cut

```python
np.where(center_CALIOP['05-15']["median"].mask == True)
```

#### Make the fit for the three sections

<!-- #region jp-MarkdownHeadingCollapsed=true tags=[] -->
##### Uncorrected version
<!-- #endregion -->

```python
wl = 21
fo = 2
mode = 'interp'
l1 = 60
l2 = 64
ff0 = {}
ff1 = {}
center = 'median'

for sec in secs:
    # ff0 fit to the function and ff1 to its derivative
    ff0[sec] = np.ma.zeros(len(center_CALIOP[sec][center]))
    ff1[sec] = np.ma.zeros(len(center_CALIOP[sec][center]))
    ff0[sec][l1:l2] = np.ma.masked
    ff1[sec][l1:l2] = np.ma.masked
    ff0[sec][:l1] = sps.savgol_filter(center_CALIOP[sec][center][:l1],wl,fo,deriv=0,mode=mode)
    ff1[sec][:l1] = sps.savgol_filter(center_CALIOP[sec][center][:l1],wl,fo,deriv=1,mode=mode)
    ff0[sec][l2:] = sps.savgol_filter(center_CALIOP[sec][center][l2:],wl,fo,deriv=0,mode=mode)
    ff1[sec][l2:] = sps.savgol_filter(center_CALIOP[sec][center][l2:],wl,fo,deriv=1,mode=mode)
    # Conversion of the derivative m/day
    ff1[sec] *= 1000

fig = plt.figure(figsize=(15,6))
xx = mdates.date2num(days)
jg = 0
date_format = mdates.DateFormatter('%b-%d')    
for sec in secs:
    jg += 1
    ax0 = plt.subplot(2,3,jg)
    ax0.plot(xx,center_CALIOP[sec][center],xx,ff0[sec])
    ax0.set_title('CALIOP '+sec)
    ax0.xaxis_date()   
    ax0.xaxis.set_major_formatter(date_format)
    ax1 = plt.subplot(2,3,jg+3)
    ax1.plot(xx,ff1[sec])
    ax1.xaxis_date()   
    ax1.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
```

##### Corrected version

```python
wl = 21
fo = 2
mode = 'interp'
l1 = 60
l2 = 64
ff0 = {}
ff1 = {}
ff1b = {}
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
for sec in secs:
    # ff0 fit to the function and ff1 to its derivative
    ff0[sec] = np.ma.zeros(len(center_CALIOP_corr[sec][center]))
    ff1[sec] = np.ma.zeros(len(center_CALIOP_corr[sec][center]))
    ff1b[sec] = np.ma.zeros(len(center_CALIOP_corr[sec][center]))
    ff0[sec] = sps.savgol_filter(center_CALIOP_corr[sec][center],wl,fo,deriv=0,mode=mode)
    ff1[sec] = sps.savgol_filter(center_CALIOP_corr[sec][center],wl,fo,deriv=1,mode=mode)
    ff1b[sec][1:-1] = 0.5*(ff0[sec][2:] - ff0[sec][:-2])
    ff1b[sec][0] = ff0[sec][1]-ff0[sec][0]
    ff1b[sec][-1] = ff0[sec][-1]-ff0[sec][-2]
    # Conversion of the derivative m/day
    ff1[sec] *= 1000
    ff1b[sec] *= 1000
    

fig = plt.figure(figsize=(15,6))
xx = mdates.date2num(days)
jg = 0
date_format = mdates.DateFormatter('%b-%d')    
for sec in secs:
    jg += 1
    ax0 = plt.subplot(2,3,jg)
    ax0.plot(xx,center_CALIOP_corr[sec][center],xx,ff0[sec])
    ax0.set_title('CALIOP '+sec)
    ax0.xaxis_date()   
    ax0.xaxis.set_major_formatter(date_format)
    ax1 = plt.subplot(2,3,jg+3)
    ax1.plot(xx,ff1[sec],xx,ff1b[sec])
    ax1.xaxis_date()   
    ax1.xaxis.set_major_formatter(date_format)
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
wdiab = {}
wadiab = {}
for sec in secs:
    wdiab[sec] = np.ma.zeros(shape=ff0[sec].shape)
    wadiab[sec] = np.ma.zeros(shape=ff0[sec].shape)
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
        wdiab[sec][jd] = np.interp(ff0[sec][jd],Z[25:50],DZDt)
        wadiab[sec][jd] = np.interp(ff0[sec][jd],Z[25:50],DZDtAdiab)
        jd += 1
        day += timedelta(days=1)  
```

Figures from CALIOP and MLS


ACHTUNG ACHTUNG: MLS slope calculated in the next subsection must be available for this plot

```python
fig = plt.figure(figsize=(16,4))
xx = mdates.date2num(days)
jg = 0
date_format = mdates.DateFormatter('%b-%d')    
for sec in secs:
    jg += 1
    ax0 = plt.subplot(1,3,jg)
    ax0.plot(xx,ff1[sec],xx,ff1b[sec],xx,wdiab[sec],xx,wadiab[sec],xx,ff1[sec]-wdiab[sec],xx,ff1b[sec]-wdiab[sec],xx,ff1[sec]-wdiab[sec]-wadiab[sec])
    ax0.set_title('CALIOP '+sec)
    ax0.xaxis_date()   
    ax0.xaxis.set_major_formatter(date_format)
    ax0.legend(['ff1','ff1b','wdiab','wadiab','ff1-wdiab','ff1b-wdiab','ff1-wdiab-adiab'])
fig.autofmt_xdate()
plt.show()
# Version grand public
fig = plt.figure(figsize=(16,4))
xx = mdates.date2num(days)
jg = 0
date_format = mdates.DateFormatter('%b-%d')    
for sec in secs:
    jg += 1
    ax0 = plt.subplot(1,3,jg)
    ax0.plot(xx,ff1[sec],xx,wdiab[sec],xx,wadiab[sec],xx,ff1[sec]-wdiab[sec]-wadiab[sec],xx,ff1_mls[sec],linewidth=3)
    ax0.set_title('Vertical motion '+sec+'S')
    ax0.set_ylabel('w (m per day)')
    ax0.xaxis_date()   
    ax0.xaxis.set_major_formatter(date_format)
    ax0.legend(['$w_C$','$w_R$','$w_{adiab}$','$w_S$','$w_{MLS}$'],fontsize=12)
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
for sec in secs:
    rada[sec] = np.sqrt(np.clip(-18*mu*(ff1[sec]-wdiab[sec]-wadiab[sec])/(86400*cst.g*Cc*rho),0,1000))*1e6/2
    rada[sec] = np.ma.array(rada[sec])
    rada[sec][rada[sec]<=0] = np.ma.masked
```

```python
fig,ax = plt.subplots(figsize=(4,4))
ax.plot(xx,rada['15-25'],xx,rada['05-15'],linewidth=3)
ax.xaxis_date()   
ax.xaxis.set_major_formatter(date_format)
ax.set_ylabel('Aerosol radius (µm)')
ax.legend(('15-25 S','05-15 S'),fontsize=12)
ax.grid(True)
fig.autofmt_xdate()
```

Calculation of a fit for MLS

```python
center_MLS={}
alts = MLS['attr']['alts_z']
secs = ('05-15','15-25','all')
offset = {'15-25':6.e-6, '05-15':6.e-6, 'all':5.e-6}
days = []
for sec in secs:
    print(sec)
    center_MLS[sec] = {"max":np.ma.zeros(column_MLS1['15-25'].shape[0]),
                       "median":np.ma.zeros(column_MLS1['15-25'].shape[0]),
                       "mean":np.ma.zeros(column_MLS1['15-25'].shape[0])}
    jd = 0
    day0 = date(2022,1,27)
    day1 = date(2022,6,6)
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
        center_MLS[sec]["max"][jd] = alts[np.ma.argmax(col)]
        center_MLS[sec]["mean"][jd] = np.ma.sum(col * alts)/np.ma.sum(col)
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
date_format = mdates.DateFormatter('%b-%d')    
for sec in secs:
    jg += 1
    ax = plt.subplot(1,3,jg)
    ax.plot(xx,center_MLS[sec]["max"],xx,center_MLS[sec]["mean"],xx,center_MLS[sec]["median"])
    ax.set_title('MLS '+sec)
    ax.xaxis_date()   
    ax.xaxis.set_major_formatter(date_format)
    ax.legend(('max','mean','median'))
    ax.set_ylim(23,26.5)
plt.show
fig.autofmt_xdate()
```

```python
wl = 21
fo = 2
mode = 'interp'
l1 = 60
l2 = 64
ff0_mls = {}
ff1_mls = {}
ff1b_mls = {}
center = "mean"

for sec in secs:
    # ff0 fit to the function and ff1 to its derivative
    ff0_mls[sec] = np.ma.zeros(len(center_MLS[sec][center]))
    ff1_mls[sec] = np.ma.zeros(len(center_MLS[sec][center]))
    ff1b_mls[sec] = np.ma.zeros(len(center_MLS[sec][center]))
    ff0_mls[sec] = sps.savgol_filter(center_MLS[sec][center],wl,fo,deriv=0,mode=mode)
    ff1_mls[sec] = sps.savgol_filter(center_MLS[sec][center],wl,fo,deriv=1,mode=mode)
    ff1b_mls[sec][1:-1] = 0.5*(ff0_mls[sec][2:] - ff0_mls[sec][:-2])
    ff1b_mls[sec][0] = ff0_mls[sec][1]-ff0_mls[sec][0]
    ff1b_mls[sec][-1] = ff0_mls[sec][-1]-ff0_mls[sec][-2]
    # Conversion of the derivative m/day
    ff1_mls[sec] *= 1000
    ff1b_mls[sec] *= 1000
    
fig = plt.figure(figsize=(15,6))
xx = mdates.date2num(days)
jg = 0
date_format = mdates.DateFormatter('%b-%d')    
for sec in secs:
    jg += 1
    ax0 = plt.subplot(2,3,jg)
    ax0.plot(xx,center_MLS[sec][center],xx,ff0_mls[sec])
    ax0.set_title('MLS '+sec)
    ax0.xaxis_date()   
    ax0.xaxis.set_major_formatter(date_format)
    ax1 = plt.subplot(2,3,jg+3)
    ax1.plot(xx,ff1_mls[sec],xx,wdiab[sec])
    ax1.xaxis_date()   
    ax1.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()
```

```python
column_CALIOP.shape
```

### Figure 2 komposit


ACHTUNG: This composit requires the preparation scripts of all the components to have run previously

```python
from mpl_toolkits.axes_grid1 import make_axes_locatable
fig = plt.figure(constrained_layout=True,figsize=(16,12))
fs = 14
fsl = 14
gs0 = fig.add_gridspec(3,1)
gs1 = gs0[0].subgridspec(1,3,width_ratios=[9,0.3,5])
gs2 = gs0[1].subgridspec(1,3,width_ratios=[9,0.3,5])
gs3 = gs0[2].subgridspec(1,5,width_ratios=[3,0.2,3,0.2,3])
xxe = mdates.date2num(days_e)
xx = mdates.date2num(days)
alts_edge = combinat_CALIOP['attr']['alts_edge']
alts_mls = MLS['attr']['alts_z']
vmax1 = {'all':3,'15-25':6,'05-15':6}
date_format = mdates.DateFormatter('%b-%d')

ax0 = fig.add_subplot(gs1[0])
sec = '05-15'
im0 = ax0.pcolormesh(xxe,alts_edge,column_CALIOP1[sec].T,cmap='gist_ncar',vmax=vmax1[sec],vmin=0)
CS0 = ax0.contour(xx,alts_mls,gaussian_filter(column_MLS1[sec],1).T*1.e6,linewidths=3,colors='pink')
ax0.clabel(CS0,inline=1,fontsize=fs)
ax0.set_ylim(22,28)
ax0.xaxis_date()
ax0.xaxis.set_major_formatter(date_format)
#ax0.set_xlabel('Day in 2022')
ax0.set_ylabel('Altitude (km)',fontsize=fs) 
ax0.grid(True)
ax0.set_title('CALIOP 532 nm scattering ratio (colour) + MLS water vapour (ppmv) in zonal band 5°S-15°S',fontsize=fs)
divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='3%', pad=0.1) # higher pad shrinks in width (??)
fig.colorbar(im0, cax=cax, orientation='vertical')

ax1 = fig.add_subplot(gs1[2])
ax1.plot(xx,ff1[sec],xx,wdiab[sec],xx,wadiab[sec],xx,ff1[sec]-wdiab[sec]-wadiab[sec],xx,ff1_mls[sec],linewidth=3)
ax1.set_title('Vertical motion in 5°S-15°S',fontsize=fs)
ax1.set_ylabel('w (m per day)',fontsize=fs)
ax1.xaxis_date()   
ax1.xaxis.set_major_formatter(date_format)
ax1.legend(['$w_C$','$w_R$','$w_{adiab}$','$w_S$','$w_{MLS}$'],fontsize=fsl)
ax1.set_ylim(-130,50)
ax1.grid(True)
ax1.tick_params(axis='x', rotation=30) 

ax2 = fig.add_subplot(gs2[0])
sec = '15-25'
im2 = ax2.pcolormesh(xxe,alts_edge,column_CALIOP1[sec].T,cmap='gist_ncar',vmax=vmax1[sec],vmin=0)
CS2 = ax2.contour(xx,alts_mls,gaussian_filter(column_MLS1[sec],1).T*1.e6,linewidths=3,colors='pink')
ax2.clabel(CS2,inline=1,fontsize=fs)
ax2.set_ylim(22,28)
ax2.xaxis_date()
ax2.xaxis.set_major_formatter(date_format)
ax2.set_xlabel('Day in 2022',fontsize=fs)
ax2.set_ylabel('Altitude (km)',fontsize=fs) 
ax2.grid(True)
ax2.set_title('CALIOP 532 nm scattering ratio (colour) + MLS water vapour (ppmv) in zonal band 15°S-25°S',fontsize=fs)
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='3%', pad=0.1) # higher pad shrinks in width (??)
fig.colorbar(im2, cax=cax, orientation='vertical')

ax3 = fig.add_subplot(gs2[2])
ax3.plot(xx,ff1[sec],xx,wdiab[sec],xx,wadiab[sec],xx,ff1[sec]-wdiab[sec]-wadiab[sec],xx,ff1_mls[sec],linewidth=3)
ax3.set_title('Vertical motion in 15°S-25°S',fontsize=fs)
ax3.set_ylabel('w (m per day)',fontsize=fs)
ax3.xaxis_date()   
ax3.xaxis.set_major_formatter(date_format)
ax3.legend(['$w_C$','$w_R$','$w_{adiab}$','$w_S$','$w_{MLS}$'],fontsize=fs)
ax3.set_ylim(-130,50)
ax3.grid(True)
ax3.tick_params(axis='x', rotation=30) 

ax4 = fig.add_subplot(gs3[0])
ax4.plot(xx,rada['05-15'],xx,rada['15-25'],linewidth=3)
ax4.xaxis_date()   
ax4.xaxis.set_major_formatter(date_format)
ax4.set_ylabel('Aerosol radius (µm)',fontsize=fs)
ax4.set_title('Aerosol radius from fall speed',fontsize=fs)
ax4.legend(('5°S-15°S','15°S-25°S'),fontsize=fs)
ax4.grid(True)
#ax4.set_aspect(22)
ax4.tick_params(axis='x', rotation=30) 

ax6 = fig.add_subplot(gs3[4])
lats = combinat_CALIOP['attr']['lats']
cosfac = np.cos(np.deg2rad(combinat_CALIOP['attr']['lats']))
jy1 = np.where(lats >= -15)[0][0]
jy2 = np.where(lats > -5)[0][0]
factor = cosfac[jy1:jy2]/np.sum(cosfac[jy1:jy2])
ax6.plot(xx_c,np.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1)/
         np.sum(column_CALIOP[:,jy1:jy2]*factor[np.newaxis,:],axis=1),lw=3)
jy1 = np.where(lats >= -25)[0][0]
jy2 = np.where(lats > -15)[0][0]
factor = cosfac[jy1:jy2]/np.sum(cosfac[jy1:jy2])
ax6.plot(xx_c,np.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1)/
         np.sum(column_CALIOP[:,jy1:jy2]*factor[np.newaxis,:],axis=1),lw=3)
jy1 = 0
jy2 = column_CALIOP.shape[1]
factor = cosfac[jy1:jy2]/np.sum(cosfac[jy1:jy2])
ax6.plot(xx_c,np.sum(smooth_OMPS[:,jy1:jy2]*factor[np.newaxis,:],axis=1)/
         np.sum(column_CALIOP[:,jy1:jy2]*factor[np.newaxis,:],axis=1),lw=3)
ax6.xaxis_date()
ax6.xaxis.set_major_formatter(date_format)
ax6.grid(True)
ax6.legend(('5°S-15°S','15°S-25°S','20°N-35°S'),fontsize=fsl)
ax6.set_ylabel('AOD / Total BS (str)',fontsize=fs)
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
```python

```

