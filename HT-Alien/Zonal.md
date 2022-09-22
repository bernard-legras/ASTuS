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
# Analysis of Alien plume after Hunga Tonga eruption
<!-- #endregion -->

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr


Location of the volcano: 20°34' S and 175°23' W or -20.57 and 184.62 or 175.38


##### Import

```python
from datetime import datetime, timedelta, date
#import os
import numpy as np
#import io107
import matplotlib.pyplot as plt
import gzip,pickle
#import copy
import constants as cst
#import cartopy.crs as ccrs
from matplotlib import gridspec
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter
```

<!-- #region tags=[] -->
## Analysis of the mean wind conditions and heating rates during the period 15 January - 30 April  
<!-- #endregion -->

### Load and reshape for plots

```python
with gzip.open('zonDailyMean-all.pkl','rb') as f:
    zonMean = pickle.load(f)
```

```python
varList = ['U','ASR','Z','PT','T','LPV']
top = 18 # zscale[18] = 40 km
bottom = 61 # zscale[60] = 16 km
Zmeans = {}
day0 = date(2022,1,1)
days = []
for var in varList:
    # trick to extract the shape
    Zmeans[var] = np.empty(shape=[(a,b,c) for a,(b,c) in [(len(zonMean['mean']),zonMean['mean'][day0]['U'].shape)]][0])
i = 0
# Views
Z = Zmeans['Z']
U = Zmeans['U']
θ = Zmeans['PT']
T = Zmeans['T']
LPV = Zmeans['LPV']
ASR = Zmeans['ASR']
for day in zonMean['mean']:
    days.append(day)
    for var in ['U','Z','PT','T','LPV']:
        Zmeans[var][i,...] = zonMean['mean'][day][var]
    Zmeans['ASR'][i,...] = zonMean['mean'][day]['ASSWR'] + zonMean['mean'][day]['ASLWR']
    i += 1
# Conversion of Z into kilometers
Z /= 1000
# Conversion of ASR into K/day
ASR *= 86400
# Rescaling of Lait PV with 600**4
LPV *= 600**4
```

Print the days of the bounding interval 15 Jan to 15 Mar

```python
print(days[14],days[73])
d1=14
d2=74
```

### Derive secondary quantities 

```python
lats = zonMean['attr']['lats']
```

#### Angular speed


Assuming spherical earth with equator 40,000 km (exact value is 40,075.0167 km). The angular rotation is given in degree per day

```python
deglon = np.cos(np.deg2rad(lats))*4e7/360
ω = Zmeans['ω'] = Zmeans['U'] / deglon * 86400
```

#### N2


The result is in s**-1

```python
N2 = Zmeans['N2'] = np.empty(shape = Zmeans['U'].shape)
N2[:,1:-1,:] = (np.log(θ[:,2:,:]/θ[:,:-2,:]))/ \
                         (Z[:,2:,:]-Z[:,:-2,:])
N2[:,0,:] = (np.log(θ[:,1,:]/θ[:,0,:]))/ \
                         (Z[:,1,:]-Z[:,0,:])
N2[:,-1,:] = (np.log(θ[:,-1,:]/θ[:,-2,:]))/ \
                         (Z[:,-1,:]-Z[:,-2,:])
N2 *= cst.g/1000
```

#### Meridional and vertical shear


Here we use the fact that latitudes are spaced by one degree
The horizontal and vertical shear are in 1/s.

```python
deg2m = 4.e7/180
dUdy = Zmeans['dUdy'] = np.empty(shape = Zmeans['U'].shape)
dUdy[...,1:-1] = (U[...,2:]-U[...,:-2])/2
dUdy[...,0] = (U[...,1]-U[...,0])
dUdy[...,-1] = (U[...,-1]-U[...,-2])
dUdy /= deg2m
dUdz = Zmeans['dUdz'] = np.empty(shape = Zmeans['U'].shape)
dUdz[:,1:-1,:] = (U[:,2:,:]-U[:,:-2,:])/(Z[:,2:,:]-Z[:,:-2,:])
dUdz[:,0,:] = (U[:,1,:]-U[:,0,:])/(Z[:,1,:]-Z[:,0,:])
dUdz[:,-1,:] = (U[:,-1,:]-U[:,-2,:])/(Z[:,-1,:]-Z[:,-2,:])
dUdz /= 1000 
```

#### Meridional and vertical shear (in angular rotation)


Here we use the fact that latitudes are spaced by one degree
The horizontal shear is obtained in 1/day and the vertical shear in degree per day per km.

```python
dωdy = Zmeans['dωdy'] = np.empty(shape = Zmeans['U'].shape)
dωdy[...,1:-1] = (ω[...,2:]-ω[...,:-2])/2
dωdy[...,0] = (ω[...,1]-ω[...,0])
dωdy[...,-1] = (ω[...,-1]-ω[...,-2])
dωdz = Zmeans['dωdz'] = np.empty(shape = Zmeans['U'].shape)
dωdz[:,1:-1,:] = (ω[:,2:,:]-ω[:,:-2,:])/(Z[:,2:,:]-Z[:,:-2,:])
dωdz[:,0,:] = (ω[:,1,:]-ω[:,0,:])/(Z[:,1,:]-Z[:,0,:])
dωdz[:,-1,:] = (ω[:,-1,:]-ω[:,-2,:])/(Z[:,-1,:]-Z[:,-2,:])
```

<!-- #region tags=[] -->
#### Mean ascent rate in m / day due to heating
<!-- #endregion -->

dZdθ is in m/day (as it is not calculated from Z but from N2 where conversion from km is applied)

```python
dZdθ = Zmeans['dZdθ'] = cst.g /(N2 * θ)
DZDt = Zmeans['DZDt'] = ASR * (θ/T) * dZdθ
```

#### Adiabatic zonal ascent rate


The sampling is daily in the input file, hence the finite differences are also daily
dZdθ is calculated in m/K 

```python
# Incomplete version
DZDtA = Zmeans['DZDtAdiab'] = np.empty(shape = Zmeans['U'].shape)
DZDtA[1:-1,...] = - dZdθ[1:-1,...] * (θ[2:,...]-θ[:-2,...])/2
DZDtA[0,...] = - dZdθ[0,...] * (θ[1,...]-θ[0,...])
DZDtA[-1,...] = - dZdθ[-1,...] * (θ[-1,...]-θ[-2,...])
```

```python
DZDtA = Zmeans['DZDtAdiab'] = np.empty(shape = Zmeans['U'].shape)
DZDtA[1:-1,...] = - dZdθ[1:-1,...] * (θ[2:,...]-θ[:-2,...])/2 + (Z[2:,...]-Z[:-2,...])*1000/2
DZDtA[0,...] = - dZdθ[0,...] * (θ[1,...]-θ[0,...]) + (Z[1,...]-Z[0,...])*1000
DZDtA[-1,...] = - dZdθ[-1,...] * (θ[-1,...]-θ[-2,...]) + (Z[-1,...]-Z[-2,...])*1000
```

### Mean quantities over the period defined by d1 d2

```python
ZGmeans = {}
for var in Zmeans:
    ZGmeans[var] = np.mean(Zmeans[var][d1:d2,...],axis=0)
Um = ZGmeans['U']
dUdym = ZGmeans['dUdy']
dUdzm = ZGmeans['dUdz']
Zm = ZGmeans['Z']
θm = ZGmeans['PT']
Tm = ZGmeans['T']
dωdym = ZGmeans['dωdy']
dωdzm = ZGmeans['dωdz']
ωm = ZGmeans['ω']
N2m = ZGmeans['N2']
ASRm = ZGmeans['ASR']
DZDtm = ZGmeans['DZDt']
LPVm = ZGmeans['LPV']
# why this is necessary is mysterious, CHECK THE POSSIBLE CONFLICT
DZDtm = np.mean(DZDt,axis=0)
dZdθm = ZGmeans['dZdθ']
DZDtAdiabm = ZGmeans['DZDtAdiab']
N2m = np.clip(N2m,0.,1000.)
```

#### Calculation of the Rayleigh criterion (from the mean U)

```python
β = (2 * cst.Omega / cst.REarth) * np.cos(np.deg2rad(lats))
Qm = np.empty(shape = Um.shape)
Qm[:,1:-1] = (dUdym[:,2:]-dUdym[:,:-2])/2
Qm[:,0] = (dUdym[:,1]-dUdym[:,0])
Qm[:,-1] = (dUdym[:,-1]-dUdym[:,-2])
Qm /= deg2m
Rayleigh = β[np.newaxis,:] - Qm
```

#### Calculation of Lait PV meridional gradient

```python
dLPVdym = np.empty(shape = Um.shape)
dLPVdym[:,1:-1] = (LPVm[:,2:]-LPVm[:,:-2])/2
dLPVdym[:,0] = (LPVm[:,1]-LPVm[:,0])
dLPVdym[:,-1] = (LPVm[:,-1]-LPVm[:,-2])
#dLPVdym/= deg2m
```

<!-- #region jp-MarkdownHeadingCollapsed=true jp-MarkdownHeadingCollapsed=true tags=[] -->
#### Check for the problem with DZDt
<!-- #endregion -->

```python
Zmeans.keys()
```

```python
dZdθ = Zmeans['dZdθ'] = cst.g /(N2 * θ)
DZDt = Zmeans['DZDt'] = np.empty(shape = Zmeans['U'].shape)
DZDt = ASR * dZdθ * (θ/T)
#DZDt[:,1:-1,:] = 1000 * ASR[:,1:-1,:] * (θ[:,1:-1,:]/T[:,1:-1,:]) * (Z[:,2:,:]-Z[:,:-2,:])/(θ[:,2:,:]-θ[:,:-2,:])
#DZDtm = np.mean(DZDt,axis=0)
plt.imshow(DZDtm[top:bottom,:],cmap=mymap)
plt.colorbar()
plt.show()
dd = dZdθm * ASRm * (θm/Tm)
plt.imshow(dd[top:bottom,:],cmap=mymap)
plt.colorbar()
plt.show()
```

<!-- #region tags=[] -->
### Visualization of the time average means
<!-- #endregion -->

```python
import matplotlib.colors as colors
listcolors=['#161d58','#253494','#2850a6','#2c7fb8','#379abe','#41b6c4',
            '#71c8bc','#a1dab4','#d0ecc0','#ffffcc','#fef0d9','#fedeb1',
            '#fdcc8a','#fdac72','#fc8d59','#ef6b41','#e34a33','#cb251a',
            '#b30000','#7f0000']
listcolors_sw=[listcolors[1],listcolors[0],listcolors[3],listcolors[2],\
               listcolors[5],listcolors[4],listcolors[7],listcolors[6],\
               listcolors[9],listcolors[8],listcolors[11],listcolors[10],\
               listcolors[13],listcolors[12],listcolors[15],listcolors[14],\
               listcolors[17],listcolors[16],listcolors[19],listcolors[18]]
mymap=colors.ListedColormap(listcolors)
mymap_sw=colors.ListedColormap(listcolors_sw)
```

```python
cmap = mymap
ymin = 18
ymax = 30
θmin = 400
θmax = 900
shading = 'gouraud'
ymax = 34
Umin = -30
Umax = 20
figsave = False

Xlats = np.reshape(np.tile(lats,137),(137,56))
fig = plt.figure(figsize=(18,16*(5/3)))
col = 3
row = 5
# fig 1
plt.subplot(row,col,1)
im=plt.pcolormesh(lats,Zm,Um,cmap=cmap,vmin=-35,vmax=20,shading=shading)
plt.colorbar(im)
plt.contour(Xlats,Zm,Um,levels=np.arange(-30,20,5),linewidths=(1,1,1,1,1,1,4,1,1,1))
#plt.clabel(CS,inline=1,fontsize=10)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title('Zonal wind (m/s)')

# fig 2
plt.subplot(row,col,2)
im=plt.pcolormesh(lats,Zm,dUdym,cmap=cmap,vmin=-1.e-5,vmax=2.e-5,shading=shading)
plt.contour(Xlats,Zm,dUdym,levels=np.arange(-0.5,2.,0.5)*1.e-5,linewidths=(1,4,1,1,1))
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.colorbar(im)
plt.title('Zonal wind meridional shear(1/s)')

# fig 3
plt.subplot(row,col,3)
im=plt.pcolormesh(lats,Zm,dUdzm,cmap=cmap,vmin=-1.e-2,vmax=1.5e-2,shading=shading)
plt.contour(Xlats,Zm,dUdzm,levels=np.arange(-0.5,1.5,0.5)*1.e-2,linewidths=(1,4,1,1))
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.colorbar(im)
plt.title('Zonal wind vertical shear(1/s)')

# fig 4 plot of the angular velocity in degree per day
plt.subplot(row,col,4)
im=plt.pcolormesh(lats,Zm,ωm,cmap=cmap,shading=shading,vmin=-30,vmax=10)
plt.contour(Xlats,Zm,ωm,(-30,-25,-20,-15,-10,-5,0,5,10),linewidths=(1,1,1,1,1,1,4,1,1))
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title('Angular speed (degree/day)')

# fig 5
plt.subplot(row,col,5)
im=plt.pcolormesh(lats,Zm,dωdym,cmap=cmap,vmin=-2,vmax=3,shading=shading)
plt.contour(Xlats,Zm,dωdym,np.arange(-1.5,3.,0.5),linewidths=(1,1,1,4,1,1,1,1,1))
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title('Angular speed mer. gradient (1/day)')

# fig 6
plt.subplot(row,col,6)
im=plt.pcolormesh(lats,Zm,dωdzm,cmap=cmap,vmin=-7,vmax=12.5,shading=shading)
plt.contour(Xlats,Zm,dωdzm,np.arange(-6,12.1,2),linewidths=(1,1,1,4,1,1,1,1,1,1))       
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title('Angular speed vert. gradient (degree/day/km)')

# fig 7
plt.subplot(row,col,7)
im=plt.pcolormesh(lats,Zm,np.sqrt(N2m),cmap=cmap,vmin=0.021,vmax=0.028,shading=shading)
plt.contour(Xlats,Zm,np.sqrt(N2m),np.arange(0.021,0.0280,0.001))       
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title('N (1/s)')
          
# fig 8
plt.subplot(row,col,8)
im=plt.pcolormesh(lats,Zm,Tm,cmap=cmap,vmin=200,vmax=240,shading=shading)
plt.contour(Xlats,Zm,Tm,np.arange(200,240,5)) 
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title('Temperature (K)')

# fig 9
plt.subplot(row,col,9)
im=plt.pcolormesh(lats,Zm,θm,cmap=cmap,vmin=400,vmax=900,shading=shading)
plt.contour(Xlats,Zm,θm,np.arange(400,900,50))       
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title('Temperature potentielle (K)')

# old fig 10
#plt.subplot(row,col,10)
#im=plt.pcolormesh(lats,Zm,ASRm,cmap=cmap,vmax=1,vmin=-0.5,shading=shading)
#plt.contour(Xlats,Zm,ASRm,np.arange(-0.4,1.,0.2),linewidths=(1,1,4,1,1,1,1)) 
#plt.colorbar(im)
#plt.ylim(ymin,ymax)
#plt.grid(True)
#plt.ylabel('Altitude (km)')
#plt.title('Heating rate (K/day)')

# fig 10
plt.subplot(row,col,10)
im=plt.pcolormesh(lats,Zm,ASRm*θm/Tm,cmap=cmap,vmax=4,vmin=-2,shading=shading)
plt.contour(Xlats,Zm,ASRm*θm/Tm,np.arange(-1.5,4,0.5),linewidths=(1,1,1,4,1,1,1,1,1,1,1)) 
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title('Radiative Dθ/Dt (K/day)')

# fig 11
plt.subplot(row,col,11)
#im=plt.pcolormesh(lats,Zm[t:b,:],DZDtm[t:b,:],cmap=cmap,shading=shading)
im=plt.pcolormesh(lats,Zm,dZdθm * ASRm * (θm/Tm),vmin=-50,vmax=100,cmap=cmap,shading=shading)
plt.contour(Xlats,Zm,dZdθm * ASRm * (θm/Tm),np.arange(-25,100,25),linewidths=(1,4,1,1,1))
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title('Diabatic radiative DZ/Dt (m/day)')

# fig 12 
plt.subplot(row,col,12)
im=plt.pcolormesh(lats,Zm,DZDtAdiabm,cmap=cmap,vmax=10,vmin=-7,shading=shading)
plt.contour(Xlats,Zm,DZDtAdiabm,np.arange(-6,10,2),linewidths=(1,1,1,4,1,1,1,1)) 
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title(' Adiabatic DZ/Dt (m/day)')

# fig 13
plt.subplot(row,col,13)
im=plt.pcolormesh(lats,Zm,1.e11*Rayleigh,cmap=mymap,vmax=4,vmin=0,shading=shading)
plt.contour(Xlats,Zm,1.e11*Rayleigh,np.arange(0.5,4,0.5)) 
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title(r'Rayleigh criterion (10$^{-11}$ s$^{-1}$ m$^{-1}$)')

# fig 14
plt.subplot(row,col,14)
im=plt.pcolormesh(lats,Zm,1.e6*LPVm,cmap=mymap,vmin=-60,vmax=50,shading=shading)
plt.contour(Xlats,Zm,1.e6*LPVm,np.arange(-50,50,10),linewidths=(1,1,1,1,1,4,1,1,1,1)) 
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
#plt.title('Lait PV (10$^{-16}$  m$^2$ K$^{-3}$ s$^{-1}$ kg$^{-1}$) ')
plt.title(r'Lait PV: PV $(600/θ)^4$ (PVU) ')

# fig 15
plt.subplot(row,col,15)
im=plt.pcolormesh(lats,Zm,1.e6*dLPVdym,vmin=-3,vmax=8,cmap=mymap,shading=shading)
plt.contour(Xlats,Zm,1.e6*dLPVdym,np.arange(-2,8),linewidths=(1,1,4,1,1,1,1,1,1,1)) 
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title('Lait PV gradient (PVU/degree)) ')

#fig.autofmt_xdate()
if figsave: plt.savefig('zonal14.png',dpi=300,bbox_inches='tight')
plt.show()
```

Notice that the ascent rate calculated from the mean profile is not exactly the same as the mean ascent rat.

```python
im=plt.pcolormesh(lats,Zm,DZDtm,vmin=-50,vmax=100,cmap=cmap,shading=shading)
plt.ylim(ymin,ymax)
plt.colorbar(im)
plt.show()
im=plt.pcolormesh(lats,Zm,dZdθm * ASRm * (θm/Tm),vmin=-50,vmax=100,cmap=cmap,shading=shading)
plt.ylim(ymin,ymax)
plt.colorbar(im)
plt.show()
```

### Time sequences in 10° average sections in latitude


#### Average in latitude ranges 

```python
# Here we fix the latitude range and determine the boundaries
secs = {'15-25','05-15','all'}
jy0 = {'15-25':np.where(lats>=-25)[0][0],'05-15':np.where(lats>-15)[0][0],'all':0}
jy1 = {'15-25':np.where(lats>=-15)[0][0],'05-15':np.where(lats>-5)[0][0],'all':len(lats)}

# The averages are stored in a dictionary of sections
sections = {}
for sec in secs: 
    sections[sec]={}
    for var in Zmeans:
        sections[sec][var] = np.mean(Zmeans[var][...,jy0[sec]:jy1[sec]],axis=2)

    # Special processing
    # Zmeans['DZDt'] must be corrupted
    sections[sec]['DZDt'] = np.mean(DZDt[...,jy0[sec]:jy1[sec]],axis=2)
    sections[sec]['N2'] = np.clip(sections[sec]['N2'],0.,1000.)
   
    # Temporal derivative from the smoothed version of θ & Z
    θ2 = gaussian_filter(sections[sec]['PT'],3)
    Z2 = gaussian_filter(sections[sec]['Z'],3)
    DZDtAdiab2 = np.empty(shape=sections[sec]['DZDtAdiab'].shape)
    DZDtAdiab2[1:-1,:] = - (θ2[2:,:]-θ2[:-2,:])/2
    DZDtAdiab2[0,:] = - (θ2[1,:]-θ2[0,:])
    DZDtAdiab2[-1,:] = - (θ2[-1,:]-θ2[-2,:])
    DZDtAdiab3 = np.empty(shape=sections[sec]['DZDtAdiab'].shape)
    DZDtAdiab3[1:-1,:] = - (Z2[2:,:]-Z2[:-2,:])/2
    DZDtAdiab3[0,:] = - (Z2[1,:]-Z2[0,:])
    DZDtAdiab3[-1,:] = - (Z2[-1,:]-Z2[-2,:])
    sections[sec]['DZDtAdiab2'] = sections[sec]['dZdθ'] * DZDtAdiab2 - DZDtAdiab3*1000
```

```python
sections[sec].keys()
```

Backup

```python
with gzip.open('sections_ERA5.pkl','wb') as f:
    pickle.dump([sections,{'days':days}],f,protocol=pickle.HIGHEST_PROTOCOL)
```

```python tags=[]
sec = '05-15'
figsave = False
fig,axes = plt.subplots(figsize=(16,6),nrows=2,ncols=2,sharex=True,sharey=True)
xx = mdates.date2num(days)
date_format = mdates.DateFormatter('%d-%b')
Xxx = np.reshape(np.tile(xx,137),(137,len(xx)))
Zl = sections[sec]['Z']

# panel 0, angular rotation
ax0 = axes[0,0]
# buf defined as a dictionary to be used i!n a composite with the two latitude ranges below
#try: _=len(bufω)
#except: bufω={}
#bufω[sec] = gaussian_filter(ωl.T,3)
bufω = gaussian_filter(sections[sec]['ω'].T,3)
im0 = ax0.pcolormesh(xx,Zl.T,bufω,vmin=-30,vmax=20,shading='gouraud',cmap=mymap)
ax0.contour(Xxx,Zl.T,bufω,levels=np.arange(-25,20,5),linewidths=(1,1,1,1,1,4,1,1,1))
ax0.set_ylim(18,30)
ax0.grid(True)
ax0.set_xlabel('Time')
ax0.set_ylabel('Altitude (km)')
ax0.xaxis_date()
ax0.xaxis.set_major_formatter(date_format)
ax0.set_title('Angular speed in '+sec+'S (degree/day)')
divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical')

# panel 1, diabatic ascent
ax1 = axes[1,0]
#try: _=len(bufw)
#except: bufw={}
#bufw[sec] = gaussian_filter(DZDtl.T,3)
bufw = gaussian_filter(sections[sec]['DZDt'].T,3)
im1 = ax1.pcolormesh(xx,Zl.T,bufw,shading='gouraud',vmin=-20,vmax=70,cmap=mymap)
ax1.contour(Xxx,Zl.T,bufw,levels=np.arange(-10,70,10),linewidths=(1,4,1,1,1,1,1,1))
ax1.set_ylim(18,30)
ax1.xaxis_date()
ax1.grid(False)
ax1.set_xlabel('Time')
ax1.set_ylabel('Altitude (km)')
ax1.xaxis.set_major_formatter(date_format)
ax1.set_title('Diabatic radiative ascent rate in '+sec+'S (m/day)')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')

# panel 2 potential temperature
ax2 = axes[0,1]
buf2 = gaussian_filter(sections[sec]['PT'].T,3)
im2 = ax2.pcolormesh(xx,Zl.T,buf2,shading='gouraud',vmin=400,vmax=850,cmap=mymap)
ax2.contour(Xxx,Zl.T,buf2,levels=np.arange(-350,850,50))
ax2.set_ylim(18,30)
ax2.xaxis_date()
ax2.grid(True)
ax2.set_xlabel('Time')
ax2.set_ylabel('Altitude (km)')
ax2.xaxis.set_major_formatter(date_format)
ax2.set_title('Potential temperature in '+sec+'S (K)')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')

# panel 3
ax3 = axes[1,1]
buf3 = sections[sec]['DZDtAdiab2'].T
im3 = ax3.pcolormesh(xx,Zl.T,buf3,shading='gouraud',vmin=-10,vmax=15,cmap=mymap)
ax3.set_ylim(18,30)
ax3.xaxis_date()
ax3.grid(True)
ax3.set_xlabel('Time')
ax3.set_ylabel('Altitude (km)')
ax3.xaxis.set_major_formatter(date_format)
ax3.set_title('Adiabatic ascent rate in '+sec+'S (m/day)')
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')

fig.autofmt_xdate()
if figsave: plt.savefig('time14-'+sec+'.png',dpi=144,bbox_inches='tight')
plt.show()
```

<!-- #region tags=[] -->
#### Interpolator
<!-- #endregion -->

**FIX THE PROBLEMS WITH INTERPOLATION**

```python
from scipy.interpolate import interp2d
buf1 = gaussian_filter(DZDtl,4)
inter2 = interp2d(Xxx.T[:,30:48],Zl[:,30:48],buf1[:,30:48],kind='linear')
```

```python
aa = np.empty(shape=Zl[:,30:48].shape)
for t in range(120):
    for kz in range(18):
        aa[t,kz] = inter2(xx[t],Zl[t,30+kz])
plt.imshow(aa.T,aspect=2,cmap=mymap)
plt.colorbar()
```

```python
plt.imshow(buf1[:,30:48].T,aspect=2,cmap=mymap)
```

```python
inter2(mdates.date2num(datetime(2022,2,12,4,56)),26)
```

#### Plots at 24, 25 and 26 kms (or the more approaching model levels that is 35, 37, 39)


Must be run after the average in latitude range above and takes the value of sec


Determining the level

```python
Zprof[35:40]
```

Plot the evolution on levels 35, 37, 39

```python
fig,axes = plt.subplots(figsize=(16,6),nrows=2,ncols=2,sharex=True,sharey=False)
xx = mdates.date2num(days)
date_format = mdates.DateFormatter('%d-%b')
labs = {35:'26.4 km',37:'25.3 km',39:'24.2 km'}
figsave = False

ωl = sections[sec]['ω']
DZDtl = sections[sec]['DZDt']
DZDtAdiab2l = sections[sec]['DZDtAdiab2']

ax0 = axes[0,0]
for kz in (35,37,39):
    ax0.plot(xx,ωl[:,kz],label=labs[kz])
ax0.set_ylabel('ω (degree/day)')
ax0.set_title('Angular rotation in '+sec+'S')
ax0.xaxis_date()
ax0.grid(True)
ax0.xaxis.set_major_formatter(date_format)
ax0.legend()

ax1 = axes[0,1]
for kz in (35,37,39):
    ax1.plot(xx,Zl[:,kz],label=labs[kz])
ax1.set_ylabel('Z (km)')
ax1.set_title('Altitude at model levels 35 37 and 39 in '+sec+'S')
ax1.xaxis_date()
ax1.grid(True)
ax1.xaxis.set_major_formatter(date_format)
ax1.legend()

ax2 = axes[1,0]
for kz in (35,37,39):
    ax2.plot(xx,DZDtl[:,kz],label=labs[kz])
ax2.set_ylabel('DZ/Dt (m/day)')
ax2.set_title('Diabatic radiative ascent in '+sec+'S')
ax2.xaxis_date()
ax2.grid(True)
ax2.xaxis.set_major_formatter(date_format)
ax2.legend()

ax3 = axes[1,1]
for kz in (35,37,39):
    ax3.plot(xx,DZDtAdiab2[:,kz],label=labs[kz])
ax3.set_ylabel('DZ/Dt (m/day)')
ax3.set_title('Aiabatic ascent in '+sec+'S')
ax3.xaxis_date()
ax3.grid(True)
ax3.xaxis.set_major_formatter(date_format)
ax3.legend()

fig.autofmt_xdate()
if figsave: plt.savefig('plot14-'+sec+'.png',dpi=144,bbox_inches='tight')
plt.show()
```

## Publication figures 


### Figs A1 & A2


Must be run after the average in latitude range above for both values of sec in order to generate buffers bufω and bufw
The enddate is the last one of the time plots.

```python
bufω.shape
```

```python
Zl.shape
```

```python
cmap = mymap
ymin = 18
ymax = 30
enddate = date(2022,7,26)
end = np.where([enddate < day for day in days])[0][0]
xx = mdates.date2num(days[:end])
date_format = mdates.DateFormatter('%d-%b')
Xxx = np.reshape(np.tile(xx,137),(137,len(xx)))
shading = 'gouraud'
Xlats = np.reshape(np.tile(lats,137),(137,56))
figsave = True
divgt = True

fig = plt.figure(constrained_layout=True,figsize=(14,10*14/16))
gs0 = fig.add_gridspec(2,1,height_ratios=(2,3))
gs1 = gs0[0].subgridspec(1,3)
gs2 = gs0[1].subgridspec(2,2)

ax4 = fig.add_subplot(gs1[0])
ax11 = fig.add_subplot(gs1[1])
ax12 = fig.add_subplot(gs1[2])
ax0 = fig.add_subplot(gs2[0,0])
ax1 = fig.add_subplot(gs2[0,1])
ax2 = fig.add_subplot(gs2[1,0])
ax3 = fig.add_subplot(gs2[1,1])

# a) Angular velocty
vmin = -30
if divgt: vmax = -vmin
else: vmax = 10
im4 = ax4.pcolormesh(lats,Zm,ωm,cmap=cmap,shading=shading,vmin=vmin,vmax=vmax)
ax4.contour(Xlats,Zm,ωm,(-30,-25,-20,-15,-10,-5,0,5,10),linewidths=(1,1,1,1,1,1,4,1,1))
divider = make_axes_locatable(ax4)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im4, cax=cax, orientation='vertical')
ax4.set_ylim(ymin,ymax)
ax4.grid(True)
ax4.set_ylabel('Altitude (km)')
#plt.xlabel('Latitude')
ax4.set_title('Angular speed (degree/day)')
ax4.annotate('a)',(-0.12,1.02),xycoords='axes fraction',fontsize=14)

# b) DZDt diabatic
vmax = 100
if divgt: vmin = -vmax
else: vmin = -50
im11 = ax11.pcolormesh(lats,Zm,dZdθm * ASRm * (θm/Tm),vmin=vmin,vmax=vmax,cmap=cmap,shading=shading)
ax11.contour(Xlats,Zm,dZdθm * ASRm * (θm/Tm),np.arange(-25,100,25),linewidths=(1,4,1,1,1))
divider = make_axes_locatable(ax11)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im11, cax=cax, orientation='vertical')
ax11.set_ylim(ymin,ymax)
ax11.grid(True)
#ax11.set_ylabel('Altitude (km)')
ax11.set_ylabel('  ')
ax11.set_yticklabels(' ')
ax11.set_xlabel('Latitude')
ax11.set_title('Diabatic radiative ascent rate (m/day)')
ax11.annotate('b)',(-0.,1.02),xycoords='axes fraction',fontsize=14)

# c) DZDt adiabatic
vmax = 5
if divgt: vmin = -vmax
else: vmin = -5
im12 = ax12.pcolormesh(lats,Zm,DZDtAdiabm,cmap=cmap,vmax=vmax,vmin=vmin,shading=shading)
ax12.contour(Xlats,Zm,DZDtAdiabm,np.arange(-6,10,2),linewidths=(1,1,1,4,1,1,1,1)) 
divider = make_axes_locatable(ax12)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im12, cax=cax, orientation='vertical')
ax12.set_ylim(ymin,ymax)
ax12.grid(True)
#ax12.set_ylabel('Altitude (km)')
ax12.set_ylabel('  ')
ax12.set_yticklabels(' ')
#plt.xlabel('Latitude')
ax12.set_title(' Adiabatic ascent rate (m/day)')
ax12.annotate('c)',(-0.,1.02),xycoords='axes fraction',fontsize=14)

# d) Angular velocity
sec ='05-15'
# panel 0, angular rotation
bufω = gaussian_filter(sections[sec]['ω'][:end,:].T,3)
Zl = sections[sec]['Z'][:end,:]
vmin = -30
if divgt: vmax = -vmin
else: vmax = 20
im0 = ax0.pcolormesh(xx,Zl.T,bufω,vmin=vmin,vmax=vmax,shading='gouraud',cmap=mymap)
ax0.contour(Xxx,Zl.T,bufω,levels=np.arange(-25,20,5),linewidths=(1,1,1,1,1,4,1,1,1))
ax0.set_ylim(18,30)
ax0.grid(True)
#ax0.set_xlabel('Day in 2022')
ax0.set_ylabel('Altitude (km)')
ax0.xaxis_date()
ax0.xaxis.set_major_formatter(date_format)
ax0.set_title('Angular speed in 15°S-5°S (degree/day)')
divider = make_axes_locatable(ax0)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im0, cax=cax, orientation='vertical')
ax0.annotate('d)',(-0.075,1.02),xycoords='axes fraction',fontsize=14)

# e) DZDt diab
bufw = gaussian_filter(sections[sec]['DZDt'][:end,:].T,3)
vmax = 70
if divgt: vmin = -vmax
else: vmin = -20
im1 = ax1.pcolormesh(xx,Zl.T,bufw,shading='gouraud',vmin=vmin,vmax=vmax,cmap=mymap)
ax1.contour(Xxx,Zl.T,bufw,levels=np.arange(-10,70,10),linewidths=(1,4,1,1,1,1,1,1))
ax1.set_ylim(18,30)
ax1.xaxis_date()
ax1.grid(False)
#ax1.set_xlabel('Day in 2022')
#ax1.set_ylabel('Altitude (km)')
ax1.set_ylabel('  ')
ax1.set_yticklabels('')
ax1.xaxis.set_major_formatter(date_format)
ax1.set_title('Diabatic radiative ascent rate in 15°S-5°S (m/day)')
divider = make_axes_locatable(ax1)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im1, cax=cax, orientation='vertical')
ax1.annotate('e)',(-0.0,1.02),xycoords='axes fraction',fontsize=14)

# f) Angular velocity
sec ='15-25'
# panel 0, angular rotation
bufω = gaussian_filter(sections[sec]['ω'][:end,:].T,3)
Zl = sections[sec]['Z'][:end,:]
vmin = -30
if divgt: vmax = -vmin
else: vmax = 20
im2 = ax2.pcolormesh(xx,Zl.T,bufω,vmin=vmin,vmax=vmax,shading='gouraud',cmap=mymap)
ax2.contour(Xxx,Zl.T,bufω,levels=np.arange(-25,20,5),linewidths=(1,1,1,1,1,4,1,1,1))
ax2.set_ylim(18,30)
ax2.grid(True)
ax2.set_xlabel('Day in 2022')
ax2.set_ylabel('Altitude (km)')
ax2.xaxis_date()
ax2.xaxis.set_major_formatter(date_format)
ax2.set_title('Angular speed in 25°S-15°S (degree/day)')
divider = make_axes_locatable(ax2)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im2, cax=cax, orientation='vertical')
ax2.annotate('f)',(-0.075,1.02),xycoords='axes fraction',fontsize=14)

# g) DZDt diab
bufw = gaussian_filter(sections[sec]['DZDt'][:end,:].T,3)
vmax = 70
if divgt: vmin = -vmax
else: vmin = -20
im3 = ax3.pcolormesh(xx,Zl.T,bufw,shading='gouraud',vmin=vmin,vmax=vmax,cmap=mymap)
ax3.contour(Xxx,Zl.T,bufw,levels=np.arange(-10,70,10),linewidths=(1,4,1,1,1,1,1,1))
ax3.set_ylim(18,30)
ax3.xaxis_date()
ax3.grid(False)
ax3.set_xlabel('Day in 2022')
#ax3.set_ylabel('Altitude (km)')
ax3.set_ylabel('  ')
ax3.set_yticklabels('')
ax3.xaxis.set_major_formatter(date_format)
ax3.set_title('Diabatic radiative ascent rate in 25°S-15°S (m/day)')
divider = make_axes_locatable(ax3)
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im3, cax=cax, orientation='vertical')
ax3.annotate('g)',(-0.0,1.02),xycoords='axes fraction',fontsize=14)

if figsave: 
    plt.savefig("meteo-context.png",dpi=300,bbox_inches='tight')
    plt.savefig("meteo-context.pdf",dpi=300,bbox_inches='tight')
plt.show()
```

### Rayleigh criterion

```python
# fig 15
figsave = True
divgt = True
fig = plt.figure(figsize=(5,4.5))
vmax = 6
if divgt: vmin = -vmax
else: vmin = -3
im=plt.pcolormesh(lats,Zm,1.e6*dLPVdym,vmin=vmin,vmax=vmax,cmap=mymap,shading=shading)
plt.contour(Xlats,Zm,1.e6*dLPVdym,np.arange(-2,8),linewidths=(1,1,4,1,1,1,1,1,1,1)) 
plt.colorbar(im)
plt.ylim(ymin,ymax)
plt.grid(True)
plt.ylabel('Altitude (km)')
plt.title('Lait PV gradient (PVU/degree)) ')
if figsave: plt.savefig('RayleighLPV.png',dpi=300,bbox_inches='tight')
plt.show
```

```python

```
