---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.14.0
  kernelspec:
    display_name: p310
    language: python
    name: p310
---

# IMS combi and movie maker


Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr


This notebook produces the combined SA N + D views of the IMS/IASI product and makes a movie of the series of images


## Initialisation

```python
import IMS
from datetime import date, timedelta, datetime
from matplotlib import pyplot as plt
import os
import cartopy.crs as ccrs
import gzip, pickle
from matplotlib.patches import Rectangle
```

```python
# Read the Metop-B crossings from IXION
with gzip.open('Metop-B_xings.pkl','rb') as f:
    xings = pickle.load(f)
```

## Production of the combined SA N + D images with a strip showing the orbit times  

```python
day0 = date(2022,1,13)
day1 = date(2022,4,30)
day = day0
# loop on days
while day <= day1:
    # read the IMS data
    try:
        dat = IMS.IMS(day)
        dat.read('SA','N')
        dat.read('SA','D')
    except:
        print('missing IMS data for',day)
        day += timedelta(days=1)
        continue
    # rescale maximum after the cut in IMS data
    if day < date(2022,3,14): vmax= 0.03
    else: vmax = 0.02
    # Get list of crossings for that day
    dd = datetime(day.year,day.month,day.day)
    Dxings = xings['D'][dd]
    Nxings = xings['N'][dd]
    # Generate figure framework
    fig = plt.figure(figsize=(12,5.25))
    gs = fig.add_gridspec(2,1)
    
    # Make axis for day part
    axD = fig.add_subplot(gs[0],projection=ccrs.PlateCarree())
    # Plot the image using IMS package
    imD = dat.show('SA_D',axf=axD,txt='',vmin=0.002,vmax=vmax,ylims=(-40,10),log=True,cmap=IMS.cmap3,show=False)
    # Make the background of the information strip
    axD.add_patch(Rectangle((-190,12),380,20,linewidth=1,edgecolor=None,facecolor='burlywood',clip_on=False))
    # Put the title in the strip
    axD.annotate(
        day.strftime('IMS/IASI SA OD       %d %B %Y       descending day orbits               \xA9 RAL & CNRS'),
        (-90,25),annotation_clip=False)
    # Add the times at which each orbit is crossing the equator
    lD = len(Dxings)
    for i in range(lD):
        hours = Dxings[i][0].seconds // 3600
        minutes = Dxings[i][0].seconds // 60 - 60 * hours
        # This complication is made to avoid overlaps of strings in the strip when the first and the last swaths of the day overlap
        offset = 0
        color = 'k'
        if lD == 15:
            if i == 0: 
                offset = -5
                color = 'red'
            if i == 14: offset = 5
        axD.annotate('{:02d}:{:02d}'.format(hours,minutes),(Dxings[i][1]-5+offset,15),color = color,annotation_clip=False)

    # Make axis for the night part (same as for the day part)
    axN = fig.add_subplot(gs[1],projection=ccrs.PlateCarree())
    imN = dat.show('SA_N',axf=axN,txt='',vmin=0.002,vmax=vmax,ylims=(-40,10),log=True,cmap=IMS.cmap3,show=False)
    axN.add_patch(Rectangle((-190,12),380,20,linewidth=1,edgecolor=None,facecolor='burlywood',clip_on=False))
    axN.annotate(
        day.strftime('IMS/IASI SA OD       %d %B %Y        ascending night orbits             \xA9 RAL & CNRS'),(-90,25),annotation_clip=False)
    lN = len(Nxings)
    for i in range(lN):
        hours = Nxings[i][0].seconds // 3600
        minutes = Nxings[i][0].seconds // 60 - 60 * hours
        offset = 0
        color = 'k'
        if lN == 15:
            if i == 0: 
                offset = -5
                color = 'red'
            if i == 14: offset = 5
        axN.annotate('{:02d}:{:02d}'.format(hours,minutes),(Nxings[i][1]-10+offset,15),color = color,annotation_clip=False)
          
    cbar_ax = fig.add_axes([0.93,0.2,0.01,0.6])
    if vmax == 0.03:
        cbar1 = fig.colorbar(imN,cax=cbar_ax,orientation='vertical',ticks=[2e-3,5e-3,1e-2,2e-2,3e-2])
        cbar1.ax.set_yticklabels([u'2 10$^{-3}$',u'5 10$^{-3}$',u'10$^{-2}$',u'2 10$^{-2}$',u'3 10$^{-2}$'])
    else:
        # colorbar wants to label 3, 4 & 6 x e-3 that we do not want. Hence to complication to mute it
        cbar2 = fig.colorbar(imN,cax=cbar_ax,orientation='vertical',ticks=[2e-3,3e-3,4e-3,5e-3,6e-3,1e-2,2e-2])
        cbar2.ax.set_yticklabels([u'2 10$^{-3}$','','',u'5 10$^{-3}$','',u'10$^{-2}$',u'2 10$^{-2}$'])
    plt.savefig(os.path.join('SA-combi',day.strftime('IMS-SAOD-%Y-%m-%d.png')),dpi=144,bbox_inches='tight')
    #plt.show()
    plt.close('all')                 
    print(day)
 
    day += timedelta(days=1)
```

## Production of the movie

```python
import imageio
from PIL import Image
images = []

day0 = date(2022,1,13)
day1 = date(2022,4,30)
day = day0
# loop on days
while day <= day1:
    try:
        #im = imageio.imread(os.path.join('SA-combi',day.strftime('IMS-SAOD-%Y-%m-%d.jpg')))
        im = Image.open(os.path.join('SA-combi',day.strftime('IMS-SAOD-%Y-%m-%d.png')))
    except:
        print('no image for',day)
    else:
        print(day,im.size)
        # paste into an image with suitable size for viewers with number of pixels as multiples of 16
        cc= Image.new('RGB',size=(1584,640),color=(255,255,255))
        cc.paste(im,(8,4))
        images.append(cc)
    day += timedelta(days=1)

imageio.mimsave('movie-SAOD.gif', images,fps=2)
imageio.mimsave('movie-SAOD.mp4', images,fps=2)

```
```python

```

