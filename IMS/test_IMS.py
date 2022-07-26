#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright or © or Copr. Pasquale Sellitto, Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr

Created on Thu Jun  2 18:26:29 2022

@author: Bernard Legras
"""
import IMS
from datetime import date, timedelta
from matplotlib import pyplot as plt
import os
import cartopy.crs as ccrs

# How to read
day = date(2022,2,2)
dat = IMS.IMS(day)
dat.read('SA','D')
# Simple plot with default colormap
dat.show('SA_D',vmin=0.002,vmax=0.02)
# Changing the color map
dat.show('SA_D',vmin=0.002,vmax=0.02,cmap=IMS.cmap2)
# With log scale
dat.show('SA_D',vmin=0.002,vmax=0.03,log=True,cmap=IMS.cmap2)
# Selecting a subdomain
dat.show('SA_D',vmin=0.002,vmax=0.03,cmap=IMS.cmap2,xlims=(-180,-30),ylims=(-40,0))

#%%
# Example with another day where is the spiral structure
# Unfortunately, it spans the datetine
day = date(2022,2,11)
dat = IMS.IMS(day)
dat.read('SA','N')
dat.show('SA_N',vmin=0.,vmax=0.02,cmap=IMS.cmap2,xlims=(-180,-120),ylims=(-40,0))

#%%
# Rotating the data
dat2 = dat.shift(0)
cm = 180
# Plotting the data again in the rotated full domain
dat2.show('SA_N',vmin=0.,vmax=0.02,cmap=IMS.cmap2,cm=cm)
# Extracting a subdomain
# ACHTUNG: Pay attention to the usage of cm. Weird but it is the way cartopy likes it.
dat2.show('SA_N',vmin=0.,vmax=0.02,cmap=IMS.cmap2,cm=cm,xlims=(140-cm,230-cm),ylims=(-40,0))
# Resizing the figure for better appearance
dat2.show('SA_N',figsize=(6,3),vmin=0.,vmax=0.02,cmap=IMS.cmap2,cm=cm,xlims=(140-cm,230-cm),ylims=(-40,0))
# Testing Pasquale's colormap 
dat2.show('SA_N',figsize=(6,3),vmin=0.,vmax=0.02,cmap=IMS.cmap3,cm=cm,xlims=(140-cm,230-cm),ylims=(-40,0))
#%%
# Now we load the day data for the same day
dat.read('SA','D')
# We combine the two plots into a single composite figure and plot a common colorbar
# This can be extended to many plots
# The band with orbit times can be produced in the same way by adjusting the subplot and using the room
# made to plot the band/ To be done.

fig, [ax0, ax1] = plt.subplots(figsize=(12,6),nrows=2,ncols=1,sharex=True,\
                      subplot_kw={"projection":ccrs.PlateCarree()})
im1=dat.show('SA_N',axf=ax0,vmin=0.,vmax=0.02,cmap=IMS.cmap2)
im2=dat.show('SA_D',axf=ax1,vmin=0.,vmax=0.02,cmap=IMS.cmap2)
fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.91,0.2,0.02,0.6])
fig.colorbar(im1,cax=cbar_ax,orientation='vertical')
plt.show()
