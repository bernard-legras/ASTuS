#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Copyright or © or Copr. Pasquale Sellitto, Bernard Legras (2022)
 
 bernard.legras@lmd.ipsl.fr

 This software is a computer program whose purpose is to perform Legendre
 transforms on the sphere.

 This software is governed by the CeCILL-C license under French law and
 abiding by the rules of distribution of free software.  You can  use,
 modify and/ or redistribute the software under the terms of the CeCILL-C
 license as circulated by CEA, CNRS and INRIA at the following URL
 "http://www.cecill.info".

 As a counterpart to the access to the source code and  rights to copy,
 modify and redistribute granted by the license, users are provided only
 with a limited warranty  and the software's author,  the holder of the
 economic rights,  and the successive licensors  have only  limited
 liability.

 In this respect, the user's attention is drawn to the risks associated
 with loading,  using,  modifying and/or developing or reproducing the
 software by the user in light of its specific status of free software,
 that may mean  that it is complicated to manipulate,  and  that  also
 therefore means  that it is reserved for developers  and  experienced
 professionals having in-depth computer knowledge. Users are therefore
 encouraged to load and test the software's suitability as regards their
 requirements in conditions enabling the security of their systems and/or
 data to be ensured and,  more generally, to use and operate it in the
 same conditions as regards security.

 The fact that you are presently reading this means that you have had
 knowledge of the CeCILL-C license and that you accept its terms.

Class to open and manipulate gridded IMS data provided by RAL 

Created on Mon May 30 18:34:02 2022
Adapted from a script of Pasquale Sellitto

@author: Bernard Legras
"""
import numpy as np
import os
from datetime import datetime,timedelta, date
from netCDF4 import Dataset
import socket
from matplotlib import colors
from matplotlib import cm as cmp
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

# Locate data source 
# This part needs to be adapted to the local installation 
if socket.gethostname() == 'satie':
    pathIMS = '/data/IMS'
elif socket.gethostname() == 'gort':
    pathIMS = '/dkol/data/IMS'
elif socket.gethostname() == 'Mentat':
    pathIMS = os.path.join('..','..','IMS')

trl = {'SA':'aot0','SO2':'so2'}

# new discrete color maps adapted to display sulfur with yellow at the upper end
fid = open('ims4.rgb')
fid.readlines(1)
fid.readlines(1)
reduced_colors = np.array([np.genfromtxt(x.rstrip('#*.\n').split('\t')) for x in fid.readlines()])/255
nr = reduced_colors.shape[0]
aa2 = np.repeat(reduced_colors,np.full(nr,12),axis=0)
cmap2 = colors.ListedColormap(aa2,'RWB',12*nr)

fid = open('ims5.rgb')
fid.readlines(1)
fid.readlines(1)
reduced_colors = np.array([np.genfromtxt(x.rstrip('#*.\n').split('\t')) for x in fid.readlines()])/255
nr = reduced_colors.shape[0]
aa3 = np.repeat(reduced_colors,np.full(nr,12),axis=0)
cmap3 = colors.ListedColormap(aa3,'RWB',12*nr)

class IMS_pure(object):
    def __init__(self):
        self.attr={}
        self.var={}

    def show(self,var,axf=None,vmax=0.02,vmin=0,cm=0,aspect=1,
             ylims=(-40,40),xlims=None,log=False,cmap=None,show=True,txt=None,
             xlocs=None,figsize=None,left=True,bottom=True):
        """ 
           Make nice charts with IMS, see the accompanying test for usage
	     examples. 
           By default, produces a global chart.
        """
        if axf == None:
            if figsize is None: fig = plt.figure(figsize=[12,3])
            else: fig = plt.figure(figsize=figsize)
            proj = ccrs.PlateCarree(central_longitude=cm)
            ax = plt.axes(projection = proj)
            colorbar = True
        else:
            ax = axf
            show = False
            colorbar = False
        if (cm == 180) & (xlocs == None): xlocs = [0,30,60,90,120,150,180,-150,-120,-90,-60,-30]
        if cmap == None:
            cb = cmp.viridis.copy()
            cb.set_under(cb(0))
            cb.set_over(cb(cb.N-1))
        else: cb = cmap
        if log:
            iax = ax.imshow(self.var[var],transform=ax.projection,
                       extent=(self.attr['lo0']-cm,self.attr['lo1']-cm,self.attr['la0'],self.attr['la1']),
                       origin='lower',interpolation='nearest',aspect=aspect,cmap=cb,
                       norm=colors.LogNorm(vmin=vmin,vmax=vmax))
        else:
            iax = ax.imshow(self.var[var],transform=ax.projection,
                       extent=(self.attr['lo0']-cm,self.attr['lo1']-cm,self.attr['la0'],self.attr['la1']),
                       origin='lower',interpolation='nearest',aspect=aspect,cmap=cb,
                       norm=colors.Normalize(vmin=vmin,vmax=vmax))

        if xlims is not None:
            ax.set_xlim(xlims[0],xlims[1])
        ax.set_ylim(ylims[0],ylims[1])
        gl = ax.gridlines(draw_labels=True,xlocs=xlocs)
        ax.coastlines('50m')
        gl.top_labels = False
        gl.right_labels = False
        if not left: gl.left_labels = False
        if not bottom: gl.bottom_labels = False
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        #cbar = plt.colorbar(iax,location='bottom',pad='12%',format='%.2f',extend='max')
        if colorbar:
            cbar = plt.colorbar(iax,location='right',pad=0.02,format='%.2f',shrink=0.77)
        #cbar.set_label('SA OD IMS',fontsize=12)
        if txt == None:
            ax.set_title(var+self.day.strftime('  %d/%m/%Y'))
        else:
            ax.set_title(txt)
        if show: plt.show()
        return iax

class IMS(IMS_pure):
    def __init__(self,day):
        IMS_pure.__init__(self)
        self.day = day

    def read(self,specie='SA',ND='N',close=True):
        # Read the IMS file
        # The grid is described by its boundaries and the mesh size
        # The data should be considered as centered in the mesh
        if specie not in ['SA','SO2']:
            print('unknown specie')
            return
        if ND not in ['N','D']:
            print('unknown ND value')
            return
        # generate filename
        if ND == 'D':
            fname = self.day.strftime('%Y%m%d')+'_ims_metopb_tir_qnrt_'+trl[specie]+'_day_global_g0.5_qc0.nc'
        else:
            fname = self.day.strftime('%Y%m%d')+'_ims_metopb_tir_qnrt_'+trl[specie]+'_night_global_g0.5_qc0.nc'
        fullpath = os.path.join(pathIMS,self.day.strftime('%Y'),fname)
        file = Dataset(fullpath,'r')
        if not close: self.file = file
        if 'lon' not in self.attr:
            lons = file.variables['x_grid'][:]
            #self.attr['lons'] = lons[0]+range(1440)*lons[2]
            self.attr['lo0'] = lons[0]
            self.attr['lo1'] = lons[1]
            self.attr['dlo'] = lons[2]
            nx = self.attr['nx'] =  int((lons[1]-lons[0])/lons[2]+.1)
            lats = file.variables['y_grid'][:]
            #self.attr['lats'] = lats[0]+range(720)*lats[2]
            self.attr['la0'] = lats[0]
            self.attr['la1'] = lats[1]
            self.attr['dla'] = lats[2]
            ny = self.attr['ny'] =  int((lats[1]-lats[0])/lats[2]+.1)
            SA_OD = file.variables['data'][:]
            try:
                SA_OD = np.ma.reshape(SA_OD,(ny,nx))
            except:
                print('read field inconsistent with dimensions')
                return
            QF = file.variables['qa_value'][:]
            SA_OD[QF == 0] = np.ma.masked
            SA_OD[SA_OD.data <= -999] = np.ma.masked
            self.var[specie+'_'+ND] = SA_OD
        if close: file.close()
        return

    def extract(self,latRange=None,lonRange=None):
        eps = 1.e-3
        new = IMS_pure()
        new.day = self.day
        if latRange == None:
           new.attr['la0'] = self.attr['la0']
           new.attr['la1'] = self.attr['la1']
           jy0 = 0
           jy1 = self.attr['ny']
        else:
           if (latRange[0] < self.attr['la0']) |  (latRange[1] > self.attr['la1']):
               print('out of bounds latitudes')
               return None
           else:
               jy0 = int((latRange[0] - new.attr['la0']) / self.attr['dla'] + eps)
               jy1 = int((latRange[1] - new.attr['la1']) / self.attr['dla'] + eps)
               new.attr['la0'] = self.attr['la0'] + jy0*self.attr['dla']
               new.attr['la1'] = self.attr['la0'] + jy1*self.attr['dla']

        if lonRange == None:
            new.attr['lo0'] = self.attr['lo0']
            new.attr['lo1'] = self.attr['lo1']
            ix0 = 0
            ix1 = self.attr['nx']
        else:
            if (lonRange[0] < self.attr['lo0']) |  (lonRange[1] > self.attr['lo1']):
                print('out of bounds longitudes')
                return None
            else:
                ix0 = int((lonRange[0] - new.attr['lo0']) / self.attr['dlo'] + eps)
                ix1 = int((lonRange[1] - new.attr['lo1']) / self.attr['dlo'] + eps)
                new.attr['lo0'] = self.attr['lo0'] + ix0*self.attr['dla']
                new.attr['lo1'] = self.attr['lo0'] + ix1*self.attr['dla']

        for var in self.var:
            new.var[var] = self.var[var][jy0:jy1,ix0:ix1]
            new.attr['ny'] = new.var[var].shape[0]
            new.attr['nx'] = new.var[var].shape[1]
        return new

    def shift(self,lon0):
        # This function rotates the array to change the origin of longitudes
        # This applies only for a full gridfield in longtude
        if self.attr['lo1']-self.attr['lo0'] != 360:
            print('Rotation can only be applied to full longitude grid')
            return None
        eps = 1.e-3
        new = IMS_pure()
        new.day = self.day
        # find pivot longitude that will be the new starting longitude
        ix0 = int((lon0 - self.attr['lo0']) / self.attr['dlo'] + eps)
        new.attr = self.attr.copy()
        new.attr['lo0'] = self.attr['lo0'] + ix0*self.attr['dlo']
        new.attr['lo1'] = self.attr['lo1'] + ix0*self.attr['dlo']
        for var in self.var:
            new.var[var] = np.ma.empty(shape=self.var[var].shape)
            new.var[var][:,0:self.attr['nx']-ix0] = self.var[var][:,ix0:].copy()
            new.var[var][:,self.attr['nx']-ix0:] = self.var[var][:,:ix0].copy()
        return new
