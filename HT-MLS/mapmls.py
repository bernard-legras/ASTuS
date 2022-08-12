# -*- coding: utf-8 t-*-
"""
readMLS: reads MLS data from a granule, retaining only those which satisfy
quality criterion and are optionnaly located within an altitude and a
regional domain.

Created on Sun Dec  4 12:46:54 2016
Adapted to the Hunga Tonga case on 29 May 2022

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr
"""
import h5py
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import socket
#from mpl_toolkits.basemap import Basemap
#from subprocess import call
import os
import pickle, gzip

# rootdir and main directories for aerosol profiles and L1 data
if 'gort' == socket.gethostname():
    rootdir = '/dkol/data/ASTuS/HT-MLS'
    dirMLS = '/dkol/data/MLS'
elif 'satie' in socket.gethostname():
    rootdir = '/data/ASTuS/HT-MLS'
    dirMLS = '/data/MLS'

# Domain and resolution of the bins
# HT: Here we take the full domain
source_range=[[-180.,180.],[-82.5,82.5]]
# Too fine grid, generates fluctuations in the count
#source_binx=55
#source_biny=25
source_binx=72
source_biny=20
#offz=3
#binz=10
xedges=np.arange(source_range[0][0],source_range[0][1]+0.001,
                 (source_range[0][1]-source_range[0][0])/source_binx)
yedges=np.arange(source_range[1][0],source_range[1][1]+0.001,
                 (source_range[1][1]-source_range[1][0])/source_biny)
#presslev=offz+np.arange(binz)
biny_size=(source_range[1][1]-source_range[1][0])/source_biny

# yedges for pole to pole averages, with optimal resolution
yedges_pp=np.linspace(-82.5,82.5,114)
ycent_pp=0.5*(yedges_pp[0:-1]+yedges_pp[1:])

# select levels between 316 and 10 hPa
levs={}
levs['CO'] = range(3,13)
levs['H2O'] = range(6,25)
levs['HNO3'] = range(3,13)
# HT Modified Hunga Tonga to select the stratosphere up to the mesosphere
levs['H2O'] = range(11,43)

# set quality threshold
# from 2016 of 4.2x quality document
quality_threshold={}
quality_threshold['CO'] = 1.5
quality_threshold['H2O'] = 1.45
quality_threshold['HNO3'] = 0.8
quality_threshold['O3'] = 1
quality_threshold['SO2'] = 0.95
conv_threshold={}
conv_threshold['CO'] = 1.03
conv_threshold['H2O'] = 2.
conv_threshold['HNO3'] = 1.03
conv_threshold['O3'] = 1.03
conv_threshold['SO2'] = 1.03

# revised 2018 version
quality_threshold['H2O'] = 0.7
value_threshold = {}
value_threshold['H2O'] = 1.e-6*0.101 # in ppmv


def readMLS(fname,molec,quiet=True,frame=True,bbox=source_range,bypass=False):
    """ Read from the MLS file given by fname within the boundaries defined
    by source_range and get the data for the molecule given in molec (str) *
    which should match fname.
    arguments:
        fname: full path to the file to be read
        molec: retrieved molecule among those defined above
        quiet: no message
        frame: selection of an extraction frame in bbox
        bbox: extraction box((lon1,lat1),(lon2,lat2))
        bypass: no quality test (introduced for Hunga Tonga processing)
    """
    print('bbox',bbox)
    # test that molec is contained in the name.
    # to do: extract molec from fname
    if molec not in fname:
        print('inconsistent molec and fname')
        print(fname+"   "+molec)

    ff=h5py.File(fname,'r')

    geoloc=ff['HDFEOS/SWATHS/'+molec+'-APriori/Geolocation Fields']
    prior=ff['HDFEOS/SWATHS/'+molec+'-APriori/Data Fields']
    poste=ff['HDFEOS/SWATHS/'+molec+'/Data Fields']

    # select profiles the frame if selected
    if frame:
        selec = ((geoloc['Longitude'][:]<=bbox[0][1]) &
                 (geoloc['Longitude'][:]>=bbox[0][0]) &
                 (geoloc['Latitude'][:]<=bbox[1][1]) &
                 (geoloc['Latitude'][:]>=bbox[1][0]))
    else:
        selec = np.full(len(geoloc['Longitude']),True,dtype=np.bool_)

    ss1 = np.sum(selec)
    # screening of the profiles
    # test Status first bit is 0, which means even value
    # HT: bypassing scheme introduced
    selec = selec & ((poste['Status'][:] & 1)==0)
    ss2 = np.sum(selec)
    # test Quality
    if not bypass:
        selec = selec & (poste['Quality'][:]>quality_threshold[molec])
    ss3 = np.sum(selec)
    # test Convergence
    if not bypass:
        selec = selec & (poste['Convergence'][:]<conv_threshold[molec])
    ss4 = np.sum(selec)
    # selection based on the threshold
    if not bypass:
        selec = selec & ~np.any(poste[molec][:,levs[molec]]<value_threshold[molec],axis=1)
    ss5 = np.sum(selec)

    # print selection diags if not quiet run
    if not quiet:
        print(fname)
        print('number of framed pixels     ',str(ss1))
        print('number of bad status        ',str(ss1-ss2))
        print('number of bad quality       ',str(ss2-ss3))
        print('number of bad convergence   ',str(ss3-ss4))
        print('number of bad values        ',str(ss4-ss5))
        print('number of retained profiles ',str(ss5))
        # number of profiles within the selection with precision <0
        print('among which bad precision   ',
              np.sum(np.any(poste[molec+'Precision'][:][selec,:][:,levs[molec]]<0,axis=1)))

    # selection
    data={}
    data['Longitude'] = geoloc['Longitude'][selec]
    data['Latitude'] = geoloc['Latitude'][selec]
    data['Pressure'] = geoloc['Pressure'][:][levs[molec]]
    data['numpix'] = len(data['Longitude'])
    data[molec] = poste[molec][:][selec,:][:,levs[molec]]
    data[molec+'-APriori'] = prior[molec+'-APriori'][:][selec,:][:,levs[molec]]
    data['Convergence'] = poste['Convergence'][selec]
    data['Precision'] = poste[molec+'Precision'][:][selec,:][:,levs[molec]]
    data['L2gpPrecision'] = poste['L2gpPrecision'][:][selec,:][:,levs[molec]]
    data['L2gpValue'] = poste['L2gpValue'][:][selec,:][:,levs[molec]]
    data['Quality'] = poste['Quality'][selec]

    # print diags about the data quality if not quiet
    #if not quiet:
    #    print('Bad quality '+str(np.sum(data['Quality']<quality_threshold[molec])))
    #    print('Bad Convergence '+str(np.sum(data['Convergence']>conv_threshold[molec])))
    #    #print('Bad Precision '+str(np.sum (data['L2gpPrecision'] <=0,axis=0)))
    #if not quiet:
    #    if np.sum(data['L2gpPrecision'] <=0)>0:
            #raise BadPrecision(str(np.sum(data['L2gpPrecision'] <=0,axis=0)))
    #        print('Bad precision ' + str(np.sum(data['L2gpPrecision'] <=0,axis=0)))
        #print('Bad Precision '+str(np.sum(data['L2gpPrecision'] <=0,axis=0)))
    return data

def collectMLS(date,molec,quiet=False,interval='Monthly'):
    """ Attribute MLS measurements to a grid of boxes and generates averages
    of mixing ratios for a given interval (month or day) and a given molecule."""
    year = date.strftime('%Y')
    month = date.strftime('%m')

    nlevs=len(levs[molec])
    res={}
    res['lat']=[]
    res['count'] = np.zeros(shape=[nlevs,source_biny,source_binx])
    cumul = np.zeros(shape=[nlevs,source_biny,source_binx])
    cumulprior = np.zeros(shape=[nlevs,source_biny,source_binx])
    # loop on the files assuming that the directory contains only MLS files
    # two choices for the moment, monthly or daily, leaves weekly selection
    # for later as it
    if interval == 'Monthly':
        listfiles = os.listdir(os.path.join(molec,year,month))
    elif interval == 'Daily':
        listfiles = [file for file in os.listdir(os.path.join(molec,year,month)) if date.strftime('%j') in file]
    else:
        print('wrong interval choice')
        return -1
    if len(listfiles) == 0:
        print('no data for this date',date)
        return -1
    for fname in listfiles:
        ff = os.path.join(molec,year,month,fname)
        data = readMLS(ff,molec,quiet=quiet)
        res['lat']=np.append(res['lat'],data['Latitude'])
        idx = np.digitize(data['Longitude'],xedges)-1
        idy = np.digitize(data['Latitude'], yedges)-1
        numpix = data['numpix']

        for k in range(numpix):
            res['count'][:,idy[k],idx[k]]+=1
            cumul[:,idy[k],idx[k]]+=data[molec][k,:]
            cumulprior[:,idy[k],idx[k]]+=data[molec+'-APriori'][k,:]

    # Calculation of the mean in each box by dividing the accumulation by the
    # count
    # Does this do exactly what we want??????
    mass = res['count']
    mass[mass==0] = 1
    res[molec] = cumul / mass
    res[molec+'-Prior'] = cumulprior / mass
    # Pressure scale for completion
    res['Pressure'] = data['Pressure']
    return res

def collectAll():
    dat7 = datetime(year=2016,month=7,day=1)
    dat8 = datetime(year=2016,month=8,day=1)
    maps = {}
    for molec in ('CO','H2O'):
        maps[molec+'_07'] = collectMLS(dat7,molec,quiet=True)
        maps[molec+'_08'] = collectMLS(dat8,molec,quiet=True)
    return maps

#%%
def dailyLatprofMLS(date,molec,quiet=False,bypass=False):
    """ Calculate a daily latitude profile from pole to pole
    Does work as it is if the whole range of lattitudes are considered"""
    year = date.strftime('%Y')
    month = date.strftime('%m')

    nlevs=len(levs[molec])
    nlat=len(ycent_pp)
    res={}
    res['count'] = np.zeros(nlat)
    cumul = np.zeros(shape=[nlevs,nlat])
    cumulprior = np.zeros(shape=[nlevs,nlat])
    # loop on the files assuming that the directory contains only MLS files
    for fname in os.listdir(os.path.join(dirMLS,molec,year,month)):
        if date.strftime('_%Yd%j.he5') not in fname: continue
        ff = os.path.join(dirMLS,molec,year,month,fname)
        data = readMLS(ff,molec,quiet=quiet,frame=False,bypass=bypass)
        idy = np.digitize(data['Latitude'], yedges_pp) - 1
        for k in range(data['numpix']):
            if (idy[k]>=0) & (idy[k]<len(ycent_pp)):
                res['count'][idy[k]]+=1
                cumul[:,idy[k]]+=data[molec][k,:]
                cumulprior[:,idy[k]]+=data[molec+'-APriori'][k,:]
        break
    res[molec] = cumul / res['count']
    res[molec+'-Prior'] =  cumulprior / res['count']
    # Pressure scale for completion
    res['Pressure'] = data['Pressure']
    res['LogPressure'] = np.log(data['Pressure'])
    lpe = 0.5*(res['LogPressure'][1:]+res['LogPressure'][:-1])
    lpe = np.append(lpe,2*lpe[-1]-lpe[-2])
    res['LogPressureEdges'] = np.insert(lpe,0,2*lpe[0]-lpe[1])
    res['Latitude'] = ycent_pp
    res['LatEdges'] = yedges_pp
    return res

#%%
def latprofMLS(date,molec,quiet=False):
    """ Calculate a monthly latitude profile from pole to pole"""
    year = date.strftime('%Y')
    month = date.strftime('%m')

    nlevs=len(levs[molec])
    nlat=len(yedges_pp)-1
    res={}
    res['count'] = np.zeros(nlat)
    cumul = np.zeros(shape=[nlevs,nlat])
    cumulprior = np.zeros(shape=[nlevs,nlat])
    # loop on the files assuming that the directory contains only MLS files
    for fname in os.listdir(os.path.join(dirMLS,molec,year,month)):
        ff = os.path.join(dirMLS,molec,year,month,fname)
        print(ff)
        data = readMLS(ff,molec,quiet=quiet,frame=False)
        idy = np.digitize(data['Latitude'], yedges_pp)-1
        for k in range(data['numpix']):
            if (idy[k]>=0) & (idy[k]<len(yedges_pp)-1):
                res['count'][idy[k]]+=1
                cumul[:,idy[k]]+=data[molec][k,:]
                cumulprior[:,idy[k]]+=data[molec+'-APriori'][k,:]

    # Calculation of the mean in ecah box by dividing the accumulation by the
    # count
    # Does this do exactly what we want??????

    #res[molec] = np.zeros(shape=[nlevs,nlat])
    #res[molec+'-Prior'] = np.zeros(shape=[nlevs,nlat])
    #for lev in range(nlevs):
    #    res[molec][lev,:] = cumul[lev,:] / res['count']
    #    res[molec+'-Prior'][lev,:] = cumulprior[lev,:] / res['count']
    # Better (it works)
    res[molec] = cumul / res['count']
    res[molec+'-Prior'] =  cumulprior / res['count']
    # Pressure scale for completion
    res['Pressure'] = data['Pressure']
    res['LogPressure'] = np.log(data['Pressure'])
    lpe = 0.5*(res['LogPressure'][1:]+res['LogPressure'][:-1])
    lpe = np.append(lpe,2*lpe[-1]-lpe[-2])
    res['LogPressureEdges'] = np.insert(lpe,0,2*lpe[0]-lpe[1])
    res['Latitude'] = ycent_pp
    res['LatEdges'] = yedges_pp
    return res

def latprofAll():
    dat7 = datetime(year=2016,month=7,day=1)
    dat8 = datetime(year=2016,month=8,day=1)
    dat6 = datetime(year=2016,month=6,day=1)
    prof = {}
    for molec in ('H2O',):
        prof[molec+'_07'] = latprofMLS(dat7,molec,quiet=True)
        prof[molec+'_08'] = latprofMLS(dat8,molec,quiet=True)
        prof[molec+'_06'] = latprofMLS(dat6,molec,quiet=True)
    return prof

# The graphics part is to be re-activated by converting calls to basemap to cartopy
#%% Charts
# def chart(field,txt=None,vmin=0,vmax=None,cmap='jet'):
#     try:
#         n1=plt.get_fignums()[-1]+1
#         fig=plt.figure(plt.get_fignums()[-1]+1,figsize=[13,6])
#     except:
#         n1=1
#         fig=plt.figure(n1+1,figsize=[13,6])
#     m = Basemap(projection='cyl',llcrnrlat=source_range[1][0],
#                 urcrnrlat=source_range[1][1],
#                 llcrnrlon=source_range[0][0],
#                 urcrnrlon=source_range[0][1],resolution='c')
#     m.drawcoastlines(color='w'); m.drawcountries(color='w')
#     meridians = np.arange(source_range[0][0],source_range[0][1],5.)
#     parallels = np.arange(source_range[1][0],source_range[1][1],5.)
#     m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=8)
#     m.drawparallels(parallels,labels=[1,0,0,0],fontsize=8)
#     iax=plt.imshow(field,extent=np.array(source_range).flatten(),
#                origin='lower',aspect=1.,vmin=vmin,vmax=vmax,cmap=cmap)
#     if txt is not None:
#         plt.title(txt)
#     cax = fig.add_axes([0.91, 0.26, 0.03, 0.5])
#     fig.colorbar(iax,cax=cax)
#     #plt.colorbar()

# def chart2(maps,molec,lev):
#     chart(0.5*(maps[molec+'_07'][molec][lev,:,:]+maps[molec+'_08'][molec][lev,:,:]))

if __name__ == '__main__':
    prof = latprofAll()
    pickle_name = 'MeanMLSProf-H2O-2016-07.pkl'
    pickle.dump(prof['H2O_07'],gzip.open(pickle_name,'wb',pickle.HIGHEST_PROTOCOL))
    pickle_name = 'MeanMLSProf-H2O-2016-08.pkl'
    pickle.dump(prof['H2O_08'],gzip.open(pickle_name,'wb',pickle.HIGHEST_PROTOCOL))
    pickle_name = 'MeanMLSProf-H2O-2016-06.pkl'
    pickle.dump(prof['H2O_06'],gzip.open(pickle_name,'wb',pickle.HIGHEST_PROTOCOL))
