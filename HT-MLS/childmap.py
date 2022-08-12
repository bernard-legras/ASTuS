# -*- coding: utf-8 -*-
"""
This script explores the internal structure of a hdf5 file

Created on Wed Nov 23 01:28:14 2016

Copyright or © or Copr.  Bernard Legras (2022)
under CeCILL-C license "http://www.cecill.info".

bernard.legras@lmd.ipsl.fr
"""
import h5py
fname='exemp.he5'

def print_num_child(obj):
    if isinstance(obj,h5py.Group):
        print(obj.name,"Number of Children:",len(obj))
        for ObjName in obj: 
            print_num_child(obj[ObjName])
    else:
        print(obj.name,"Not a group")
        try:
             print("shape ", obj.shape)
        except:
             print('no length')
        

with h5py.File(fname,'r') as f:
    print_num_child(f)     
