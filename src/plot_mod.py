import os
import numpy as np
import subprocess
from datetime import datetime
import json
import shutil
import pickle


from ase.db import connect
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from trainstation import CrossValidationEstimator
from ase import Atoms
from ase.build import bulk
from itertools import chain


from icet.tools import enumerate_structures, ConvexHull
from icet import ClusterSpace
from trainstation import CrossValidationEstimator

from random import random

import pandas as pd
from time import sleep


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import time


from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


from pyiron import Project
from pyiron import ase_to_pyiron
import time
from pyiron.vasp.outcar import Outcar


from pymatgen.io.vasp.inputs import Kpoints
from pymatgen.io import ase

ase_to_pmg = ase.AseAtomsAdaptor.get_structure
pmg_to_ase = ase.AseAtomsAdaptor.get_atoms




def plot_dict(dicto,listo,xn,yn,lista_label=None,wspace=0.2):
    fig, axes = plt.subplots(xn, yn, figsize=(5*yn, 5*xn))
    if lista_label==None:
        lista_label=listo
    for i in range(xn):
        for j in range(yn):
            axes[i,j].plot(np.arange(1,1+len(dicto[listo[i*yn + j]])), dicto[listo[i*yn + j]])
            #axes[i,j].set_title(listo[i*yn + j])
            axes[i,j].set_ylabel(lista_label[i*yn + j])
            axes[i,j].set_xlabel('epoch')

    plt.subplots_adjust(wspace=wspace)
    plt.show()


def r2_array_index(list_arr, index):
    energy_values=np.empty((0),dtype=float)
    forces_values=np.empty((0),dtype=float)
    stress_values=np.empty((0),dtype=float)
    energy_train =np.empty((0),dtype=float)
    forces_train =np.empty((0),dtype=float)
    stress_train =np.empty((0),dtype=float)
    
    for i in index:
        res=potential_1000.get_efs(ase_to_pmg(list_arr[0][i]))
        energy_values=np.append(energy_values,res[0])
        forces_values=np.append(forces_values,res[1])
        stress_values=np.append(stress_values,res[2])
        energy_train =np.append(energy_train, list_arr[1][i])
        forces_train =np.append(forces_train, list_arr[2][i])
        stress_train =np.append(stress_train, list_arr[3][i])
        
    return r2_score(energy_train,energy_values),r2_score(forces_train,forces_values),r2_score(stress_train,stress_values)
    

def plt_all_index(list_arr, index,wspace=0.2):
    energy_values=np.empty((0),dtype=float)
    forces_values=np.empty((0),dtype=float)
    stress_values=np.empty((0),dtype=float)
    energy_train =np.empty((0),dtype=float)
    forces_train =np.empty((0),dtype=float)
    stress_train =np.empty((0),dtype=float)
    
    
    for i in index:
        res=potential_1000.get_efs(ase_to_pmg(list_arr[0][i]))
        energy_values=np.append(energy_values,res[0])
        forces_values=np.append(forces_values,res[1])
        stress_values=np.append(stress_values,res[2])
        energy_train =np.append(energy_train, list_arr[1][i])
        forces_train =np.append(forces_train, list_arr[2][i])
        stress_train =np.append(stress_train, list_arr[3][i])
#         print(res[2])
#         print(stress_values)
#         print(list_arr[3][i])
#         print(stress_train)
#         break
    
    fig, axes = plt.subplots(1, 3, figsize=(5*3, 5*1))
    
    
    axes[0].scatter(energy_train, energy_values)
    axes[0].grid(True)
    axes[0].axis('equal')
    axes[1].scatter(forces_train, forces_values)
    axes[1].grid(True)
    axes[1].axis('equal')
    axes[2].scatter(stress_train, stress_values)
    axes[2].grid(True)
    axes[2].axis('equal')
    plt.show()
    #return r2_score(energy_train,energy_values),r2_score(forces_train,forces_values),r2_score(stress_train,stress_values)
    


def plt_force_three(list_arr, index):

    forces_values=np.empty((0,3),dtype=float)
    forces_train =np.empty((0,3),dtype=float)
    
    
    
    for i in index:
        res=potential_1000.get_efs(ase_to_pmg(list_arr[0][i]))
        forces_values=np.append(forces_values,res[1],axis=0)
        forces_train =np.append(forces_train, list_arr[2][i],axis=0)
        
    fig, axes = plt.subplots(1, 3, figsize=(5*3, 5*1))
    
    for i in range(3):
        axes[i].scatter(forces_train[:,i], forces_values[:,i])
        axes[i].grid(True)
        axes[i].axis('equal')
    plt.show()

    
def plt_stress_nine(list_arr, index):

    stress_values=np.empty((0,3,3),dtype=float)
    stress_train =np.empty((0,3,3),dtype=float)
    
    
    
    for i in index:
        res=potential_1000.get_efs(ase_to_pmg(list_arr[0][i]))
        stress_values=np.append(stress_values,res[2],axis=0)
        stress_train =np.append(stress_train,list_arr[3][i].reshape(1,3,3),axis=0)
    fig, axes = plt.subplots(3, 3, figsize=(5*3, 5*3))
    
    
    
    
    for i in range(3):
        for j in range(3):
            axes[i,j].scatter(stress_train[:,i,j], stress_values[:,i,j])
            axes[i,j].grid(True)
            axes[i,j].axis('equal')
    plt.show()
    print(stress_train)
    
    
def get_lims(xs, ys, panf=0.05):
    
    h_=np.append(xs, ys)
    mi,ma=np.min(h_),np.max(h_)
    pan=panf*(ma-mi)
    return mi-pan,ma+pan

def plt_all_index_pot(pot, list_arr, index,fig_name='',dpi=200,color='blue',wspace=0.2,per_atom=True):

    energy_values=np.empty((0),dtype=float)
    forces_values=np.empty((0),dtype=float)
    stress_values=np.empty((0),dtype=float)

    energy_train =np.empty((0),dtype=float)
    forces_train =np.empty((0),dtype=float)
    stress_train =np.empty((0),dtype=float)

    
    for i in index:
        res=pot.get_efs(ase_to_pmg(list_arr[0][i]))
        if per_atom:
            divi=len(list_arr[0][i])
        else:
            divi=1
        energy_values=np.append(energy_values,res[0]/divi)
        forces_values=np.append(forces_values,res[1])
        stress_values=np.append(stress_values,res[2])
        energy_train =np.append(energy_train, list_arr[1][i]/divi)
        forces_train =np.append(forces_train, list_arr[2][i])
        stress_train =np.append(stress_train, list_arr[3][i])

    fig, axes = plt.subplots(1, 3, figsize=(5*3, 5*1))

    for ni,[i,j] in enumerate(zip([energy_train,forces_train,stress_train],[energy_values,forces_values,stress_values])):
        axes[ni].scatter(i, j,c=color)
        mi,ma=get_lims(i, j)
        axes[ni].set_xlim(mi,ma)
        axes[ni].set_ylim(mi,ma)
        axes[ni].grid(True)
        print(r2_score(i,j))
        axes[ni].annotate(r'$r^2=$'+'{:.4f}'.format(r2_score(i,j)),xy=(0.1,0.9),xycoords='axes fraction')
        axes[ni].annotate(r'$RMSE=$'+'{:.4f}'.format(np.sqrt(mean_squared_error(i,j))),xy=(0.1,0.8),xycoords='axes fraction')
        axes[ni].annotate(r'$MAE=$'+'{:.4f}'.format(mean_absolute_error(i,j)),xy=(0.1,0.7),xycoords='axes fraction')
    if per_atom:
        axes[0].set_xlabel('Energy VASP (eV/atom)')
        axes[0].set_ylabel('Energy m3gnet (eV/atom)')
    else:
        axes[0].set_xlabel('Energy VASP (eV)')
        axes[0].set_ylabel('Energy m3gnet (eV)')
    axes[1].set_xlabel('Forces VASP (eV/A)')
    axes[2].set_xlabel('Stresses VASP (GPa)')
    
    axes[1].set_ylabel('Forces m3gnet (eV/A)')
    axes[2].set_ylabel('Stresses m3gnet (GPa)')
    plt.subplots_adjust(wspace=wspace)
    if fig_name!='':
        plt.savefig(fig_name,dpi=dpi)
    plt.show()
    
        
        
        
def parity_plots(gtcl_list,s=5,color='blue',alpha=1):
    no_values=len(gtcl_list)
    fig, axs = plt.subplots(1, no_values, figsize=(5*no_values, 5))  # Set the figsize parameter to control the figure size
    if no_values==1:
        ii=gtcl_list[0][0]
        jj=gtcl_list[0][1]
        axs.scatter(ii,jj,s=s,color=color,alpha=alpha)

        mi,ma=get_lims(ii, jj)
        axs.set_xlim(mi,ma)
        axs.set_ylim(mi,ma)
        axs.grid(True)
        axs.annotate(r'$r^2=$'+'{:.4f}'.format(r2_score(ii,jj)),xy=(0.1,0.9),xycoords='axes fraction')
        axs.annotate(r'$RMSE=$'+'{:.4f}'.format(np.sqrt(mean_squared_error(ii,jj))),xy=(0.1,0.8),xycoords='axes fraction')
        axs.annotate(r'$MAE=$'+'{:.4f}'.format(mean_absolute_error(ii,jj)),xy=(0.1,0.7),xycoords='axes fraction')
    else:
        for i in range(no_values):

            ii=gtcl_list[i][0]
            jj=gtcl_list[i][1]
            axs[i].scatter(ii,jj,s=s,color=color,alpha=alpha)

            mi,ma=get_lims(ii, jj)
            axs[i].set_xlim(mi,ma)
            axs[i].set_ylim(mi,ma)
            axs[i].grid(True)
            axs[i].annotate(r'$r^2=$'+'{:.4f}'.format(r2_score(ii,jj)),xy=(0.1,0.9),xycoords='axes fraction')
            axs[i].annotate(r'$RMSE=$'+'{:.4f}'.format(np.sqrt(mean_squared_error(ii,jj))),xy=(0.1,0.8),xycoords='axes fraction')
            axs[i].annotate(r'$MAE=$'+'{:.4f}'.format(mean_absolute_error(ii,jj)),xy=(0.1,0.7),xycoords='axes fraction')


        

def res_vs(res_vm,wspace=0.2,per_atom=False,color='b'):
    #print(res_vm)
    #works for 2d arrays where each row is sectioned [train,test]
    no_values= int(len(res_vm[0]))
    fig, axs = plt.subplots(1, no_values, figsize=(5*no_values, 5))  # Set the figsize parameter to control the figure size
    for i in range(no_values):
        if hasattr(np.array(res_vm)[0,i][0], "__len__") :
            ii=np.array(res_vm)[:,i][:,0]
            jj=np.array(res_vm)[:,i][:,1]
            
            
            #print(ii.ravel())
            #print(ii)
            #print(jj[:3])
            ii=np.concatenate(ii,axis=0)
            jj=np.concatenate(jj,axis=0)
            #print(jj[:100])
        else:    
            ii=np.array(res_vm)[:,i][:,0]
            jj=np.array(res_vm)[:,i][:,1]
        #print('ii')
        #print(ii)
        #print('jj')
        #print(jj)
        axs[i].scatter(ii,jj,c=color)

        mi,ma=get_lims(ii, jj)
        axs[i].set_xlim(mi,ma)
        axs[i].set_ylim(mi,ma)
        axs[i].grid(True)
        axs[i].annotate(r'$r^2=$'+'{:.4f}'.format(r2_score(ii,jj)),xy=(0.1,0.9),xycoords='axes fraction')
        axs[i].annotate(r'$RMSE=$'+'{:.4f}'.format(np.sqrt(mean_squared_error(ii,jj))),xy=(0.1,0.8),xycoords='axes fraction')
        axs[i].annotate(r'$MAE=$'+'{:.4f}'.format(mean_absolute_error(ii,jj)),xy=(0.1,0.7),xycoords='axes fraction')
    if per_atom==False:
        if no_values==5:
            axs[0].set_xlabel('Energy (eV)')
            axs[1].set_xlabel(r'Volume ($A^3$)')
            axs[2].set_xlabel(r'Forces ($eV/A$)')
            axs[3].set_xlabel(r'Stresses ($GPa$)')
            axs[4].set_xlabel(r'Atom positions ($A$)')
        if no_values==3:
            axs[0].set_xlabel('Energy VASP (eV)')
            axs[1].set_xlabel(r'Volume VASP (${A^0}^3$)')
            axs[2].set_xlabel(r'Atom positions VASP ($A^0$)')

            axs[0].set_ylabel('Energy m3gnet (eV)')
            axs[1].set_ylabel(r'Volume m3gnet (${A^0}^3$)')
            axs[2].set_ylabel(r'Atom positions m3gnet ($A^0$)')
    else:
        if no_values==5:
            axs[0].set_xlabel('Energy (eV/atom)')
            axs[1].set_xlabel(r'Volume ($A^3$/atom)')
            axs[2].set_xlabel(r'Forces ($eV/A$)')
            axs[3].set_xlabel(r'Stresses ($GPa$)')
            axs[4].set_xlabel(r'Atom positions ($A$)')
        if no_values==3:
            axs[0].set_xlabel('Energy VASP (eV/atom)')
            axs[1].set_xlabel(r'Volume VASP (${A^0}^3$/atom)')
            axs[2].set_xlabel(r'Atom positions VASP ($A^0$)')

            axs[0].set_ylabel('Energy m3gnet (eV/atom)')
            axs[1].set_ylabel(r'Volume m3gnet (${A^0}^3$/atom)')
            axs[2].set_ylabel(r'Atom positions m3gnet ($A^0$)')
    plt.subplots_adjust(wspace=wspace)
    plt.show()    
    
    
    
