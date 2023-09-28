import os
import time
import numpy as np
import json
import shutil
import pickle

from ase import Atoms
from ase.build import bulk
from itertools import chain


from icet.tools import enumerate_structures, ConvexHull
from icet import ClusterSpace, StructureContainer, ClusterExpansion
from trainstation import CrossValidationEstimator

from random import random

import pandas as pd
from time import sleep


import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors


from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


from pyiron import Project
from pyiron import ase_to_pyiron
from pyiron.vasp.outcar import Outcar

from pymatgen.io.vasp.inputs import Kpoints


from pymatgen.io import ase as pgase

ase_to_pmg = pgase.AseAtomsAdaptor.get_structure
pmg_to_ase = pgase.AseAtomsAdaptor.get_atoms



from ase.spacegroup import get_spacegroup

from pymatgen.analysis.magnetism import CollinearMagneticStructureAnalyzer


import copy


from elastool import stress_tens_to_voigt, stress_to_dict, stress_accom, calc_elastic_constants, group_ordering, expanded_strain_ase_list,flatten_list

np.random.seed(seed=42)


class dft_job_array():
    def __init__(self):
        self.RUN_STARTED = False
        
        self.CONV = -1.60218E2

    def from_arrays_starter(self, szs, prs, ics, pyiron_proj_name, **kwargs):
        self.szs_ARR = szs
        self.prs_ARR = prs
        self.ics_ARR = ics
        self.DFT_dir_PA = pyiron_proj_name
        if 'own_incars' in kwargs:
            self.own_incars = kwargs['own_incars']
            self.OWN_INCARS = True
        else:
            self.OWN_INCARS = False
        
        self.RUN_indices = np.zeros(self.szs_ARR.shape[1], dtype=int)
        
        self.statuses = np.empty(self.szs_ARR.shape, dtype=object)
        
        self.open_cores = ['open']*self.szs_ARR.shape[1]
        self.cross_ref = np.empty(self.szs_ARR.shape[1], dtype=object)
        
        print(self.open_cores)
        self.indeces_RUN = np.arange(self.szs_ARR.shape[1])
        print(self.indeces_RUN)
        
        self.NUM_cores = self.szs_ARR.shape[1]
        
        print(self.prs_ARR)
        self.job_crrnt = self.NUM_cores
        
        self.dft_JOBS = np.empty(self.szs_ARR.shape, dtype=object)
        
        print(self.dft_JOBS)
        
        self.running_PLOT = np.ones((self.NUM_cores, 2))*-0.5
        
        self.FINI_grid = []
        self.ABOR_grid = []
        self.NOTC_grid = []
        self.WARN_grid = []
        self.FINI_color = []
        self.ABOR_color = []
        self.NOTC_color = []
        self.WARN_color = []

    def start_specific(self, job_ind, DEL_EXJ=True, custom_relaxer=None):
        jobs = [0]*self.szs_ARR.shape[1]
        proj_pyiron = Project(path=self.DFT_dir_PA)
        
        for i in range(self.NUM_cores):
            if self.open_cores[i] == 'open':
                
                self.cross_ref[i] = job_ind
                
                if job_ind[0] >= self.szs_ARR.shape[0]:
                    print('OUT of scope')
                    self.open_cores[i] = 'open'
                    return
                if self.szs_ARR[job_ind[0], job_ind[1]] <= 1E-12:
                    print('OUT of scope')
                    self.open_cores[i] = 'open'
                    return

                print(job_ind, 'relax_{:000006}_VASP'.format(
                    self.prs_ARR[job_ind[0]][job_ind[1]]))

                try:
                    jobs[i] = proj_pyiron.create_job(
                        job_type=proj_pyiron.job_type.Vasp,
                        job_name='relax_{:000006}_VASP'.format(
                            self.prs_ARR[job_ind[0]][job_ind[1]]),
                        delete_existing_job=DEL_EXJ
                        )

                except Exception as e:
                    print(e)
                    jobs[i] = proj_pyiron.load('relax_{:000006}_VASP'.format(
                        self.prs_ARR[job_ind[0]][job_ind[1]]))

                print(jobs[i].status)

                if jobs[i].status == 'initialized':
                    jobs[i].structure = ase_to_pyiron(
                        self.ics_ARR[job_ind[0]][job_ind[1]].copy())
                    if custom_relaxer is None:
                        self.relax_isif3(jobs[i])
                    else:
                        custom_relaxer(jobs[i])

                    sleep(3+3*np.random.random())

                    [self.ABOR_grid, self.ABOR_color] = delete_entry(
                        job_ind[::-1], self.ABOR_grid, self.ABOR_color)
                    self.running_PLOT[i] = np.array([job_ind[1], job_ind[0]])
                    
                    print(self.running_PLOT)

                    self.open_cores[i] = 'in use'
                    self.statuses[job_ind[0], job_ind[1]] = 'in rerun'
                   
                break
    
    def plot_progress(self):
        print(self.szs_ARR)
        cmap = colors.ListedColormap(['red','blue','green','yellow','purple','teal'])
        if 0 in np.array(self.szs_ARR):
            cmap = colors.ListedColormap(['gray','red','blue','green','yellow','purple','teal'])
        plt.figure(figsize=(self.szs_ARR.shape[1]/3,self.szs_ARR.shape[0]/3))
        #plt.figure()
        plt.pcolor(self.szs_ARR,cmap=cmap,edgecolors='k', linewidths=3)
        
        
        if self.RUN_STARTED:
            plt.scatter(self.running_PLOT[:,0]+0.5, self.running_PLOT[:,1]+0.5, marker='o',
                        color='white',edgecolor='red', s=50)
            if len(self.FINI_grid) > 0:
                dones=np.array(self.FINI_grid)
                plt.scatter(dones[:,0]+0.5,dones[:,1]+0.5, marker='o', 
                            color=self.FINI_color, edgecolor='white', s=70)
            if len(self.WARN_grid) > 0:
                warns=np.array(self.WARN_grid)
                plt.scatter(warns[:, 0]+0.5, warns[:, 1]+0.5, marker='^', 
                            color=self.WARN_color, edgecolor='white',s=70)
            if len(self.NOTC_grid) > 0:
                notcs=np.array(self.NOTC_grid)
                plt.scatter(notcs[:,0]+0.5,notcs[:,1]+0.5, marker='v', color=self.NOTC_color, edgecolor='white',s=70)
            if len(self.ABOR_grid) > 0:
                abors=np.array(self.ABOR_grid)
                plt.scatter(abors[:,0]+0.5,abors[:,1]+0.5, marker='s', color=self.ABOR_color, edgecolor='white',s=70)

        plt.show()
        
        
    def prepare_data(self):
        all_ener,all_forces,all_traj,all_stss=[],[],[],[]
        all_stss_vasp=[]
        
        self.index_data=    np.empty(self.szs_ARR.shape,dtype=object)
        self.no_ionic_steps=0
        for i,ii in enumerate(self.dft_JOBS):
            for j,jj in enumerate(ii):
                if (jj is not None) and (jj != "error"):
                    
                    all_ener.append(jj['energies'])
                    all_forces.append(jj['forces'])
                    all_traj.append(jj['trajectories'])
                    all_stss.append(jj['stresses'])
                    
                    all_stss_vasp.append(jj['stresses']*self.CONV)
                    
                    self.index_data[i][j] = \
                        np.arange(self.no_ionic_steps,
                                  self.no_ionic_steps + 
                                  len(jj['trajectories']), 1)
                    
                    self.no_ionic_steps = self.no_ionic_steps + len(
                        jj['trajectories'])

        self.ALL_ENER = all_ener
        self.ALL_FORC = all_forces
        self.ALL_TRAJ = all_traj
        self.ALL_STSS = all_stss
        
        self.ALL_STSS_VASP = all_stss_vasp
        
        return self.ALL_TRAJ, self.ALL_ENER, self.ALL_FORC, self.ALL_STSS_VASP
    
    def retrieve_data(self):
        self.prepare_data()
        a_e = np.concatenate(self.ALL_ENER)
        a_f = np.array(list(chain.from_iterable(self.ALL_FORC)), dtype=object)
        a_s = np.array(list(chain.from_iterable(self.ALL_STSS_VASP)), 
                       dtype=object)
        a_t = np.array(list(chain.from_iterable(self.ALL_TRAJ)), dtype=object)
        return a_t, a_e, a_f, a_s
    
    def break_data(self, frac_parameter, random_bool=True, 
                   by_trajectories=True):
        tr_i = np.empty((0,), dtype=int)
        te_i = np.empty((0,), dtype=int)
        
        if by_trajectories:
            for i, ii in enumerate(self.dft_JOBS):
                for j, jj in enumerate(ii):
                    if (jj is not None) and (jj != "error"):
                        if random_bool:
                            train_index, test_index, _, _ = train_test_split(
                                self.index_data[i][j], self.index_data[i][j], 
                                train_size=frac_parameter, 
                                random_state=42)
                        else:
                            train_index, test_index, _, _ = train_test_split(
                                self.index_data[i][j], self.index_data[i][j],
                                train_size=frac_parameter,
                                shuffle=False)

                        tr_i = np.append(tr_i, train_index.astype(int))
                        te_i = np.append(te_i, test_index.astype(int))
            return tr_i, te_i
        else:
            tr_i, te_i, _, _ = train_test_split(
                np.arange(0, self.no_ionic_steps, 1),
                np.arange(0, self.no_ionic_steps, 1), 
                train_size=frac_parameter, random_state=42)
            return tr_i, te_i

    def print_cut(self):
        print('############################################################\n')

    def pre_steps(self, **kwargs):
        if 'strain_test' in kwargs.keys():
            self.strains_test = kwargs['strain_test']
            
        if 'elastic' in kwargs.keys():
            self.elastic = kwargs['elastic']
        else:
            self.elastic = False
            
        if 'input_dict' in kwargs.keys():
            self.input_dict=kwargs['input_dict']
            
        if 'steps_blank' in  kwargs.keys():
            self.steps_blank = kwargs['steps_blank']
        else:
            self.steps_blank = [
                [
                    {'strain':None,
                        'job_specs': {
                            '-INCAR-NCORE':16,
                            '-INCAR-LORBIT':1,
                            '-INCAR-ISMEAR':1,
                            '-INCAR-NSW':200,
                            '-INCAR-PREC':'Accurate',
                            '-INCAR-IBRION':2,
                            '-INCAR-ISMEAR':1,
                            '-INCAR-SIGMA':0.2,
                            '-INCAR-ENCUT':400,
                            '-INCAR-EDIFF':1E-5,
                            '-INCAR-EDIFFG':1E-4,
                            'KPPA':3000,
                            '-INCAR-ISIF':3,
                        #'-INCAR-ISPIN':2,
                        }
                            }
                ],
                    [
                        {'strain':None,#this one gets modified
                        'job_specs': {
                            '-INCAR-NCORE':16,
                            '-INCAR-LORBIT':1,
                            '-INCAR-ISMEAR':1,
                            '-INCAR-NSW':200,
                            '-INCAR-PREC':'Accurate',
                            '-INCAR-IBRION':2,
                            '-INCAR-ISMEAR':1,
                            '-INCAR-SIGMA':0.2,
                            '-INCAR-ENCUT':400,
                            '-INCAR-EDIFF':1E-5,
                            '-INCAR-EDIFFG':1E-4,
                            'KPPA':3000,
                            '-INCAR-ISIF':2,
                        #'-INCAR-ISPIN':2,
                        }
                            },

                    ],
                [
                    {'strain':None,
                    'job_specs': {
                        '-INCAR-NCORE':16,
                        '-INCAR-LORBIT':1,
                            '-INCAR-ISMEAR':1,
                            '-INCAR-NSW':0,
                            '-INCAR-PREC':'Accurate',
                            '-INCAR-IBRION':-1,
                            '-INCAR-ISMEAR':1,
                            '-INCAR-SIGMA':0.2,
                            '-INCAR-ENCUT':500,
                            '-INCAR-EDIFF':1E-5,
                            
                            '-INCAR-ISIF':3,
                            'KPPA':5000,
                        '-INCAR-ISPIN':2,
                    }
                        },

                    
                ]
        ]
        if self.elastic:
            self.steps_sizes=[1, -1, -1]
        else:
            self.steps_sizes=[len(i) for i in self.steps_blank]
        
        steps_blank=np.array([[]]*len(self.steps_sizes),dtype=object)
        
        '''!!!!! This looks too ugly, is there another way? There must be .copy() works wonder, bug was somewhere else'''
        self.dft_steps_RES=np.empty(self.szs_ARR.shape,dtype=object)#self.dft_JOBS.copy()
        for i in range(len(self.dft_steps_RES)):
            for j in range(len(self.dft_steps_RES[i])):
                self.dft_steps_RES[i,j]=[[] for _ in range(len(self.steps_sizes))]
        self.dft_steps_RES=np.array(self.dft_steps_RES)

        
        self.steps_statuses=np.empty(self.szs_ARR.shape,dtype=object)#self.dft_JOBS.copy()        
        for i in range(len(self.steps_statuses)):
            for j in range(len(self.steps_statuses[i])):
                self.steps_statuses[i,j]=[[] for _ in range(len(self.steps_sizes))]
        self.steps_statuses=np.array(self.steps_statuses)
        
        self.dict_steps=np.empty(self.szs_ARR.shape,dtype=object)
        for i in range(len(self.dict_steps)):
            for j in range(len(self.dict_steps[i])):
                self.dict_steps[i,j]=[[] for _ in range(len(self.steps_sizes))]
        #self.dict_steps=np.array(self.dict_steps)
        #BM
        ## err_arr is filled each time we hit an err abort or we can fill it up each time we set a pre-run or run
        self.err_arr=self.get_errs()#[np.empty((0,3),dtype=int),np.empty((0,2),dtype=int),[],[]]
        
        

        
        if self.elastic:
            if 'lst_strains' in  kwargs.keys():
                self.lst_strains = kwargs['lst_strains']
            else:
                self.lst_strains           =[-0.05,0.05]
            self.crystal_system       =np.empty(self.szs_ARR.shape,dtype=object)
            self.elastic_size         =np.empty(self.szs_ARR.shape,dtype=object)
            self.strains_per_mag      =np.empty(self.szs_ARR.shape,dtype=object)
            self.strains              =np.empty(self.szs_ARR.shape,dtype=object)
            
        
        self.progress =  np.tile(np.array([0,0]), (*self.dft_steps_RES.shape,1))

    def run_steps(self):
        
        
        proj_pyiron = Project(path=self.DFT_dir_PA)
        
        jobs=[0]*self.szs_ARR.shape[1]
        self.NUM_cores = self.szs_ARR.shape[1]
        self.RUN_STARTED = True
        
        
        
        
        for i in range(self.NUM_cores):
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
            print('core : ',i)
            print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n')
            job_ind=[self.indeces_RUN[i]//self.NUM_cores,self.indeces_RUN[i]%self.NUM_cores]
            self.cross_ref[i]= job_ind
            print('self.indices_RUN',self.indeces_RUN)
            print(self.indeces_RUN[i],self.NUM_cores)

            print('job_ind',job_ind)
            if job_ind[0]>=self.szs_ARR.shape[0]:
                print('THIS core has reached the end of its use')
                self.open_cores[i]='open'
                continue
            if self.szs_ARR[job_ind[0],job_ind[1]]<=1E-12:
                print('THIS core has reached the end of its use')
                self.open_cores[i]='open'
                continue
                
            print(job_ind,':',self.prs_ARR[job_ind[0]][job_ind[1]],'job_{:000006}'.format(self.prs_ARR[job_ind[0]][job_ind[1]]))
            
            LAST=False
            
            ### modify flags 
            if len(self.ics_ARR[job_ind[0]][job_ind[1]]) == 1:
                steps_b = copy.deepcopy(self.steps_blank)
                steps_b[0][0]['job_specs']['queue_number']=1
                if self.elastic:
                    steps_b[1][0]['job_specs']['queue_number']=1
                    steps_b[2][0]['job_specs']['queue_number']=1
                
                
                
                
                
            else:
                steps_b = self.steps_blank
            
            obj_magmoms=CollinearMagneticStructureAnalyzer(ase_to_pmg(
                    self.ics_ARR[job_ind[0]][job_ind[1]]),'replace_all')
            
            ####################
            #steps_b[0][0]['job_specs']['-INCAR-MAGMOM']=obj_magmoms.magmoms
            for mag_i in range(len(steps_b)):
                if '-INCAR-ISPIN' in steps_b[mag_i][0]['job_specs']:
                    if steps_b[mag_i][0]['job_specs']['-INCAR-ISPIN']==1 or steps_b[mag_i][0]['job_specs']['-INCAR-ISPIN']==2:
                        steps_b[mag_i][0]['job_specs']['-INCAR-MAGMOM']=obj_magmoms.magmoms
            ####################
            
            
            ###############
            #steps_b[0][0]['job_specs']['-INCAR-MAGMOM']=np.array([0.5]*len(self.ics_ARR[job_ind[0]][job_ind[1]]))
            #steps_b[1][0]['job_specs']['-INCAR-MAGMOM']=np.array([0.5]*len(self.ics_ARR[job_ind[0]][job_ind[1]]))
            #steps_b[2][0]['job_specs']['-INCAR-MAGMOM']=np.array([0.5]*len(self.ics_ARR[job_ind[0]][job_ind[1]]))
            ###############
            
            
            for n_outer in range(self.progress[job_ind[0],job_ind[1]][0], len(self.steps_sizes)):
                
                
                
                
                
                
                if n_outer > self.progress[job_ind[0],job_ind[1]][0]:
                    continue
                    
                    
                    
                    
                #### OUTER 0: set up all for no elastic, set up first one for elastic
                if n_outer == 0:
                    ase_outer = self.ics_ARR[job_ind[0]][job_ind[1]].copy()
                    same_root = True
                    
                    
                    
                    
                    if len(self.dict_steps[job_ind[0],job_ind[1]][0])==0:
                        '''check if this hasn't run before, for errors more likely'''
                        if self.elastic == True:
                            if self.OWN_INCARS:
                                self.dict_steps[job_ind[0],job_ind[1]][0].append(copy.deepcopy(steps_b[0][0]))
                                
                                self.dict_steps[job_ind[0],job_ind[1]][0][0]['job_specs']=merge_with_priority(
                                    self.own_incars[job_ind[0],job_ind[1]],self.dict_steps[
                                        job_ind[0],job_ind[1]][0][0]['job_specs'])
                            else:    
                                self.dict_steps[job_ind[0],job_ind[1]][0].append(copy.deepcopy(steps_b[0][0]))
                        
                        else:
                            self.dict_steps[job_ind[0],job_ind[1]]=copy.deepcopy(steps_b)
                
                #### OUTER 1: only elastic info has to be added here
                elif n_outer == 1:

                    #### run in all above 0 ####
                    same_root = (len(self.dict_steps[job_ind[0],job_ind[1]][n_outer-1]) == 1)
                    if same_root:
                        ase_outer = self.dft_steps_RES[job_ind[0],job_ind[1]][n_outer-1][0]['trajectories'][-1].copy()
                    ############################
                    
                    if self.elastic == True:
                        if len(self.dict_steps[job_ind[0],job_ind[1]][1])==0:
                            '''check if this hasn't run before , has to be run after ase_outer for consistency'''
                            self.strains_per_mag[job_ind[0],job_ind[1]]=expanded_strain_ase_list(
                                ase_outer, self.input_dict,self.lst_strains)
                            
                            self.strains[job_ind[0],job_ind[1]]=flatten_list(self.strains_per_mag[job_ind[0],job_ind[1]])
                            
                            
                            
                            
                            self.elastic_size[job_ind[0],job_ind[1]]=len(self.strains[job_ind[0],job_ind[1]])
                            self.crystal_system[job_ind[0],job_ind[1]]=get_spacegroup(ase_outer)


                            
                            if self.OWN_INCARS:
                                for ei in range(self.elastic_size[job_ind[0],job_ind[1]]):
                                    self.dict_steps[job_ind[0],job_ind[1]][1].append(copy.deepcopy(steps_b[1][0]))
                                    self.dict_steps[job_ind[0],job_ind[1]][1][ei]['strain']=self.strains[
                                        job_ind[0],job_ind[1]][ei]
                                    
                                    self.dict_steps[job_ind[0],job_ind[1]][1][ei]['job_specs']=merge_with_priority(
                                    self.own_incars[job_ind[0],job_ind[1]],self.dict_steps[job_ind[0],job_ind[1]][1][ei]['job_specs'])
                                    
                                    
                                    
                                    self.dict_steps[job_ind[0],job_ind[1]][2].append(copy.deepcopy(steps_b[2][0]))
                                    self.dict_steps[job_ind[0],job_ind[1]][2][ei]['job_specs']=merge_with_priority(
                                    self.own_incars[job_ind[0],job_ind[1]],self.dict_steps[job_ind[0],job_ind[1]][2][ei]['job_specs'])
                                                                                                                  
                                    
                            else:
                                for ei in range(self.elastic_size[job_ind[0],job_ind[1]]):
                                    self.dict_steps[job_ind[0],job_ind[1]][1].append(copy.deepcopy(steps_b[1][0]))
                                    self.dict_steps[job_ind[0],job_ind[1]][1][ei]['strain']=self.strains[
                                        job_ind[0],job_ind[1]][ei]
                                    
                                    self.dict_steps[job_ind[0],job_ind[1]][2].append(copy.deepcopy(steps_b[2][0]))
                else:
                    same_root = (len(self.dict_steps[job_ind[0],job_ind[1]][n_outer-1]) == 1)
                    
                    if same_root:
                        ase_outer = self.dft_steps_RES[job_ind[0],job_ind[1]][n_outer-1][0]['trajectories'][-1].copy()
                


                #print('len(self.dict_steps[job_ind[0],job_ind[1]][n_outer])',len(self.dict_steps[job_ind[0],job_ind[1]][n_outer]))
                for n_inner in range(self.progress[job_ind[0],job_ind[1]][1], 
                                     len(self.dict_steps[job_ind[0],job_ind[1]][n_outer])):
                    #print('n_inner',n_inner)
                    #print('self.progress[job_ind[0],job_ind[1]]',self.progress[job_ind[0],job_ind[1]])
                    if n_inner > self.progress[job_ind[0],job_ind[1]][1]:
                        continue
                        
                    print('job_ind: {:000006}'.format(self.prs_ARR[job_ind[0]][job_ind[1]]),
                          'n_outer:',n_outer,' n_inner:',n_inner,'...',
                         'vasp_{:000006}'.format(
                                                  self.prs_ARR[job_ind[0]][job_ind[1]]
                                              )+'_{:02}'.format(
                                                  n_outer
                                              )+'_{:02}'.format(
                                                  n_inner
                                              )
                         )
                    
                    #print('HERE#####',[len(self.steps_sizes)-1,len(self.dict_steps[-1])-1])
                    if [n_outer,n_inner]==[len(self.dict_steps[job_ind[0],job_ind[1]])-1,
                                           len(self.dict_steps[job_ind[0],job_ind[1]][-1])-1]:
                        print('LAST')
                        LAST=True
                    
                    for key, value in self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner].items():
                        print(key,' : ', value)
                    
                    if same_root:
                        ase_inner = self.apply_strain(ase_outer.copy(),
                                                      self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['strain'])
                    
                    else:
                        prev_inner = self.dft_steps_RES[job_ind[0],job_ind[1]][n_outer-1][n_inner]['trajectories'][-1].copy()
                        ase_inner = self.apply_strain(prev_inner.copy(),
                                                      self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['strain'])
                    
                    try:
                        jobs[i]=proj_pyiron.create_job(job_type=proj_pyiron.job_type.Vasp,
                                              job_name='vasp_{:000006}'.format(
                                                  self.prs_ARR[job_ind[0]][job_ind[1]]
                                              )+'_{:02}'.format(
                                                  n_outer
                                              )+'_{:02}'.format(
                                                  n_inner
                                              ),
                                          delete_existing_job=False)
                    except:
                        jobs[i]=proj_pyiron.load('vasp_{:000006}'.format(
                                                  self.prs_ARR[job_ind[0]][job_ind[1]]
                                              )+'_{:02}'.format(
                                                  n_outer
                                              )+'_{:02}'.format(
                                                  n_inner
                                              ))
                        
                    print('status : ',jobs[i].status)

                    if jobs[i].status=='initialized':
                        
                        jobs[i].structure = ase_to_pyiron(ase_inner.copy())
                        ### job function change
                        
                        print(i, '-->',job_ind,':',self.prs_ARR[job_ind[0]][job_ind[1]],' [',n_outer,',',n_inner,'] ','PASS TO RUN')
                        self.relax_blank(jobs[i],self.dict_steps[job_ind[0]][job_ind[1]][n_outer][n_inner]['job_specs'])
                        
                        #sleep(3+3*np.random.random())
                        self.running_PLOT[i]=np.array([job_ind[1],job_ind[0]])
                        self.open_cores[i]='in use'
                        continue
                        
                    if jobs[i].status=='running' or jobs[i].status=='collect':
                        self.running_PLOT[i]=np.array([job_ind[1],job_ind[0]])
                        self.open_cores[i]='in use'
                        continue
                    
                    if jobs[i].status=='finished': 
                        self.running_PLOT[i]=np.array([-0.5,-0.5])
                        try:
                            self.dft_steps_RES[job_ind[0],job_ind[1]][n_outer].append(self.get_dft_results_pyiron(jobs[i]))
                            
                            
                            if [job_ind[1],job_ind[0]] not in self.FINI_grid:
                                self.FINI_grid.append([job_ind[1],job_ind[0]])
                                self.FINI_color.append('green')

                            
                            print('pass FINI')
                            self.print_cut()
                            self.steps_statuses[job_ind[0],job_ind[1]][n_outer].append('pass FINI')
                            
                            

                        except Exception as e:
                            print(e)
                            self.dft_steps_RES[job_ind[0],job_ind[1]][n_outer].append('error')

                            if [job_ind[1],job_ind[0]] not in self.FINI_grid:
                                self.FINI_grid.append([job_ind[1],job_ind[0]])
                                self.FINI_color.append('red')

                            print('error FINI: ', e)
                            self.print_cut()
                            self.steps_statuses[job_ind[0],job_ind[1]][n_outer].append('error FINI')
                        
                        if self.progress[job_ind[0],job_ind[1]][1]==len(self.dict_steps[job_ind[0],job_ind[1]][n_outer])-1:
                            self.progress[job_ind[0],job_ind[1]][0] = self.progress[job_ind[0],job_ind[1]][0] + 1
                            self.progress[job_ind[0],job_ind[1]][1] = 0
                        else:
                            self.progress[job_ind[0],job_ind[1]][1] = self.progress[job_ind[0],job_ind[1]][1] + 1
                        if LAST:
                            self.open_cores[i]='open'
                            self.indeces_RUN[i] = self.job_crrnt
                            self.job_crrnt = self.job_crrnt + 1
                        continue
                        
                    if jobs[i].status=='aborted': 
                        self.running_PLOT[i]=np.array([-0.5,-0.5])
                        try:
                            self.dft_steps_RES[job_ind[0],job_ind[1]][n_outer].append(self.get_dft_results_pyiron(jobs[i]))
                            if [job_ind[1],job_ind[0]] not in self.ABOR_grid:
                                self.ABOR_grid.append([job_ind[1],job_ind[0]])
                                self.ABOR_color.append('green')

                            print('pass ABOR')
                            self.print_cut()
                            self.steps_statuses[job_ind[0],job_ind[1]][n_outer].append('pass ABOR')

                        except Exception as e:
                            # self.dft_steps_RES[job_ind[0],job_ind[1]][n_outer].append('error')

                            # if [job_ind[1],job_ind[0]] not in self.ABOR_grid:
                            #     self.ABOR_grid.append([job_ind[1],job_ind[0]])
                            #     self.ABOR_color.append('red')

                            print('error ABOR: ', e)
                            self.print_cut()
                            #self.steps_statuses[job_ind[0],job_ind[1]][n_outer].append('error ABOR')
                            
                            
                            ######
                            #self.err_arr=[np.empty((0,3),dtype=int),[],[]]
                            
                            
                            
                            err_count = np.count_nonzero(np.all(self.err_arr[0] == np.array(
                                [self.prs_ARR[job_ind[0]][job_ind[1]],n_outer,n_inner]), axis=1))
                            
                            
                                                   
                                                   
                            self.err_arr[0] = np.vstack((self.err_arr[0],
                                                        np.array(
                                [self.prs_ARR[job_ind[0]][job_ind[1]],n_outer,n_inner]) ))
                               
                            self.err_arr[1] = np.vstack((self.err_arr[1],job_ind))
                               
                            self.err_arr[2].append(jobs[i].name+'_err_{:02}'.format(err_count))
                               
                            self.err_arr[3].append(copy.deepcopy(self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]))
                            
                            if err_count >= 5:
                                print('giving up on this run')
                                self.open_cores[i]='open'
                                self.indeces_RUN[i] = self.job_crrnt
                                self.job_crrnt = self.job_crrnt + 1
                                continue
                            
                            #BM
                            jobs[i].copy_to(new_job_name=jobs[i].name+'_err_{:02}'.format(err_count),copy_files=True,
                                            new_database_entry=False)
                            
                            jobs[i]=proj_pyiron.create_job(job_type=proj_pyiron.job_type.Vasp,
                                              job_name='vasp_{:000006}'.format(
                                                  self.prs_ARR[job_ind[0]][job_ind[1]]
                                              )+'_{:02}'.format(
                                                  n_outer
                                              )+'_{:02}'.format(
                                                  n_inner
                                              ),
                                          delete_existing_job=True)
                            
                            
                            #### update
                            self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['job_specs']['KPPA']=\
                            self.steps_blank[n_outer][0]['job_specs']['KPPA']+500*(err_count+1)
                            self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['job_specs']['-INCAR-ENCUT']=\
                            self.steps_blank[n_outer][0]['job_specs']['-INCAR-ENCUT']+20*(err_count+1)

                            #### update
                            #self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['job_specs']['k_mesh']=\
                            #self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['job_specs']['k_mesh']+1E-1
                            
                            #### update
                            #self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['job_specs']['-INCAR-ISYM']=\
                            #self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['job_specs']['-INCAR-ISYM']*10
                            
                            #######
                            continue    ####Here when we get an error we continue to next n_inner who should not run again.... 
                                        ########then why?
                            
                            

                        #self.open_cores[i]='open'
                        if self.progress[job_ind[0],job_ind[1]][1]==len(self.dict_steps[job_ind[0],job_ind[1]][n_outer])-1:
                            self.progress[job_ind[0],job_ind[1]][0] = self.progress[job_ind[0],job_ind[1]][0] + 1
                            self.progress[job_ind[0],job_ind[1]][1] = 0
                        else:
                            self.progress[job_ind[0],job_ind[1]][1] = self.progress[job_ind[0],job_ind[1]][1] + 1
                        #self.indeces_RUN[i] = self.job_crrnt
                        #self.job_crrnt = self.job_crrnt + 1
                        if LAST:
                            self.open_cores[i]='open'
                            self.indeces_RUN[i] = self.job_crrnt
                            self.job_crrnt = self.job_crrnt + 1
                        continue
                    if jobs[i].status=='not_converged': 
                        self.running_PLOT[i]=np.array([-0.5,-0.5])
                        try:
                            self.dft_steps_RES[job_ind[0],job_ind[1]][n_outer].append(self.get_dft_results_pyiron(jobs[i]))
                            if [job_ind[1],job_ind[0]] not in self.NOTC_grid:
                                self.NOTC_grid.append([job_ind[1],job_ind[0]])
                                self.NOTC_color.append('green')

                            print('pass NOTC')
                            self.print_cut()
                            self.steps_statuses[job_ind[0],job_ind[1]][n_outer].append('pass NOTC')

                        except Exception as e:
                            print(e)
                            self.dft_steps_RES[job_ind[0],job_ind[1]][n_outer].append('error')

                            if [job_ind[1],job_ind[0]] not in self.NOTC_grid:
                                self.NOTC_grid.append([job_ind[1],job_ind[0]])
                                self.NOTC_color.append('red')

                            print('error NOTC: ', e)
                            self.print_cut()
                            self.step_statuses[job_ind[0],job_ind[1]][n_outer].append('error NOTC')

                        #self.open_cores[i]='open'
                        if self.progress[job_ind[0],job_ind[1]][1]==len(self.dict_steps[job_ind[0],job_ind[1]][n_outer])-1:
                            self.progress[job_ind[0],job_ind[1]][0] = self.progress[job_ind[0],job_ind[1]][0] + 1
                            self.progress[job_ind[0],job_ind[1]][1] = 0
                        else:
                            self.progress[job_ind[0],job_ind[1]][1] = self.progress[job_ind[0],job_ind[1]][1] + 1
                        #self.indeces_RUN[i] = self.job_crrnt
                        #self.job_crrnt = self.job_crrnt + 1
                        
                        if LAST:
                            self.open_cores[i]='open'
                            self.indeces_RUN[i] = self.job_crrnt
                            self.job_crrnt = self.job_crrnt + 1
                        continue
                    if jobs[i].status=='warning': 
                        self.running_PLOT[i]=np.array([-0.5,-0.5])
                        try:
                            self.dft_steps_RES[job_ind[0],job_ind[1]][n_outer].append(self.get_dft_results_pyiron(jobs[i]))
                            if [job_ind[1],job_ind[0]] not in self.WARN_grid:
                                self.NOTC_grid.append([job_ind[1],job_ind[0]])
                                self.NOTC_color.append('green')

                            print('pass WARN')
                            self.print_cut()
                            self.steps_statuses[job_ind[0],job_ind[1]][n_outer].append('pass WARN')

                        except Exception as e:
                            print(e)
                            self.dft_steps_RES[job_ind[0],job_ind[1]][n_outer].append('error')

                            if [job_ind[1],job_ind[0]] not in self.NOTC_grid:
                                self.NOTC_grid.append([job_ind[1],job_ind[0]])
                                self.NOTC_color.append('red')

                            print('error WARN: ', e)


                            self.print_cut()
                            #self.steps_statuses[job_ind[0],job_ind[1]][n_outer].append('error ABOR')
                            
                            
                            ######
                            #self.err_arr=[np.empty((0,3),dtype=int),[],[]]
                            
                            
                            
                            err_count = np.count_nonzero(np.all(self.err_arr[0] == np.array(
                                [self.prs_ARR[job_ind[0]][job_ind[1]],n_outer,n_inner]), axis=1))
                            
                            
                                                   
                                                   
                            self.err_arr[0] = np.vstack((self.err_arr[0],
                                                        np.array(
                                [self.prs_ARR[job_ind[0]][job_ind[1]],n_outer,n_inner]) ))
                               
                            self.err_arr[1] = np.vstack((self.err_arr[1],job_ind))
                               
                            self.err_arr[2].append(jobs[i].name+'_err_{:02}'.format(err_count))
                               
                            self.err_arr[3].append(copy.deepcopy(self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]))
                            
                            if err_count >= 5:
                                print('giving up on this run')
                                self.open_cores[i]='open'
                                self.indeces_RUN[i] = self.job_crrnt
                                self.job_crrnt = self.job_crrnt + 1
                                continue
                            
                            #BM
                            jobs[i].copy_to(new_job_name=jobs[i].name+'_err_{:02}'.format(err_count),copy_files=True,
                                            new_database_entry=False)
                            
                            jobs[i]=proj_pyiron.create_job(job_type=proj_pyiron.job_type.Vasp,
                                              job_name='vasp_{:000006}'.format(
                                                  self.prs_ARR[job_ind[0]][job_ind[1]]
                                              )+'_{:02}'.format(
                                                  n_outer
                                              )+'_{:02}'.format(
                                                  n_inner
                                              ),
                                          delete_existing_job=True)
                            
                            
                            #### update
                            self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['job_specs']['KPPA']=\
                            self.steps_blank[n_outer][0]['job_specs']['KPPA']+500*(err_count+1)
                            self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['job_specs']['-INCAR-ENCUT']=\
                            self.steps_blank[n_outer][0]['job_specs']['-INCAR-ENCUT']+20*(err_count+1)
                            
                            #### update
                            #self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['job_specs']['k_mesh']=\
                            #self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['job_specs']['k_mesh']+1E-1
                            
                            #### update
                            #self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['job_specs']['-INCAR-ISYM']=\
                            #self.dict_steps[job_ind[0],job_ind[1]][n_outer][n_inner]['job_specs']['-INCAR-ISYM']*10
                            
                            #######
                            continue    ####Here when we get an error we continue to next n_inner who should not run again.... 
                                        ########then why?

                        #self.open_cores[i]='open'
                        
                        if self.progress[job_ind[0],job_ind[1]][1]==len(self.dict_steps[job_ind[0],job_ind[1]][n_outer])-1:
                            self.progress[job_ind[0],job_ind[1]][0] = self.progress[job_ind[0],job_ind[1]][0] + 1
                            self.progress[job_ind[0],job_ind[1]][1] = 0
                        else:
                            self.progress[job_ind[0],job_ind[1]][1] = self.progress[job_ind[0],job_ind[1]][1] + 1
                        #self.indeces_RUN[i] = self.job_crrnt
                        #self.job_crrnt = self.job_crrnt + 1
                        if LAST:
                            self.open_cores[i]='open'
                            self.indeces_RUN[i] = self.job_crrnt
                            self.job_crrnt = self.job_crrnt + 1
                        continue
                    
                
                
                        
    def relax_blank(self,job,dic):
        for key, value in dic.items():
            if key=='-INCAR-MAGMOM':
                
                #print('magmoms')
                #print()
                
                job.structure.set_initial_magnetic_moments(dic['-INCAR-MAGMOM'])
                continue
            if key.startswith('-INCAR-'):
                job.input.incar[key.split('-INCAR-')[-1]]=value
        
            
        
        if 'KPPA' in dic.keys():
            KPPA=dic['KPPA']
        else:
            KPPA=3000
        
        
        print(job.server.list_queues()) 
        if 'queue_number' in dic.keys():
            #print(dic['queue_number'])
            job.server.queue = job.server.list_queues()[dic['queue_number']]
        else:
            job.server.queue = job.server.list_queues()[0]
            
        #print(job.server.queue)
            
        if 'k_mesh' in dic.keys():
            if dic['k_mesh'] is list:
                job.k_mesh=dic['k_mesh']
            else:
                job.k_mesh_spacing=dic['k_mesh']
        else:
            ### RUN KPPA
            work_str=Kpoints().automatic_density(ase_to_pmg(job.structure), kppa=KPPA)
            print('change')
            print(work_str)
            
            job.set_kpoints(mesh=work_str.as_dict()['kpoints'][0],center_shift=[0,0,0])
            #job.set_kpoints_file(method='Gamma', size_of_mesh=mesh=work_str.as_dict()['kpoints'][0])
        
        
        job.run()
        
        
    def get_dft_results_pyiron(self,job_func):
        return {'job_name':job_func.job_name,
                'status':job_func.status.string,
                'forces':job_func['output/generic/forces'].copy(),
                'stresses':job_func['output/generic/stresses'].copy(),
                'energies':job_func['output/generic/energy_pot'].copy(),
                'trajectories':[j.to_ase().copy() for j in job_func.trajectory()],
		'internal_energies': job_func['output/generic/dft/scf_energy_int'].copy()}

                    
                    
                    
    def apply_strain(self,ase_struct,strain):
        if strain is None:
            return ase_struct
        ase_new=ase_struct.copy()
        
        ase_new.set_cell(ase_struct.get_cell().array@(np.identity(3)+strain),scale_atoms=True)
        
        
        #
        ##### modify struct based in strain #####
        #
        return ase_new           


    def get_errs(self):
        directories = []
        inds = []
        directory= self.DFT_dir_PA
        if os.path.isdir(directory):
            for entry in os.scandir(directory):
                if entry.is_dir():
                    if 'err' in entry.name:
                        directories.append(entry.name)

                        ind = entry.name.split('_')
                        ind = ind[1:4]
                        ind = [int(i) for i in ind]
                        inds.append(ind)
            if len(inds)>0:
                return [np.array(inds), np.array([[-1,-1]]*len(inds)), directories, [{}]*len(inds)]
            else: 
                return [np.empty((0,3),dtype=int),np.empty((0,2),dtype=int),[],[]]
        else:
            return [np.empty((0,3),dtype=int),np.empty((0,2),dtype=int),[],[]]
                
                
            #### When vasp6 gets stuck... but probably pyiuron is the one who is not completting it


    def abort_specific(self,inds,DELETE=False):
        
        proj_pyiron = Project(path=self.DFT_dir_PA)
        
        try:
            jobs=proj_pyiron.create_job(job_type=proj_pyiron.job_type.Vasp,
                                  job_name='vasp_{:000006}'.format(
                                      inds[0]
                                  )+'_{:02}'.format(
                                      inds[1]
                                  )+'_{:02}'.format(
                                      inds[2]
                                  ),
                              delete_existing_job=DELETE)
        except:
            jobs=proj_pyiron.load('vasp_{:000006}'.format(
                                      inds[0]
                                  )+'_{:02}'.format(
                                      inds[1]
                                  )+'_{:02}'.format(
                                      inds[2]
                                  ))
        
        print('status : ',jobs.status)
        
        
        #print(dir(jobs))
        jobs.drop_status_to_aborted()

        print('status : ',jobs.status)

    def check_current(self):
        for i,ii in enumerate(self.dft_steps_RES):
            for j,jj in enumerate(self.dft_steps_RES[i]):
                print('###########################################################')
                print('[',i,j,']')
                print(self.prs_ARR[i,j])
                #print(len(run_job.dft_steps_RES[i,j]))
                #print(len(run_job.dft_steps_RES[i,j][0]))
                if (self.dft_steps_RES[i,j][0] is not None) and (self.dft_steps_RES[i,j][0] != "error") and (len(self.dft_steps_RES[i,j][0])>0):

                    print(self.dft_steps_RES[i,j][0][0]['job_name'])
                    if abs(self.dft_steps_RES[i,j][0][0]['energies'][-1]-self.dft_steps_RES[i,j][0][0]['energies'][-2])>1E-2:
                        ### check if it actually converges ZBRENT problem
                        print('ENERGY CONVERGENCE ISSUE, LAST STEPS')
                    if self.dft_steps_RES[i,j][0][0]['energies'][-1]> 0.0:
                        ### check if it actually converges ZBRENT problem
                        print('POSITIVE ENERGY')
                        continue


                    
                    print(len(self.dft_steps_RES[i,j][0]),
                         len(self.dft_steps_RES[i,j][1]),
                         len(self.dft_steps_RES[i,j][2]),
                         )
                else:
                    continue
                    
                    
    def getting(self):
        indexes = []
        sizes   = []
        for i,ii in enumerate(self.dft_steps_RES):
            for j,jj in enumerate(self.dft_steps_RES[i]):
                if (self.dft_steps_RES[i,j][0] is not None) and (self.dft_steps_RES[i,j][0] != "error") and (len(self.dft_steps_RES[i,j][0])>0):
                    ##check if the one is actually ran

                    if len(self.dft_steps_RES[i,j][1]) == len(self.dft_steps_RES[i,j][2]):
                        ##check if actually ran the whole elastic steps

                        indexes.append([i,j])
                        sizes.append(len(self.ics_ARR[i][j]))
                else:
                    continue
        return indexes, sizes

    def getting_with_n_frac(self, n_size, fraction):
        indexes = []
        sizes   = []
        for i,ii in enumerate(self.dft_steps_RES):
            for j,jj in enumerate(self.dft_steps_RES[i]):
                if (self.dft_steps_RES[i,j][0] is not None) and (self.dft_steps_RES[i,j][0] != "error") and (len(self.dft_steps_RES[i,j][0])>0):
                    ##check if the one is actually ra

                    if len(self.dft_steps_RES[i,j][1]) == len(self.dft_steps_RES[i,j][2]):
                        ##check if actually ran the whole elastic steps

                        indexes.append([i,j])
                        sizes.append(len(self.ics_ARR[i][j]))
                else:
                    continue


        uni=list(set(sizes))
        uni.sort()
        uni_i = []
        train=[]
        test=[]
        for i in uni:



            iter_li=[index for index, item in enumerate(sizes) if item == i]
            uni_i.append(iter_li)

            if i >= n_size:
    #             print(i)
    #             print(len(iter_li))

                sel_n=int(fraction*len(iter_li))

    #             print(iter_li)

    #             print(sel_n)


                np.random.seed(seed=42)
                s_li = np.random.choice(iter_li,sel_n,replace=False)

                s_li.sort()

    #             print(s_li)

                n_li=[i for i in iter_li if i not in s_li]
                #print(n_li)
                train.extend(s_li)
                test.extend(n_li)
            else:
                train.extend(iter_li)
        #print(uni)
        #print(uni_i)
        train= np.array(indexes)[train]
        test= np.array(indexes)[test]


        return train, test
    
    def getting_data_set(self, set_, nns):
        all_ener      = []
        all_forces    = []
        all_stss_vasp = []
        all_traj      = []
        for ij in set_:
            for k in nns:
                #print(ij,k)
                for run in self.dft_steps_RES[ij[0],ij[1]][k]:
                    #print(len(run))
                    #print(run['trajectories'])

                    all_ener.append(run['energies'])
                    all_forces.append(run['forces'])
                    all_stss_vasp.append(run['stresses']*self.CONV)
                    all_traj.append(run['trajectories'])




        a_e=np.concatenate(all_ener)
        #a_f=np.array(list(chain.from_iterable(all_forces)),dtype=object)
        #a_s=np.array(list(chain.from_iterable(all_stss_vasp)),dtype=object)
        #a_t=np.array(list(chain.from_iterable(all_traj)),dtype=object)
        a_f=list(chain.from_iterable(all_forces))
        a_s=list(chain.from_iterable(all_stss_vasp))
        a_t=list(chain.from_iterable(all_traj))
        #print(a_t)
        return [a_t,a_e,a_f,a_s]
    
    def getting_data_set_clean(self, set_, nns):
        all_ener      = []
        all_forces    = []
        all_stss_vasp = []
        all_traj      = []
        for ij in set_:
            for k in nns:
                #print(ij,k)
                for run in self.dft_steps_RES[ij[0],ij[1]][k]:
                    iter_=[]
                    ener_iter = run['internal_energies']
                    for i,ii in enumerate(ener_iter):
        #                     if run['energies'][i]> 0.0:
        #                         print('HERE : ',len(ii))
        #                         print(run['job_name'], i)
        #                         print(ii)

                        if len(ii)>99 or run['energies'][i]> 0.0:
                            pass
                        else:
                            iter_.append(i)



                    all_ener.append([run['energies'][i] for i in iter_])
                    all_forces.append([run['forces'][i] for i in iter_])
                    all_stss_vasp.append([run['stresses'][i]*self.CONV for i in iter_])
                    all_traj.append([run['trajectories'][i] for i in iter_])


        a_e=np.concatenate(all_ener)
        #a_f=np.array(list(chain.from_iterable(all_forces)),dtype=object)
        #a_s=np.array(list(chain.from_iterable(all_stss_vasp)),dtype=object)
        #a_t=np.array(list(chain.from_iterable(all_traj)),dtype=object)
        a_f=list(chain.from_iterable(all_forces))
        a_s=list(chain.from_iterable(all_stss_vasp))
        a_t=list(chain.from_iterable(all_traj))
        #print(a_t)
        return [a_t,a_e,a_f,a_s]

    def __getstate__(self):
        state = self.__dict__.copy()
        #print(state.keys())
        #del(state['dft_steps_RES'])
        #print(state)
        return state
    
    
    
    def all_calculated_elastic(self,step):
        res_ohess=[]
        help_i=[]
        for i,ii in enumerate(self.dft_steps_RES):
            for j,jj in enumerate(self.dft_steps_RES[i]):
                help_i.append(self.ics_ARR[i][j])
                if (self.dft_steps_RES[i,j][0] is not None) and (self.dft_steps_RES[i,j][0] != "error") and \
                (len(self.dft_steps_RES[i,j][0])>0):

                    if len(self.dft_steps_RES[i,j][1]) == len(self.dft_steps_RES[i,j][2]) and \
                    len(self.dft_steps_RES[i,j][1]) != 0:


                        #print(self.dft_steps_RES[i,j][0][0]['job_name'])
                        iter_res = get_last_from_step(self.dft_steps_RES[i,j],step)

                        #print(len(iter_res))
                        sf=np.array([stress_tens_to_voigt(i[3]) for i in iter_res])
                        # print('len(sf)',len(sf))
                        # print('sf',sf)
                        # print('len(self.strains_per_mag[i,j])',len(self.strains_per_mag[i,j]))
                        # print('sf',self.strains_per_mag[i,j])
                        #print(len(run_job.strains_per_mag[i,j]),len(sf))
                        stress_dict=stress_to_dict(self.lst_strains, stress_accom(self.strains_per_mag[i,j],sf))
                        res_ohess.append(calc_elastic_constants(self.input_dict,None, 
                                    group_ordering(self.dft_steps_RES[i,j][0][0]['trajectories'][-1].copy()),{},stress_dict))


        return help_i,res_ohess

    
    def all_calculated(self,step):
        for i,ii in enumerate(self.dft_steps_RES):
            for j,jj in enumerate(self.dft_steps_RES):
                if (self.dft_steps_RES[i,j][0] is not None) and (self.dft_steps_RES[i,j][0] != "error") and \
                (len(self.dft_steps_RES[i,j][0])>0):
                    iter_res = get_last_from_step(self.dft_steps_RES[i,j],step)
                    print(len(iter_res))

                    

    def getting_indset_elas_with_n_frac(self, n_size, fraction):
        indexes = []
        sizes   = []
        for i,ii in enumerate(self.dft_steps_RES):
            for j,jj in enumerate(self.dft_steps_RES[i]):
                if (self.dft_steps_RES[i,j][0] is not None) and (self.dft_steps_RES[i,j][0] != "error") and (len(self.dft_steps_RES[i,j][0])>0):
                    ##check if the one is actually ran
                    if abs(self.dft_steps_RES[i,j][0][0]['energies'][-1]-self.dft_steps_RES[i,j][0][0]['energies'][-2])>1E-2:
                        ### check if it actually converges ZBRENT problem
                       continue
                    if self.dft_steps_RES[i,j][0][0]['energies'][-1]>0.0:
                        ### check if energy is positive
                       continue
                    if len(self.dft_steps_RES[i,j][1]) == len(self.dft_steps_RES[i,j][2]):
                        indexes.append([i,j])
                        sizes.append(len(self.ics_ARR[i][j]))
                else:
                    continue
        uni=list(set(sizes))
        uni.sort()
        uni_i = []
        train=[]
        test=[]
        for i in uni:
            iter_li=[index for index, item in enumerate(sizes) if item == i]
            uni_i.append(iter_li)
            if i >= n_size:
                sel_n=int(fraction*len(iter_li))
                s_li = np.random.choice(iter_li,sel_n,replace=False)
                s_li.sort()
                n_li=[i for i in iter_li if i not in s_li]
                train.extend(s_li)
                test.extend(n_li)
            else:
                train.extend(iter_li)
        train= np.array(indexes)[train]
        test= np.array(indexes)[test]
        return train, test
#    def __setstate__(self, state):
#        self.__dict__.update(state)
#        f = open(self.filename)
#        self.f = f
###END OBJECT



def return_one_run(entry_RES):
    return [entry_RES['energies'],
                entry_RES['forces'],
                entry_RES['trajectories'],
                entry_RES['stresses'],]

def return_last_run(entry_RES):
    return [entry_RES['energies'][-1],
                entry_RES['forces'][-1],
                entry_RES['trajectories'][-1],
                entry_RES['stresses'][-1],]

def get_last_from_step(st,step):
    res=[]
    for i in st[step]:
        res.append(return_last_run(i))
    return res

def get_sca_pos(SC_pos,PC_cell):
    ZERO=1E-12
    
    sca_pos =[]
    SC_pos = SC_pos - SC_pos[0] -1
    for vec in SC_pos:
        work_vec=np.linalg.solve(PC_cell.T,vec)
        work_vec=work_vec%1.0
        work_vec=work_vec%(1-ZERO)
        
        sca_pos.append(work_vec)
    sca_pos = np.mod(sca_pos,1.000000)
    
    return np.array(sca_pos)


def get_unit_pos(SC_pos,PC_cell):
    ZERO=1E-12
    
    sca_pos =[]
    SC_pos = SC_pos - SC_pos[0]
    for vec in SC_pos:
        work_vec=np.linalg.solve(PC_cell.T,vec)
        work_vec=work_vec%1.0
        work_vec=work_vec%(1-ZERO)
        
        sca_pos.append(work_vec)
    print('sca_pos',sca_pos)
    sca_pos = np.mod(sca_pos,1.000000)
    print('sca_pos',sca_pos)
    
    return np.array(sca_pos)@PC_cell


def get_relax_val_from_model(tr_dict_vasp,model_relaxer,per_atom=False,CONV=-1.6021766208E2,return_basic=True):
    res_tt=[]
    for i,ii in enumerate(tr_dict_vasp):
        relax_results = model_relaxer.relax(ase_to_pmg(tr_dict_vasp[i]['trajectories'][0].copy()))
        final_ase = relax_results['trajectory'].atoms.copy()
        if per_atom:
            n_atoms = len(final_ase)
        else:
            n_atoms = 1.0
            
            
        vasp_for = tr_dict_vasp[i]['forces'][-1].flatten()
        m3gn_for = relax_results['trajectory'].forces[-1].flatten()
        
        vasp_sts = (tr_dict_vasp[i]['stresses'][-1]*CONV).flatten()
        
        rear=relax_results['trajectory'].stresses[-1]*100
        m3gn_sts = np.array([[rear[0],rear[5],rear[4]],[rear[5],rear[1],rear[3]],[rear[4],rear[3],rear[2]]]).flatten()
        
        
        #instead
        #m3gn_for = potential.get_efs(ase_to_pmg(tr_dict_vasp[i]['trajectories'][-1]))[1].numpy().flatten()
        #instead 
        #m3gn_sts = potential.get_efs(ase_to_pmg(tr_dict_vasp[i]['trajectories'][-1]))[2].numpy().flatten()
        
        
        
        # print(final_ase)
        # print(tr_dict_vasp[i]['trajectories'][-1])


        # print('pos')
        # print(tr_dict_vasp[i]['trajectories'][-1].get_positions())
        # print(final_ase.get_positions())
        
        # print('pos')
        # print(tr_dict_vasp[i]['trajectories'][-1].get_scaled_positions())
        # print(final_ase.get_scaled_positions())



        #unit_m=get_unit_pos(relax_results['trajectory'].atom_positions[-1], relax_results['trajectory'].cells[-1])
        # unit_m=get_unit_pos(final_ase.get_positions(),
        #                   final_ase.cell.array)
        # unit_v=get_unit_pos(tr_dict_vasp[i]['trajectories'][-1].get_positions(),
        #                   tr_dict_vasp[i]['trajectories'][-1].cell.array)
        unit_m=final_ase.get_positions()-final_ase.get_positions()[0]
        unit_v=tr_dict_vasp[i]['trajectories'][-1].get_positions()- tr_dict_vasp[i]['trajectories'][-1].get_positions()[0]
        
        # print('unitsss')
        # print(unit_v)
        # print(unit_m)
        


        #print('values_from_model')
        #print('CELLS')
        #print(relax_results['trajectory'].cells[-1])
        #print(tr_dict_vasp[i]['trajectories'][-1].cell.array)
        
        
        
        if return_basic:

            res_tt.append([[tr_dict_vasp[i]['energies'][-1]/n_atoms,float(relax_results['trajectory'].energies[-1])/n_atoms],
                           [tr_dict_vasp[i]['trajectories'][-1].get_volume()/n_atoms,final_ase.get_volume()/n_atoms],
                          #[vasp_for, m3gn_for],
                          #[vasp_sts,m3gn_sts],
                           [unit_v,unit_m]
                          ])
        else:
            res_tt.append([[tr_dict_vasp[i]['energies'][-1]/n_atoms,float(relax_results['trajectory'].energies[-1])/n_atoms],
                           [tr_dict_vasp[i]['trajectories'][-1].get_volume()/n_atoms,final_ase.get_volume()/n_atoms],
                          [vasp_for, m3gn_for],
                          [vasp_sts,m3gn_sts],
                           [unit_v,unit_m]
                          ])
        
    
    return res_tt
