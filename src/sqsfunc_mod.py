from itertools import chain, combinations
import numpy as np
import pandas as pd
import json
import shutil
from collections import OrderedDict
from ase import Atoms
import os
from datetime import datetime
import time
import operator
import subprocess
import matplotlib.pyplot as plt
from ase.build.supercells import make_supercell


'''change NiTi hard coded functions for general ones'''

ZERO=1E-12



vesta_colors = { 
    "H": "#FFFFFF", "He": "#C4C4C4", "Li": "#8F40D4", "Be": "#FFFF00", "B": "#FF8000", "C": "#909090", "N": "#4040FF", "O": "#FF0000", "F": "#80FF80", "Ne": "#8080FF", "Na": "#FFC832", "Mg": "#00C832", "Al": "#C86400", "Si": "#D9D9D9", "P": "#FFA040", "S": "#FFFF40", "Cl": "#00FF00", "Ar": "#A1A1A1", "K": "#C832FF", "Ca": "#A1A1A1", "Sc": "#A1A1A1", "Ti": "#A1A1A1", "V": "#A1A1A1", "Cr": "#A1A1A1", "Mn": "#A1A1A1", "Hf": "yellow", "Fe": "#A1A1A1", "Co": "#A1A1A1", "Ni": "#00FF00", "Cu": "#FF8000", "Zn": "#A1A1A1", "Ga": "#C86400", "Ge": "#A1A1A1", "As": "#FF8000", "Se": "#FFFF40", "Br": "#008040", "Kr": "#A1A1A1", "Rb": "#C832FF", "Sr": "#A1A1A1", "Y": "#A1A1A1", "Zr": "#A1A1A1", "Nb": "#A1A1A1", "Mo": "#A1A1A1", "Tc": "#A1A1A1", "Ru": "#A1A1A1", "Rh": "#A1A1A1", "Pd": "#A1A1A1", "Ag": "#A1A1A1", "Cd": "#A1A1A1", "In": "#C86400", "Sn": "#A1A1A1", "Sb": "#FF8000", "Te": "#FFFF40", "I": "#8000FF", "Xe": "#A1A1A1", "Cs": "#C832FF", "Ba": "#A1A1A1", "La": "#A1A1A1", "Ce": "#A1A1A1", "Pr": "#A1A1A1", "Nd": "#A1A1A1", "Pm": "#A1A1A1", "Sm": "#A1A1A1", "Eu": "#A1A1A1", "Gd": "#A1A1A1", "Tb": "#A1A1A1", "Dy": "#A1A1A1",
}

atomic_radii = {
 "H": 37, "He": 32, "Li": 134, "Be": 90, "B": 82, "C": 77, "N": 75, "O": 73, "F": 71, "Ne": 69, "Na": 154, "Mg": 130, "Al": 118, "Si": 111, "P": 106, "S": 102, "Cl": 99, "Ar": 97, "K": 196, "Ca": 174, "Hf": 150, "Sc": 144, "Ti": 180, "V": 125, "Cr": 127, "Mn": 139, "Fe": 125, "Co": 126, "Ni": 121, "Cu": 138, "Zn": 131, "Ga": 126, "Ge": 122, "As": 119, "Se": 116, "Br": 114, "Kr": 110, "Rb": 211, "Sr": 192, "Y": 162, "Zr": 148, "Nb": 137, "Mo": 145, "Tc": 156, "Ru": 126, "Rh": 135, "Pd": 131, "Ag": 153, "Cd": 148, "In": 144, "Sn": 141, "Sb": 138, "Te": 135, "I": 133, "Xe": 130, "Cs": 225, "Ba": 198, "La": 169, "Ce": 154, "Pr": 247, "Nd": 206, "Pm": 205, "Sm": 238, "Eu": 231, "Gd": 233, "Tb": 225, "Dy": 228, "Ho": 226, "Er": 226, "Tm": 222, "Yb": 222, "Lu": 217,
}

###### ADDED FOR SAMPLING #######

def create_sampling(ele_subl, elements, n_comps = 4, lower = 2, upper = 4):
    sublattices, element_indices, sublattices_size = get_unique_list_of_lists(ele_subl,elements)
    
    if len(sublattices) == 1:
        order_elements = sublattices[0]
        order_elements = sorted(order_elements)
        final_comp = _create_sampling_active_elements(order_elements, n_comps = n_comps, lower = lower, upper = upper)
    else:
        sublattice_mulplicity = [len(i) for i in sublattices]
        if np.sum([i > 1 for i in sublattice_mulplicity]) > 1:
            raise Exception('No systems with more than one active sublattice')
        active_sublattice_index = np.where(np.array(sublattice_mulplicity) > 1)[0][0]
        sampling_elements = sublattices[active_sublattice_index]
        
        active_comp = _create_sampling_active_elements(sampling_elements, n_comps = n_comps, lower = lower, upper = upper)
        weights_sublattices = sublattices_size/np.sum(sublattices_size)
    
        sublattice_comps = []
        
        
        for i, sublattice_i in enumerate(sublattices):
            
            if i == active_sublattice_index:
                sublattice_comps.append(active_comp*weights_sublattices[i])
            else:
                sublattice_comps.append((np.ones(active_comp.shape[0])*(weights_sublattices[i])).reshape(-1, 1))
        
        final_comp = np.concatenate(sublattice_comps, axis = 1)

        order_elements = [item for row in sublattices for item in row]
        order_elements, final_comp = _get_duplicates(order_elements, final_comp)
    
    return order_elements, final_comp



def _get_duplicates(as_created, comp):
    unique = list(set(as_created))
    unique = sorted(unique)
    arrs = []
    for i in unique:
        arrs.append( comp[:, np.where(np.array(as_created) == i)[0]].sum(axis = 1) )
        # print(comp[:, np.where(np.array(order_elements) == i)[0]].sum(axis = 1))
    return unique, np.array(arrs).T

def _create_sampling_active_elements(sampling_elements, n_comps = 4, lower = 2, upper = 4):
    sys_d = len(sampling_elements)
    # n_comps = 4 # number of compositions per 100 at%
    # lower = 2 # dimension of lowest subsystem
    # upper = 4 # dimension of largest subsystem, can't be larger than n_comps
    final_comp=np.empty([0,sys_d], float)
    
    for i in range(lower,upper+1):
        # creation of compositions datapoints 
        # per size of subsystem
        comps = []
        indices = np.ndindex(*[n_comps-i for i in range(i)])
        j = 0
        for index in indices:
            j += 1
            comp = np.array(index)+1
            if (np.sum(comp)==(n_comps)):
                comps.append(comp)
                toc = time.time()
        comps = np.vstack(comps)/(n_comps)
        # creation of subsystems per size of subsystem
        # (combinations of subsystems of same size in 
        # in the 7-atom system
        for comb in combinations(range(sys_d),i):
            dummy=np.zeros([comps.shape[0],sys_d],dtype=float)
            dummy[:,comb]=comps
            final_comp=np.vstack((final_comp, dummy))

        
        
    
    
    return final_comp



##### Added diborides ####

class mini_structures_pool:
    
    def __init__(self, system_elements):
        self.system_elements = system_elements
        
        
        self.structures = []
        self.sizes = []
        
    def _create_prim(self,  system_lat, lattice_vals, extra_sites = None):
        if system_lat == 'FCC' or system_lat == 'BCC':
            self.a0 = lattice_vals['a0']
        if system_lat == 'HCP' or system_lat == 'boride_191':
            self.a0 = lattice_vals['a0']
            self.c0 = lattice_vals['c/a'] * self.a0
            
        if system_lat == 'FCC' and extra_sites == None:
            a0 = self.a0
            self.prim = Atoms(self.system_elements[0],
                 scaled_positions=[
                     [0.0,   0.0,   0.0],
                 ],
                 cell = [[a0/2,a0/2,0],
                         [0,a0/2,a0/2],
                         [a0/2,0,a0/2],
                        ],
                pbc=[1,1,1]
                        )
            
        if system_lat == 'BCC' and extra_sites == None:
            a0 = self.a0
            self.prim = Atoms(self.system_elements[0], 
                scaled_positions=[
                 [0.0,   0.0,   0.0],
                ],
                cell = [[a0/2,a0/2,-a0/2],
                     [-a0/2,a0/2,a0/2],
                     [a0/2,-a0/2,a0/2],
                    ],
                pbc=[1,1,1])
        
        if system_lat == 'HCP' and extra_sites == None:
            a0 = self.a0
            c0 = self.c0
            self.prim = Atoms(self.system_elements[0]*2,
                scaled_positions=[
                    [1/3, 2/3, 1/4],    
                    [2/3, 1/3, 3/4]
                ],
                cell = [[np.abs(a0*np.cos(120*np.pi/180)), -np.abs(a0*np.sin(120*np.pi/180)), 0.0],
                     [np.abs(a0*np.cos(120*np.pi/180)), np.abs(a0*np.sin(120*np.pi/180)), 0.0],
                     [0.0,0.0,c0],
                    ],
                pbc=[1,1,1])
            
            
        if system_lat == 'boride_191' and extra_sites == None:
            a0 = self.a0
            c0 = self.c0
            self.prim = Atoms(self.system_elements[0]+'B'*2,
                scaled_positions=[
                    [0.0, 0.0, 0.0],
                    [1/3, 2/3, 1/2],
                    [2/3, 1/3, 1/2]
                ],
                cell = [[np.abs(a0*np.cos(120*np.pi/180)), -np.abs(a0*np.sin(120*np.pi/180)), 0.0],
                     [np.abs(a0*np.cos(120*np.pi/180)), np.abs(a0*np.sin(120*np.pi/180)), 0.0],
                     [0.0,0.0,c0],
                    ],
                pbc=[1,1,1])
        print(self.prim)



def arrays_equal_with_tolerance(arr1, arr2, tol = 1E-8):
    if len(arr1) != len(arr2):
        return False
    else:
        for i, _ in enumerate(arr1):
            a1 = np.sort(arr1[i])
            a2 = np.sort(arr2[i])
            if not np.allclose(a1, a2, atol=tol):
                return False
    return True


def add_sym(df):
    non_sym = []
    df_out = df.copy()
    df_out ['sym'] = None
    
    for ii, i_row in df.iterrows():
        unique = True
        if ii in non_sym:
            continue
        this_row = i_row['sublattice'].copy()
        for jj, j_row in df[ii+1:].iterrows():
            if arrays_equal_with_tolerance(this_row, j_row['sublattice'].copy() ):
                
                non_sym.append(jj)
                df_out.at[jj, 'sym'] = ii
        df_out.at[ii, 'sym'] = 'Y'
    return df_out


##############








def elements_from_subl(ele_subl):
    test_list = [item for row in ele_subl for item in row]
    res = []
    [res.append(str(x)) for x in test_list if str(x) not in res]
    # print('res', res)
    return sorted(res)


def get_unique_list_of_lists(SUBL, ELEMENTS):

    sizes = []
    inds_ELE = []
    
    unique_list = list(set(tuple(sublist) for sublist in SUBL))
    unique_list = [list(i) for i in unique_list]
    for i in unique_list:
        inds_ELE.append([ELEMENTS.index(j) for j in i])
        sizes.append(sum(1 for sublist in SUBL if sublist == i))
    
    return unique_list, inds_ELE, sizes


'''dataframes functions'''
def add_unique(df_i):
    df=df_i.copy()
    for i in range(len(df)):
        if df.loc[i,'unique']=='Y':
            for j in range(i+1,len(df)):
                #print(i,j)
                #print(df.loc[i,'sublattice'])
                if np.array_equal(df.loc[i,'sublattice'].astype(int), df.loc[j,'sublattice'].astype(int)):
                    df.loc[j,'unique']=i
            df.loc[i,'unique']='Y'
    return df

def display_full(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        display(df)



'''ASE and matrices own functions'''

def fix_ase(ase_clean):
    '''change ASE atom positions alphabetically
    and then positionally x,y,z'''
    ase_cell=ase_clean.copy()
    pos=ase_cell.get_scaled_positions()
    aa = ase_cell.get_chemical_symbols()
    sort_ind=np.array(aa).argsort()
    
    ase_cell.set_scaled_positions(pos[sort_ind])
    ase_cell.set_chemical_symbols(np.array(aa)[sort_ind])
    
    aa = ase_cell.get_chemical_symbols()
    
    uni=list(set(aa))
    
    aa=np.array(aa)
    
    
    dummy = ase_cell.get_scaled_positions()
    
    new_array = np.zeros(dummy.shape,dtype=float)
    
    
    for i in uni:
        ind_list = np.where(aa==i)[0]
        new_array[ind_list] = sorted(dummy[ind_list], key=operator.itemgetter(0, 1, 2))
        
    ase_cell.set_scaled_positions(new_array)
    return ase_cell

def fix_shuffle(matr):
    '''FIX matrix shuffle to be the number with the lesser magnitude'''
    matr = matr.copy()
    if len(matr.shape)==2:
        ni,nj= matr.shape
        for i in range(ni):
            for j in range(nj):
                if 1 - abs(matr[i,j]) < abs(matr[i,j]):
                    if matr[i,j] < 0.0:
                        matr[i,j]= matr[i,j]%1
                    else:
                        matr[i,j]= matr[i,j] - 1.0
                else:
                    continue

        return matr


def ase_to_rndstr(ase_cell,file_name=''):
    '''from ase structure to writing into a rndstr file!'''
    ll,bb = ase_cell.get_cell().array,ase_cell.get_scaled_positions()
    aa = ase_cell.get_chemical_symbols()
    res_txt = ''
    res_txt = res_txt + str(ll[0])[1:-1]+'\n'+str(ll[1])[1:-1]+'\n'+str(ll[2])[1:-1]+'\n'
    res_txt = res_txt + '1 0 0'+'\n'+'0 1 0'+'\n'+'0 0 1'+'\n\n'
    
    saved=aa[0]
    for i,j in zip(bb,aa):
        if j==saved:
            pass
        else:
            res_txt = res_txt +'\n'
            saved=j
            
        res_txt = res_txt + '{0:1.6f}'.format(i[0].round(6))+' '\
                +'{0:1.6f}'.format(i[1].round(6))+' '\
                +'{0:1.6f}'.format(i[2].round(6))+' '\
                +j+'\n'
        
    if file_name != '':
        with open(file_name, "w") as out_file:
            out_file.write(res_txt)
    return res_txt

def ase_to_rndstr_subl(ase_cell,subl,file_name=''):
    '''from ase structure to writing into a rndstr file!

    ase_cell; an ase structure where the chemical symbols 
    don't really matter
    
    subl; list that correspond to each atom and the elements that can ocupy it'''
    
    
    unq_list, _ , unq_siz = get_unique_list_of_lists(subl, elements_from_subl(subl))
    unq_subl_ind = []
    for i in unq_list:
        unq_subl_ind.append([index for index, value in enumerate(subl) if value == i])
    
    ll,bb = ase_cell.get_cell().array,ase_cell.get_scaled_positions()
    
    res_txt = ''
    res_txt = res_txt + str(ll[0])[1:-1]+'\n'+str(ll[1])[1:-1]+'\n'+str(ll[2])[1:-1]+'\n'
    res_txt = res_txt + '1 0 0'+'\n'+'0 1 0'+'\n'+'0 0 1'+'\n\n'
    

    
    for i,[i_u,subl_u] in enumerate(zip(unq_list,unq_subl_ind)):
        res_txt = res_txt +'\n'
        for j in subl_u:
            res_txt = res_txt + '{0:1.6f}'.format(bb[j,0].round(6))+' '\
                    +'{0:1.6f}'.format(bb[j,1].round(6))+' '\
                    +'{0:1.6f}'.format(bb[j,2].round(6))+' '\
                    +'SUBL_'+'{:02d}'.format(i)+'\n'
        
    if file_name != '':
        with open(file_name, "w") as out_file:
            out_file.write(res_txt)
    return res_txt

def create_ase(ll,bb,aa):
    
    return Atoms(aa, positions=bb,cell=ll)






'''folder, files and other functions'''


def folder_or_delete(dir_name, delete=True):
    '''checks if there's a floder with that name, if not it creates it, 
    if yes it deletes it and it creates it again'''

    if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    else:
        if delete:
            shutil.rmtree(dir_name)
            os.mkdir(dir_name)
        else:
            pass
        
def nth_repl(s, sub, repl, n):
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find+len(sub):]
    return s

def send_job(file_name,SQS_dir):
    '''not using it anymore since it is already on the class'''
    send_job=['sbatch',file_name]
    wd = os.getcwd()
    os.chdir(SQS_dir)
    subprocess.run(send_job)
    os.chdir(wd)

def write_log(file_log,string_to_write):
    with open(file_log, "a+") as in_file:
            in_file.write(string_to_write+'\n')
def open_log(file_log):
    with open(file_log, "w") as in_file:
            in_file.write('LOG\n')

'''SQS function creations'''



def compositions_in_2s(vec, elements):
    '''get vector of compositions, and element names 
    list and output to two dictionaries ordered, 
    extremely hard-coded for the CuHfNiTi system 
    but will accept elements and comp in any order'''
    
    niquel_subl=['Ni','Cu']
    titani_subl=['Ti','Hf']
    
    
    ni_sub_content={'Ni':0.0,
                    'Cu':0.0,
                    'Hf':0.0,
                    'Ti':0.0,
                   }
    ti_sub_content={'Ni':0.0,
                    'Cu':0.0,
                    'Hf':0.0,
                    'Ti':0.0,
                   }
    n_s=[elements.index(i) for i in niquel_subl]
    n_c=vec[n_s].sum()
    
    
    
    if n_c < 0.5:
        ni_sub_content['Ni']=vec[elements.index('Ni')]
        ni_sub_content['Cu']=vec[elements.index('Cu')]
        
    else:
        if  vec[elements.index('Ni')] > 0.5:
            ni_sub_content['Ni']=0.5
            
            ti_sub_content['Cu']=vec[elements.index('Cu')]
            ti_sub_content['Hf']=vec[elements.index('Hf')]
            ti_sub_content['Ni']=vec[elements.index('Ni')]-0.5
            ti_sub_content['Ti']=vec[elements.index('Ti')]
        else:
            ni_sub_content['Ni']=vec[elements.index('Ni')]
            ni_sub_content['Cu']=0.5-vec[elements.index('Ni')]
            
            
            ti_sub_content['Cu']=vec[elements.index('Cu')]-ni_sub_content['Cu']
            
    
    t_s=[elements.index(i) for i in titani_subl]
    t_c=vec[t_s].sum()
    
    
    
    if t_c < 0.5:
        ti_sub_content['Ti']=vec[elements.index('Ti')]
        ti_sub_content['Hf']=vec[elements.index('Hf')]
        
    else:
        if  vec[elements.index('Ti')] > 0.5:
            ti_sub_content['Ti']=0.5
            
            ni_sub_content['Cu']=vec[elements.index('Cu')]
            ni_sub_content['Hf']=vec[elements.index('Hf')]
            ni_sub_content['Ni']=vec[elements.index('Ni')]
            ni_sub_content['Ti']=vec[elements.index('Ti')]-0.5
        else:
            ti_sub_content['Ti']=vec[elements.index('Ti')]
            ti_sub_content['Hf']=0.5-vec[elements.index('Ti')]
            
            
            ni_sub_content['Hf']=vec[elements.index('Hf')]-ti_sub_content['Hf']    
    ni_sub_content = OrderedDict(sorted(ni_sub_content.items()))
    ti_sub_content = OrderedDict(sorted(ti_sub_content.items()))
    return ni_sub_content,ti_sub_content


def SQS_size(around_number, dic, latt_option, ull):
    '''creates an SQS preferable size based on the lattice 
    and on the composition, if we choose B2 then SQSs
    are looked in steps of 2, else in steps of 4... DIV_SUB
    is the number of atoms per sublattice SQS_size is hard-coded
    for the number of sublattices (2)'''
    score=[]
    final_vec=[]
    a_n=around_number
    if latt_option=='B2':
        DIV_SUB = 1
    else:
        DIV_SUB = 2
    
    
    
    DIV = DIV_SUB*2
    
    limi=DIV*ull
    
    
    if a_n%DIV> ZERO:
        a_n=np.round(a_n/DIV,0)*DIV
    
    for i_n in np.arange(a_n-limi,a_n+limi+1, DIV):
        
        per_lattice=i_n/2
        
        
        val=np.array([np.fromiter(dic[0].values(), dtype=float), np.fromiter(dic[1].values(), dtype=float)])
        val_per_latt=val*2
        first_values=per_lattice*val_per_latt
 
        
        first_values=first_values.round(0)
        
        dif=(first_values/i_n)-val
        score.append(abs(dif).sum())
        
        
        #####hotfix for int_values rounding to numbers which sum to a different number (either +1, or -1)
        
        if abs(first_values.sum()-i_n) > ZERO:
            score[-1]=100000
        
            
        
        
        final_vec.append(first_values)
        
    ii=np.argmin(score)
    
    return np.arange(a_n-limi,a_n+limi+1, DIV)[ii], final_vec[ii],score[ii]

def SQS_size_from_ELE_LIST(around_number, ELE_SUBL, ELEMENTS, comp, range_num, MIN=0):
    '''creates an SQS preferable size based on the sublattice 
    and on the composition'''
    score = []
    final_vec = []
    a_n = around_number
    
    DIV = len(ELE_SUBL)
    
    range_len = DIV*range_num
    if a_n % DIV > ZERO:
        a_n = np.round(a_n/DIV, 0)*DIV
    
    unq_list, inds_ELE, unq_siz = get_unique_list_of_lists(ELE_SUBL, ELEMENTS)
        
    up_comp = comp.copy()
    org_values = np.zeros((len(unq_list), len(ELEMENTS)),dtype=float)
    
#     print(comp)
    
    for si in np.unique(unq_siz):
        for ni, i in enumerate(unq_list): 
            if si == unq_siz[ni]:
                new = np.zeros(comp.shape, dtype=float)
                new[inds_ELE[ni]] = [1.0/len(i)]*len(i)
                comp_subl = new*(unq_siz[ni]/np.sum(unq_siz))
                prov_comp = up_comp-comp_subl
                if any(prov_comp < -ZERO):
                    ret_subl = np.zeros(up_comp.shape)
                    sub_tot_frac = np.sum(comp_subl)
                    for nee in inds_ELE[ni]:
                        if up_comp[nee] > sub_tot_frac:

                            ret_subl[nee] = sub_tot_frac
                        else:
                            ret_subl[nee] = up_comp[nee]
                        comp_subl = ret_subl
                        up_comp = up_comp-comp_subl

                else:
                    up_comp = prov_comp

                org_values[ni] = comp_subl
            else:
                pass
    
    if abs(np.sum(org_values)-1.0) > ZERO:
        print('composition not apt for this sublattice arrangement')
        return
    
    for i_n in np.arange(a_n-range_len, a_n+range_len+1, DIV):
        first_values = org_values*i_n
        
        first_values = first_values.round(0)

        dif = (first_values/i_n).sum(axis=0)-comp
        
        score.append(abs(dif).sum())
        
        if abs(first_values.sum()-i_n) > ZERO:
            score[-1] = 100000
        if i_n < MIN:
            score[-1] = 100000
                    
        final_vec.append(first_values)
        
    ii = np.argmin(score)
    
    return np.arange(a_n-range_len, 
                     a_n+range_len+1, DIV)[ii], final_vec[ii], score[ii]


def create_rndst(size, rndstr_blnk, elements, file_name=''):
    ''' create rndstr for SQS creation'''
    rb=rndstr_blnk
    ac=2*(size[1]/size[0]).round(8)
    r_replace=rb[:]
    r_replace=r_replace.replace('Ni','0lattice').replace('Ti','1lattice')
    r_replace=nth_repl(r_replace,'0lattice','0lattice\n',2)
    r_replace=nth_repl(r_replace,'1lattice','1lattice\n',2)
    for row in range(ac.shape[0]):
        hili=''
        for j,jj in enumerate(elements):
            #print(j,jj)
            if ac[row,j] < ZERO:
                pass
            else:
                hili=hili+jj+'='+str(ac[row,j])+', '
        hili = hili[:-2]
        

        r_replace=r_replace.replace(str(row)+'lattice',hili)
    
    
    if file_name != '':
        with open(file_name, "w") as out_file:
            out_file.write(r_replace)
            
    return r_replace


def create_rndst_subl(size, rndstr_blnk, elements, subl, file_name=''):
    ''' create rndstr for SQS creation'''
    _, __, unq_siz = get_unique_list_of_lists(subl, elements_from_subl(subl))
    rb = rndstr_blnk
    ac = (size[1]/size[0]).round(8)
    ac = ac * np.array(np.sum(unq_siz)/unq_siz)[:, np.newaxis]

    r_replace = rb[:]
    for row in range(ac.shape[0]):
        hili = ''
        for j, jj in enumerate(elements):
            
            if ac[row, j] < ZERO:
                pass
            else:
                hili = hili+jj+'='+str(ac[row, j])+', '
        hili = hili[:-2]

        r_replace = r_replace.replace('SUBL_'+'{:02d}'.format(row), hili)
    
    if file_name != '':
        with open(file_name, "w") as out_file:
            out_file.write(r_replace)
            
    return r_replace


def create_slurm_obj(size,empty_slurm,do,file_name=''):
    '''create ATAT SQS file from the file 'empty_slurm'
    and dictionary 'do' , if 'file_name' is given it writes
    it out to this path'''

    with open(empty_slurm, "r") as in_file:
        
        buf = in_file.read()
        
    buf=buf.replace('ntasks=xx', 'ntasks='+
                    str(do['ntasks'])).replace('ntasks-per-node=xx',
                                               'ntasks-per-node='+str(do['ntasks_per_node']))

    buf=buf.split('\n')
    
    
    
    cmd='corrdump -nop -noe -2=xx-3=xx-4=xx -ro -l=rndstr.in -clus;getclus'.replace('-2=xx','-2='+str(do['clus2'])+' '
                                ).replace('-3=xx','-3='+str(do['clus3'])+' ').replace('-4=xx','-4='+str(do['clus4'])+' ')
    
    cmd=cmd+'\n\
for (( id=0 ; id<'+str(do['ntasks'])+' ; id++ ))\n\
do\n\
mcsqs -n='+str(size[0])+' -wr=10 -wn=0.75 -wd=1 -ip=$id &\n\
done\n\
    '
    for line in buf:
        if line == '#---------------------- job ----------------------------#':
            line = line + '\n' + cmd +"\n"
    
    for i in range(len(buf)):
        if buf[i] == '#---------------------- job ----------------------------#':
            buf[i] = buf[i] + '\n' + cmd +"\n"
            

    buf=[i.split('\n') for i in buf]
    buf=[item for sublist in buf for item in sublist]
    
    buf='\n'.join(buf)
    
    if file_name != '':
        with open(file_name, "w") as out_file:
            out_file.write(buf)

    return(buf)










'''OBJECT'''






class SLURM_job:
    
    def __init__(self,dir_name,updated='slurm_u.job'):
        
        self.user=self.get_user()
        
        
        self.notebook_dir=os.getcwd()
        self.dir_name_DN=dir_name
        self.working_dir=self.notebook_dir+'/'+dir_name
        self.file_log=self.working_dir+'/LOG_SLURM_PY'
        
        self.json_FN=self.working_dir+'/obj.json'
        
        self.updated_SLURMfile=self.working_dir+'/'+updated
        
        
        
        ### if no log, create
        if not os.path.exists(self.file_log):
            here_dt=self.datetime()
            
            print('SLURM LOG created for directory '+self.working_dir+' at: '+here_dt+'\n')
            self.write_log('SLURM LOG created for directory '+self.working_dir+' at: '+here_dt+'\n')
            
            self.status='CREATED'
                
            self.dump_object()
                
        
        ### if log, load json
        else:
            
            self.write_log('\nSLURM LOG accessed at'+self.datetime()+'\n')
            
            #### totally new object created, everything must be pulled from the LOG
            
            if os.path.exists(self.json_FN):
                self.load_object()
                
                
            else:
                print('log available but no json object')
                self.write_log('log available but no json object')
                
            work_str=self.refresh_log()
            
            if hasattr(self, 'ID'):
                self.write_log('id found in LOG')
                self.squeue_check()
                self.dump_object()
            else:    
                self.status='CREATED'
                self.write_log('SLURM job created but not yet sent at: '+self.datetime())
    
    
    def get_user(self):
        echo_arg = os.environ['USER']
        res = subprocess.Popen(["echo", echo_arg], stdout=subprocess.PIPE).args[1]
        return res
        
    def both_prints(self,string_print):
        print(string_print)
        self.write_log(string_print)
    
    
    def squeue_check(self):
        cmd = ['squeue', '-h', '-u', self.user, '-j', str(self.ID)]
        squeue_output=subprocess.run(cmd,capture_output=True)
        
        if squeue_output.stdout==b'':
            print('ERROR in squeue check: '+squeue_output.stderr.decode("utf-8") )
            self.write_log('ERROR in squeue check: '+squeue_output.stderr.decode("utf-8") )
            self.status = 'DONE'
            self.jobo=self.read_file(self.working_dir+'/job.o'+str(self.ID))
            self.jobe=self.read_file(self.working_dir+'/job.e'+str(self.ID))
            self.jobt=self.read_file(self.working_dir+'/job.t'+str(self.ID))
            # self.both_prints('job output:')
            # self.both_prints(self.jobo)
            # self.both_prints('job error:')
            # self.both_prints(self.jobe)
            # self.both_prints('job text:')
            # self.both_prints(self.jobt)
            
            
            self.both_prints('This job has finished already')
            
            
            # self.dump_object()
            #self.write_log('This job has finished already')
        else:
            
            self.write_log('squeue_output: '+squeue_output.stdout.decode("utf-8"))
            
            
            out_work=squeue_output.stdout.split()
            print(out_work)       
            if len(out_work)>6:
                #self.ID already taken
                self.job_name=  out_work[1].decode("utf-8")
                #self.user
                self.partition= out_work[3].decode("utf-8")
                self.nodes=     out_work[4].decode("utf-8")
                self.tasks=     out_work[5].decode("utf-8")
                self.status=    out_work[6].decode("utf-8")
                self.time =     out_work[7].decode("utf-8")
                self.time_left= out_work[8].decode("utf-8")
                self.start_time=out_work[9].decode("utf-8")
            
        
            
            
        if self.status=='RUNNING':
            self.write_log('at '+self.datetime()+' :'+'this job is been running for '+self.time+ ' since '+self.start_time+' and there is '+
                      self.time_left+ ' time left' )
        #self.write_log('object DUMP due to squeue check')
        #self.dump_object()
        #dump in squeue or not, that's the question
        
        
        
    def datetime(self):
        return str(datetime.now().isoformat())
    
    def refresh_log(self):
        with open(self.file_log, "r") as in_file:
            r=in_file.read()
        return r
            
                
                
    def write_log(self,string_to_write):
        with open(self.file_log, "a+") as in_file:
                in_file.write(string_to_write+'\n')
            
    
    def from_file(self, s_file):
        
        '''this function is supposed to load everything 
        and put it into a dictionary of SLURM related
        variables (time, nodes, ntasks, ntasks-per-node), 
        for now we are happy by getting the file name'''
        
        self.original_SLURMfile=s_file
        self.write_log('SLURM FILE (ONLY {for now}) loaded from '+self.notebook_dir+'/'
                              +self.original_SLURMfile+' at '
                              +self.datetime())
        # print(s_file)
        
        f= open(self.original_SLURMfile,'r')
        self.SLURMfile_as_string=f.read()
            
        # print('######file#####\n',self.SLURMfile_as_string)
        
        
        self.dump_object()
        
    def read_file(self,file):
        f=open(file,'r')
        txt_return = f.read()
        f.close()
        return txt_return
        
        
        
    def dump_SLURMfile(self):
        # print(self.SLURMfile_as_string)
        self.write_log('SLURM file written to: '+self.working_dir+'/'+self.updated_SLURMfile)
        with open(self.updated_SLURMfile,'w') as f:
            f.write(self.SLURMfile_as_string)
        
    
    def execute(self):
        self.both_prints('EXECUTE : ')
        if hasattr(self, 'ID'):
            pass
            self.squeue_check()
            if self.status=='RUNNING' or 'PENDING':
                
                self.both_prints('Job running at the moment, please cancel the job or wait for its culmination')
                self.both_prints('object is running with an ID : '+str(self.ID))
                
            else:
                self.both_prints('Job has likely finished ######')
                
        else:
            self.both_prints('no attribute so far we proceed to run a first try...')
            
            self.send_job()
            self.dump_object()
            
            
            
        
    def send_job(self):
        self.dump_SLURMfile()
        self.dump_object()
        send_str=['sbatch',self.updated_SLURMfile]
        
        os.chdir(self.working_dir)
        self.both_prints('submitting...')
        try:

            send_out=subprocess.run(send_str,capture_output=True)
            
            
            out_work=send_out.stdout.decode("utf-8").split()
            
            
            dummytime=self.datetime()
            
            self.both_prints('output(if any): '+send_out.stdout.decode("utf-8")+' at '+dummytime)
            self.both_prints('error(if any): '+send_out.stderr.decode("utf-8")+' at '+dummytime)
            
            
            
            self.ID=int(out_work[-1])
            
            
            
            self.both_prints('SLURM submitted with and id of: '+str(self.ID)+' at '+dummytime)
            
            
            self.both_prints('ID : '+str(self.ID))
            
            
            
            
            
            os.chdir(self.notebook_dir)
            
            
            self.status='SUBMITTED'
            
            self.both_prints('job try succesful')
            
            
        except Exception as e: 
            
            self.both_prints('python error: '+repr(e))
            self.both_prints('job try unsuccesful')
            
            os.chdir(self.notebook_dir)
            pass




    #### same logic
    def cancel(self):
        self.both_prints('CANCEL : ')
        if hasattr(self, 'ID'):
            pass
            self.squeue_check()
            if self.status=='RUNNING' or 'PENDING':
                self.both_prints('Job running at the moment, proceeding to cancel ')
                self.both_prints('object is running with an ID : '+str(self.ID))
                self.cancel_job()
                self.dump_object()


            else:
                print('Job has likely finished ######')
                
        else:
            self.both_prints('Job has no ID, nothing to cancel')



    def cancel_job(self):
        self.dump_object()
        send_str=['scancel',str(self.ID)]
        
        os.chdir(self.working_dir)
        try:

            send_out=subprocess.run(send_str,capture_output=True)
            
            dummytime=self.datetime()
            self.both_prints('canceled job '+str(self.ID)+' at '+dummytime)
            self.both_prints('output: '+send_out.stdout.decode("utf-8"))
            time.sleep(1)
            
            
            
            
            os.chdir(self.notebook_dir)
            
            


            self.status='CANCELED'
            
            self.both_prints('job cancel succesful')
            

            
            
        except Exception as e: 
            
            print(e)
            self.both_prints('job cancel unsuccesful')
            os.chdir(self.notebook_dir)
            pass



    def dump_object(self):
        
        with open(self.json_FN, "w") as fp:
            json.dump(self.__dict__,fp)
        self.write_log('JSON object dumped')
        self.write_log('object attributes DUMPED TO '+self.json_FN+' at '+self.datetime())
        
    def load_object(self):

        with open(self.json_FN, "r") as fp:
            js=json.load(fp)
        for key in js:
            setattr(self, key, js[key])
        
        self.write_log('JSON object loaded')
        self.write_log('object attributes LOADED FROM '+self.json_FN+' at '+self.datetime())
        
        
        
    def freeze_files(self,file_in=[],file_start=[],file_end=[]):
        dummytime=self.datetime()
        
        self.freeze_DN=self.working_dir+'/'+self.dir_name_DN+dummytime
        
        folder_or_delete(self.freeze_DN)
        
        
        
        
        
        self.squeue_check()
        
        
        for root, dirs, files in os.walk(self.working_dir, topdown=False):
            
        
            files.sort()
            
            for inin in file_in:
                work_list=[i for i in files if inin in i]
                for i in work_list:
                    shutil.copyfile(self.working_dir+'/'+i, self.freeze_DN+'/'+i)
        if self.status == 'RUNNING':
            with open(self.freeze_DN+'/'+'README','w') as f:
                f.write(
            'copy of SLURM job '+str(self.ID)+' at '+dummytime+'\n'+
                       'files containing '+str(file_in)+' are copied to the directory '+self.freeze_DN+'\n'+
                       'about '+self.time+' into the job, '+self.time_left+' remaining\n\n'
                       )
            self.both_prints('\n\ninto README : \n\n'+
            'copy of SLURM job '+str(self.ID)+' at '+dummytime+'\n'+
                       'files containing '+str(file_in)+' are copied to the directory '+self.freeze_DN+'\n'+
                       'about '+self.time+' into the job, '+self.time_left+' remaining\n\n'
                       )
            
        elif self.status == 'DONE':
            with open(self.freeze_DN+'/'+'README','w') as f:
                f.write(
        'copy of SLURM job '+str(self.ID)+' at '+dummytime+'\n'+
                   'files containing '+str(file_in)+' are copied to the directory '+self.freeze_DN+'\n'+
                   'after the job has completed '+'\n\n'
                   )
            self.both_prints('\n\ninto README : \n\n'+
        'copy of SLURM job '+str(self.ID)+' at '+dummytime+'\n'+
                   'files containing '+str(file_in)+' are copied to the directory '+self.freeze_DN+'\n'+
                   'after the job has completed '+'\n\n'
                   )
            
        return self.freeze_DN

''' read correlations from SQSs'''

def read_corr(f_n):
    with open(f_n, "r") as in_file:
        buf = in_file.read()
    buf = buf.split('\n')
    
    obj=float(buf[-2].split()[-1])
    
    buf = [i.split('\t') for i in buf]

    buf = [i for i in buf if len(i)>2]
    buf = np.asarray(buf, dtype=float)
    
    return buf,obj


def load_correlations(directory_cor):
    res=[]
    dc=directory_cor
    
    for root, dirs, files in os.walk(dc, topdown=False):
        print(root)
        files.sort()
        work_list=[root+'/'+i for i in files if 'bestcorr' in i]
        
        
        for i in work_list:
            res.append(read_corr(i))
            
    return res




'''read SQSs functions same as corrs, but we have an extra step to 
load an sqs into an ase'''
def read_sqs(f_n):
    with open(f_n, "r") as in_file:
        buf = in_file.read()
        buf = buf.split('\n')
    atoms = [i.split(' ')[-1] for i in buf[6:]]
    atoms = [i for i in atoms if len(i)>0]
    l_v = [i.split() for i in buf[:6]]
    l_v = np.asarray(l_v, dtype=float)
    b_v = [i.split()[:3] for i in buf[6:6+len(atoms)]]
    b_v = np.asarray(b_v, dtype=float)
    v1=l_v[:3]
    v2=l_v[3:6]
    l_v=np.matmul(v2,v1)
    b_v=np.matmul(b_v,v1)
    return l_v, b_v, atoms


def load_sqs(directory_cor):
    res=[]
    dc=directory_cor
    
    for root, dirs, files in os.walk(dc, topdown=False):
        files.sort()
        work_list=[i for i in files if 'bestsqs' in i]
        
        
        for i in work_list:
            res.append(read_sqs(dc.split('/')[-1]+'/'+i))
            
    return res

def load_sqs_into_ase(directory_cor,scaling_factor):
    res_ase=[]
    files=[]
    dc=directory_cor
    sf=scaling_factor
    for root, dirs, files in os.walk(dc, topdown=False):
        files.sort()
        work_list=[root+'/'+i for i in files if 'bestsqs' in i]
        
        
        for i in work_list:
            res=read_sqs(i)
            files.append(i)
            res_ase.append(create_ase(sf*res[0],sf*res[1],res[2]))
    return res_ase, files





def best_sqs(ct):
    corrs_suma=abs(np.array(ct)[:,:,:,-1]).sum(axis=2)
    return np.unravel_index(np.argmin(corrs_suma, axis=None), corrs_suma.shape)


def get_abc_abg_DEG(cell):
    a, b, c = np.linalg.norm(cell[0]), \
        np.linalg.norm(cell[1]), np.linalg.norm(cell[2])
    an, bn, cn = cell[0]/np.linalg.norm(cell[0]), \
        cell[1]/np.linalg.norm(cell[1]), cell[2]/np.linalg.norm(cell[2])
    return a, b, c, \
        np.rad2deg(np.arccos(
            np.dot(bn, cn)/np.linalg.norm(bn)*np.linalg.norm(cn))), \
        np.rad2deg(np.arccos(
            np.dot(an, cn)/np.linalg.norm(an)*np.linalg.norm(cn))), \
        np.rad2deg(np.arccos(
            np.dot(an, bn)/np.linalg.norm(an)*np.linalg.norm(bn)))

        
def get_abc_abg_RAD(cell):
    a,b,c=np.linalg.norm(cell[0]),np.linalg.norm(cell[1]),np.linalg.norm(cell[2])
    return a,b,c,\
        np.arccos(np.dot(cell[1],cell[2])/b*c),\
        np.arccos(np.dot(cell[0],cell[2])/a*c),\
        np.arccos(np.dot(cell[0],cell[1])/a*b)




def get_normal_from_miller(hkl,cell):
    isect=[]
    save=[]
    for i,ii in enumerate(hkl):
        if ii<1E-8:
            save.append(i)
            continue
        else:
            isect.append(i)
    if len(isect) == 1:
        v1,v2=cell[save[0]],cell[save[1]]
        return np.array(cell[isect[0]])/np.linalg.norm(np.array(cell[isect[0]])),np.array([v1,v2])
    elif len(isect) == 2:
        p1=cell[isect[0]]/hkl[isect[0]]
        p2=cell[isect[1]]/hkl[isect[1]]
        v1=p2-p1
        v2=cell[save[0]]
        return np.cross(v1,v2)/np.linalg.norm(np.cross(v1,v2)),np.array([v1,v2])
    else:
        v1,v2=(cell[0]/hkl[0])-(cell[2]/hkl[2]),(cell[1]/hkl[1])-(cell[2]/hkl[2])

        return np.cross(v1,v2)/np.linalg.norm(np.cross(v1,v2)),np.array([v1,v2])


def fix_shuffle(matr):
    '''FIX matrix shuffle to be the number with the lesser magnitude'''
    matr = matr.copy()
    if len(matr.shape)==2:
        ni,nj= matr.shape
        for i in range(ni):
            for j in range(nj):
                if 1 - abs(matr[i,j]) < abs(matr[i,j]):
                    if matr[i,j] < 0.0:
                        matr[i,j]= matr[i,j]%1
                    else:
                        matr[i,j]= matr[i,j] - 1.0
                else:
                    continue

        return matr

def get_index_trans(SC_pos,PC_sca_pos,PC_cell):
    index_trans = []
    PC_sca_pos=PC_sca_pos%1
    PC_sca_pos=PC_sca_pos%(1-ZERO)
    for vec in SC_pos:
        
        work_vec=np.linalg.solve(PC_cell.T,vec)
        work_vec=work_vec%1.0
        work_vec=work_vec%(1-ZERO)
        tolerance=0.01
        index_trans.append(np.where(np.all(np.isclose(
            work_vec, PC_sca_pos, rtol=tolerance, atol=tolerance), axis=1))[0][0])
    return index_trans
import numpy as np

def angle_between_vectors(v1, v2):
    # Calculate the dot product
    dot_product = np.dot(v1, v2)
    
    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(v1)
    magnitude2 = np.linalg.norm(v2)
    
    # Calculate the cosine of the angle
    cosine_angle = dot_product / (magnitude1 * magnitude2)
    
    # Calculate the angle in radians
    angle_radians = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    # Convert radians to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees


def ortho_proj_dis(points,vec1,vec2,cell,dis_point=np.array([0,0,0]),**kwargs):
    n=np.cross(vec1,vec2)
    
    v_d = (np.dot((points-dis_point),n)/(np.linalg.norm(n)**2))
    dists = np.linalg.norm(np.array([i*n for i in v_d]),axis=1)
    
    if 'angle_to_sign' in kwargs:
        if kwargs['angle_to_sign']:
            points_a = points-dis_point
            angles = [angle_between_vectors(n, i) for i in points_a]
            new_dists=[]
            for ii,jj in zip(angles,dists):
                if ii >=90.0:
                    new_dists.append(jj)
                else:
                    new_dists.append(jj*-1)
    dists=np.array(new_dists)
    v=(np.dot(points,n)/(np.linalg.norm(n)**2))
    res=points-np.array([i*n for i in v])
    new_b=np.array([vec1,vec2])
    res=new_b@res.T
    for i,_ in enumerate(res):
        res[i]= res[i]/np.linalg.norm(new_b[i])
    v_c =(np.dot( cell ,n)/(np.linalg.norm(n)**2))
    res_c =cell-np.array([i*n for i in v_c ])
    res_c =new_b@res_c.T
    for i,_ in enumerate(res_c):
        res_c[i]= res_c[i]/np.linalg.norm(new_b[i])
    if 'bigcell' in kwargs:
        bigcell=kwargs['bigcell']

        v_b =(np.dot( bigcell ,n)/(np.linalg.norm(n)**2))
        res_b =bigcell-np.array([i*n for i in v_b ])
        res_b =new_b@res_b.T
        for i,_ in enumerate(res_b):
            res_b[i]= res_b[i]/np.linalg.norm(new_b[i])

        return res.T,res_c.T,dists,res_b.T
    return res.T,res_c.T,dists


def unique_values_with_tolerance(arr, tolerance):
    unique_vals = []
    for val in arr:
        is_unique = all(abs(val - u) > tolerance for u in unique_vals)
        if is_unique:
            unique_vals.append(val)
    return unique_vals

def count_equal_elements_for_each(arr1, arr2, tolerance):
    # Convert input lists to numpy arrays
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    
    # Initialize a list to store counts
    counts = []
    
    # Iterate over each element in arr1
    for val1 in arr1:
        # Use broadcasting and comparison with tolerance
        equal_elements = np.abs(arr2 - val1) <= tolerance
        count = np.sum(equal_elements)
        counts.append(count)
    
    return counts

def plot_mil_BC_SC_dis_plus(BC,SC,MIL,PLUS=True):
    FS=15
    
    
    cell = SC.get_cell().array
    pos =  BC.get_positions()
    BC_color= [vesta_colors[i] for i in BC.get_chemical_symbols()]
    BC_size = np.array([atomic_radii[i] for i in BC.get_chemical_symbols()])


    normal,[vec1,vec2]=get_normal_from_miller(MIL,cell)
    
    opro,_2d_c,dists = ortho_proj_dis(pos,vec1,vec2,cell,angle_to_sign=True)
    

    

    u_dists=unique_values_with_tolerance(dists,1E-4)
    
    u_dists.sort()
    
    ind_pm = np.argmax(count_equal_elements_for_each(u_dists,dists,1E-8))
    
    ind_plane=np.where(np.abs(dists-u_dists[ind_pm])<ZERO)[0]
    
    
    plt.figure(figsize=(8, 8))
    
    
    plot_x = opro[ind_plane,0]
    plot_y = opro[ind_plane,1]

    co = np.array(BC_color)[ind_plane]
    si = np.array(BC_size)[ind_plane]
    fig, ax = plt.subplots(figsize=(8,8))
    if len(plot_x)>0:
        
        x_,y_=[np.max(plot_x)-np.min(plot_x),np.max(plot_y)-np.min(plot_y)]    

       
        if PLUS==False:
            ax.scatter(plot_x,plot_y,c=co,s=si*2,edgecolors='k',alpha=0.2)

            for i in range(len(opro)):
                if i in ind_plane:
                    ax.annotate(str(i), (opro[i,0], opro[i,1]),fontsize=15)
            
        
        max_axis=np.argmax([x_,y_])

        if max_axis>0:
            yls=[np.min(plot_y)-np.sum(_2d_c[:,1]),np.max(plot_y)+np.sum(_2d_c[:,1])]
            ax.set_ylim(yls)

            ax.set_xlim([np.min(plot_x)-(abs(yls[1]-yls[0])-x_)/2, np.max(plot_x)+(abs(yls[1]-yls[0])-x_)/2])
        else:
            xls=[np.min(plot_x)-np.sum(_2d_c[:,0]),np.max(plot_x)+np.sum(_2d_c[:,0])]
            ax.set_xlim(xls)

            ax.set_ylim([np.min(plot_y)-(abs(xls[1]-xls[0])-y_)/2, np.max(plot_y)+(abs(xls[1]-xls[0])-y_)/2])
        
        
        
    if PLUS==False:    
        ax.plot([0,_2d_c[0,0]],[0,_2d_c[0,1]],'--r')
        ax.plot([0,_2d_c[1,0]],[0,_2d_c[1,1]],'--g')
        ax.plot([0,_2d_c[2,0]],[0,_2d_c[2,1]],'--b')
    
    

    
    ######################## sBC ##########################
    if PLUS==True:
        sBC=make_supercell(BC.copy(), np.identity(3)*3)


        sBC_color= [vesta_colors[i] for i in sBC.get_chemical_symbols()]
        sBC_size = [atomic_radii[i] for i in sBC.get_chemical_symbols()]


        sindex = get_index_trans(sBC.get_positions().copy(),BC.get_scaled_positions().copy(),BC.get_cell().array.copy())



        spos =  sBC.get_positions()

        sopro,s_2d_c,sdists,b_2d_c = ortho_proj_dis(
            spos,vec1,vec2,cell,bigcell=BC.get_cell().array.copy(),
            angle_to_sign=True)
        
        u_dists=unique_values_with_tolerance(sdists,1E-4)
        u_dists.sort()
        
        

        ind_pm = np.argmax(count_equal_elements_for_each(u_dists,dists,1E-4))

        sind_plane=np.where(np.abs(sdists-u_dists[ind_pm])<ZERO)[0]
        splot_x = sopro[sind_plane,0]

        splot_y = sopro[sind_plane,1]
        sco = np.array(sBC_color)[sind_plane]
        ssi = np.array(sBC_size)[sind_plane]




        if len(splot_x)>0:
            ax.scatter(splot_x,splot_y,c=sco,s=ssi*2,edgecolors='k',alpha=0.2)

            for i in range(len(sopro)):
                if i in sind_plane:
                    ax.annotate(str(sindex[i]), (sopro[i,0], sopro[i,1]),fontsize=15)



        xls=ax.get_xlim()
        yls=ax.get_ylim()

        
        
        
        ini=7
        fin=8
        hpr=1
        
        added=np.sum(b_2d_c,axis=0)*hpr
        fin_added=_2d_c+added
    
        ax.plot([added[0],fin_added[0,0]],[added[1],fin_added[0,1]],'--r')
        ax.plot([added[0],fin_added[1,0]],[added[1],fin_added[1,1]],'--g')
        ax.plot([added[0],fin_added[2,0]],[added[1],fin_added[2,1]],'--b') 
    
        ax.set_xlim(xls+added[0])
        ax.set_ylim(yls+added[1])
        
    plt.show()





