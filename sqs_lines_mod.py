
import os
import numpy as np
import re
import pandas as pd
from sqsfunc_mod import create_ase

from ase.build.supercells import make_supercell

from pymatgen.io import ase as pg_ase
ase_to_pmg = pg_ase.AseAtomsAdaptor.get_structure
pmg_to_ase = pg_ase.AseAtomsAdaptor.get_atoms


def read_atat_rndstr(f_n):
    '''only works for explicit vectors and not abc nomenclature,
    rndstr changes in that sublattices are given by specific concentrations'''
    with open(f_n, "r") as in_file:
        buf = in_file.read()
        buf = buf.split('\n')
    atoms = [i.split(' ')[3:] for i in buf[6:]]
    pattern = []
    count_subl = 0
    subl_num = 0
    for i in atoms:

        if len(i) == 0:
            if count_subl != 0:
                pattern.extend([subl_num]*count_subl)
                count_subl = 0
                subl_num += 1
        else:
            count_subl += 1
    if count_subl != 0:
        pattern.extend([subl_num]*count_subl)
    long_atoms = [i for i in atoms if len(i) > 0]
    
    l_v = [i.split() for i in buf[:6]]
    l_v = np.asarray(l_v, dtype=float)
    b_v = [i.split()[:3] for i in buf[6:]]
    b_v = [x for x in b_v if x]
    b_v = np.asarray(b_v, dtype=float)
    
    v1 = l_v[:3]
    v2 = l_v[3:6]
    return v1, v2, b_v, long_atoms, pattern


class rndstr_structure:
    def __init__(self, file_name):
        self.from_file(file_name)
        self.differentiate_subl()
        self.ase()

    def from_file(self, file_name):
        self.file_N = file_name
        with open(self.file_N, 'r') as f:
            self.rndstr_FC = f.read()
        self.scaling, self.lattice_vectors_scaled, self.basis_vectors_scaled, \
            self.atom_strings_subl, self.pattern_subl = read_atat_rndstr(
                self.file_N)
            
        self.num_atoms = len(self.atom_strings_subl)
        
        self.lattice_vectors = np.matmul(self.lattice_vectors_scaled,
                                         self.scaling)        
        self.basis_vectors = np.matmul(self.basis_vectors_scaled,
                                       self.scaling)

    def differentiate_subl(self):

        atoms_of_site = [[] for i in range(self.num_atoms)]
        comps_of_site = [[] for i in range(self.num_atoms)]
        for c_i, i in enumerate(self.atom_strings_subl):
            for j in i:
                
                atoms_of_site[c_i].append(j.split('=')[0])
                comps_of_site[c_i].append(
                    float(j.split('=')[1].replace(',', '')))
        all_atoms = list(set(
            [item for sublist in atoms_of_site for item in sublist]))
        
        all_atoms = sorted(all_atoms)
        
        num_diff_atoms = len(all_atoms)
        comp_final = np.zeros((self.num_atoms, num_diff_atoms), dtype=float)
        for i_c, i in enumerate(comps_of_site):
            for j_c, j in enumerate(i):
                comp_final[i_c, all_atoms.index(atoms_of_site[i_c][j_c])] = j
        self.all_atoms = all_atoms
        self.df_comp = pd.DataFrame(columns=all_atoms, data=comp_final)
        self.comp_final = comp_final
        self.num_diff_atoms = num_diff_atoms
        self.atom_per_site = atoms_of_site
        self.comps_per_site = comps_of_site
        self.df_full = self.rnds_add_unique()

    def rnds_add_unique(self):
        df = self.df_comp.copy()
        df['subl'] = None
        df['unique'] = None
        for i, _ in df.iterrows():
            
            df.at[i, 'subl'] = 'SUBL_{:02d}'.format(self.pattern_subl[i])
            df.at[i, 'unique'] = self.pattern_subl[i]  
        return df
        
    def ase(self):
        self.ase_struct = create_ase(self.lattice_vectors, 
                                     self.basis_vectors,
                                     self.df_full['unique'])
    
    def __repr__(self):
        #  json doesn't look that nice
        #  r = json.dumps(self.__dict__,indent=2,cls=NpEncoder) 
        string_repr = ''
        for key, value in self.__dict__.items():
            string_repr = string_repr + str(key)+'    :\n' + str(value) + '\n'

        return string_repr

# read a composition directory 


def read_atat_sqs(f_n):
    '''only works for explicit vectors and not abc nomenclature,
    rndstr changes in that sublattices are given by specific concentrations'''
    with open(f_n, "r") as in_file:
        buf = in_file.read()
        buf = buf.split('\n')
    
    atoms = [i.split(' ')[3:] for i in buf[6:]]
    
    long_atoms = [i for i in atoms if len(i) > 0]
    
    long_atoms = [item for sublist in long_atoms for item in sublist]
    
    l_v = [i.split() for i in buf[:6]]
    l_v = np.asarray(l_v, dtype=float)
    b_v = [i.split()[:3] for i in buf[6:]]
    b_v = [x for x in b_v if x]
    b_v = np.asarray(b_v, dtype=float)
    
    v1 = l_v[:3]
    v2 = l_v[3:6]
    return v1, v2, b_v, long_atoms


class supercell_structure:
    def __init__(self, file_name):
        self.from_file(file_name)
        self.ase()

    def from_file(self, file_name):
        self.file_N = file_name
        with open(self.file_N, 'r') as f:
            self.rndstr_FC = f.read()
        self.scaling, self.lattice_vectors_scaled, self.basis_vectors_scaled, \
            self.atoms = read_atat_sqs(self.file_N)
            
        self.num_atoms = len(self.atoms)
        
        self.lattice_vectors = np.matmul(self.lattice_vectors_scaled, 
                                         self.scaling)        
        self.basis_vectors = np.matmul(self.basis_vectors_scaled, self.scaling)
   
    def ase(self):
        self.ase_struct = create_ase(self.lattice_vectors, 
                                     self.basis_vectors,
                                     self.atoms)
    
    def __repr__(self):
        
        string_repr = ''
        for key, value in self.__dict__.items():
            string_repr = string_repr + str(key)+'    :\n' + str(value) + '\n'

        return string_repr


class sqs_one:
    '''SQS structure from a file searches for the rndstr.in in-dir'''
    def __init__(self, SQS_file_N, in_dir=True):
        self.SQS_file_N = SQS_file_N
        
        if in_dir:
            self.SQS_file_PATH = os.path.abspath(SQS_file_N)

            self.SQS_dir_PATH = '/'.join(self.SQS_file_PATH.split('/')[:-1])
            self.rndstr_PATH = self.SQS_dir_PATH+'/rndstr.in'
            self.digit_reference = re.sub('\D', '',
                                          self.SQS_file_PATH.split('/')[-1])
            
            self.clusters_PATH = self.SQS_dir_PATH+'/clusters.out'
            
            self.corr_PATH = self.SQS_dir_PATH+'/bestcorr-' + \
                self.digit_reference+'.out'
            
        else:
            pass
        time_start = os.path.getmtime(self.clusters_PATH)
        time_current = os.path.getmtime(self.corr_PATH)
        self.total_time = time_current-time_start
        
        self.sup_obj = supercell_structure(self.SQS_file_N)
        self.rnd_obj = rndstr_structure(self.rndstr_PATH)
        
        self.get_mult_matrix()
        self.create_periodic_supercell()
        self.read_cluster()
        self.read_corr()
        
    def __repr__(self):
        
        string_repr = ''
        for key, value in self.__dict__.items():
            string_repr = string_repr + str(key)+'    :\n' + str(value) + '\n'

        return string_repr
    
    def get_mult_matrix(self):
        self.mult_matrix = (self.sup_obj.lattice_vectors@np.linalg.inv(
            self.rnd_obj.lattice_vectors)).round(2)
        
    def read_sqscellout(self):
        self.sqscellout_PATH = self.SQS_dir_PATH+'/sqscell.out'
        with open(self.sqscellout_PATH, "r") as in_file:
            buf = in_file.read()
            buf = buf.split('\n')

        buf = [i for i in buf if len(i) > 0]
        
        self.number_of_supercell = int(buf[0])
        buf = buf[1:]
        
        self.supercells = np.array([float(item) 
                                   for sublist in [i.split() for i in buf] 
                                   for item in sublist]).reshape(
            self.number_of_supercell, 3, 3)
        
    def create_periodic_supercell(self):
        self.per_sup = make_supercell(self.rnd_obj.ase_struct, 
                                      self.mult_matrix)
        
    def read_cluster(self):
        
        with open(self.clusters_PATH, "r") as in_file:
            buf = in_file.read()
            buf = buf.split('\n')
        buf = [i for i in buf if len(i) > 0]
        #  [x for b in a for x in b]
        #  [word for sentence in text for word in sentence]
        #  buf=[float(item) for i in buf for item in i.split()]
        mult, diam, point, coors, spe2, clusf = [], [], [], [], [], []
        ii = 0
        while ii < len(buf):
            mult.append(float(buf[ii]))
            ii = ii+1
            diam.append(float(buf[ii]))
            ii = ii+1
            point.append(float(buf[ii]))
            ii = ii+1
            
            arr_dum = np.array([float(j) 
                                for row in [i.split() 
                                for i in buf[ii:ii+int(point[-1])]] 
                                for j in row]).reshape(
                int(point[-1]), 5)
            
            coors.append(arr_dum[:, :3])
            spe2.append(arr_dum[:, 3])
            clusf.append(arr_dum[:, 4])
            
            ii = int(ii+point[-1])
        self.mult, self.diam, self.point, self.coors, self.spe2, self.clusf = \
            mult, diam, point, coors, spe2, clusf
        #  [length of the longest pair within the cluster]
        #  [number of points in cluster]
        #  [coordinates of 1st point] [number of possible species-2]
        #  [cluster function]
    
    def read_corr(self):
        
        with open(self.corr_PATH, "r") as in_file:
            buf = in_file.read()
            buf = buf.split('\n')
        buf = [i.split('\t') for i in buf if len(i) > 0]
        
        self.obj_fun = float(buf[-1][0].split()[-1])
        
        buf = buf[:-1]
        
        buf = [float(j) for i in buf for j in i]
        buf = np.array(buf).reshape(int(len(buf)/5), 5)
        self.c_point = buf[:, 0]  # its number of pointd (1st column)
        self.c_diam = buf[:, 1]  # its diameter (2nd column)
        self.c_best = buf[:, 2]  # the correlations of the best SQS found
        #                         so far (3rd column)
        self.c_diso = buf[:, 3]  # along with the target disordered state
        #                          correlation (4th column)
        self.c_diff = buf[:, 4]  # and the difference between
        #                          the two (5th column)


def create_sqs_line(pwd_dir, dig):
    #  file.startswith('bestsqs') and 
    fns = []
    for file in os.listdir(pwd_dir):
        filename = os.fsdecode(file)
        if filename.startswith('bestsqs') and filename.endswith(
                str(dig)+'.out'):
            fns.append(filename)
    num_sqs = len(fns)
    sqs_line = []
    for i in range(num_sqs):
        sqs_line.append(sqs_one(pwd_dir+'bestsqs-'+str(i+1)+str(dig)+'.out'))
    return sqs_line


def get_min_arr_of_arr(aoa):
    res = []
    for i in aoa:
        res.append([np.min(i), np.argmin(i)])
    res = np.array(res)
    return (np.argmin(res[:, 0]), int(res[np.argmin(res[:, 0]), 1]))
