import numpy as np

from sympy.ntheory import factorint
from sympy import Matrix, ZZ
from sympy.matrices.normalforms import smith_normal_form as get_SNF

from ase import Atoms
from ase.build.supercells import make_supercell


from icet.tools import enumerate_structures

from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from itertools import combinations

import time



from pymatgen.io import ase as pg_ase

ase_to_pmg = pg_ase.AseAtomsAdaptor.get_structure
pmg_to_ase = pg_ase.AseAtomsAdaptor.get_atoms


def flatten(xss):
    return [x for xs in xss for x in xs]


def _no_HNF(MOD):
    dic_f = factorint(MOD)
    prod = 1
    for i, v in dic_f.items():
        
        prod = prod*(i**(v+2)-1)*(i**(v+1)-1)*(i-1)**(-2)*(i+1)**(-1)
        
    return int(round(prod, 5))


def _get_HNF(MOD):
    HNFs = np.empty((_no_HNF(MOD), 3, 3), dtype=int)
    ci = 0
    for a in range(1, MOD + 1):
        if MOD % a == 0.0:

            nMOD = int(round(MOD/a, 2))
            
            for c in range(1, nMOD + 1):
                if nMOD % c == 0.0:
                    f = int(round(MOD/(a*c), 2))
                    
                    for b in range(c):
                        for d in range(f):
                            for e in range(f):
                                HNFs[ci] = np.array([[a, 0, 0],
                                                    [b, c, 0],
                                                    [d, e, f]]).T
                                ci = ci + 1
    return HNFs


def return_HNF_n_SNF(nH, prim_cell, rot_mats):
    HNFs = _get_HNF(nH)
    
    Us = []
    UHNFs = []
    _SNFs = []
    
    Us.append(HNFs[0]@prim_cell)
    UHNFs.append(HNFs[0])
    for i, Hi in enumerate(HNFs):
        unique = True
        Bi = Hi@prim_cell
        for ui, u in enumerate(Us):
            work = (Bi@np.linalg.inv(rot_mats)@np.linalg.inv(u)).round(2) % 1.0
            work2 = ~np.apply_along_axis(np.any, 1, work)
            if np.apply_along_axis(np.all, 1, work2).any():
                unique = False
                break
            else:
                pass
        if unique is True:
            Us.append(Bi)
            UHNFs.append(Hi)
    for uh in UHNFs:
        _SNFs.append(
            np.diagonal(np.abs(np.array(get_SNF(Matrix(uh), domain=ZZ)).astype(
                np.float64)))
        )
    return UHNFs, _SNFs


def retMembers(iDiago):
    miembro = np.array([[0]*np.prod(iDiago)]*3)
    for i in range(1, np.prod(iDiago)):
        miembro[:, i] = miembro[:, i-1]
        miembro[2, i] = (miembro[2, i-1]+1) % iDiago[2]
        if miembro[2, i] == 0:
            miembro[1, i] = (miembro[1, i-1] + 1) % iDiago[1]
            if miembro[1, i] == 0:
                miembro[0, i] = (miembro[0, i-1] + 1) % iDiago[0]
    return miembro


def retTranslationGroup(iD):
    n = np.prod(iD)
    tTG = np.array([[0]*n]*3)
    newTrans = np.array([[0]*n]*n)
    Members = retMembers(iD)
    for im in range(n):
        for i in range(n):
            tTG[:, i] = (Members[:, im] + Members[:, i]) % iD[:]

        for i in range(n):
            for j in range(n):
                if np.array_equal(tTG[:, i], Members[:, j]):
                    newTrans[im, i] = j
                    break
    return Members, newTrans


def gen_only_starter(starter_list, n=1000):
    # multiple_shuffles(starter_list, n)
    # starter_list has the element profile we need rn
    np.random.seed(42)
    all_lists = np.array([np.random.permutation(starter_list) 
                          for _ in range(n)])

    return all_lists


def find_combinations_target(target):
    result = []
    for i in range(1, target + 1):
        for j in range(i, target + 1):
            for k in range(j, target + 1):
                if i * j * k == target:
                    result.append((i, j, k))
    return result


class structures_pool:
    def __init__(self, system_elements):
        self.system_elements = system_elements   
        
        self.structures = []
        self.sizes = []

    def _create_prim(self,  system_lat, lattice_vals, extra_sites=None):
        if system_lat == 'FCC' or system_lat == 'BCC':
            self.a0 = lattice_vals['a0']
        if system_lat == 'HCP' or system_lat == 'boride_191':
            self.a0 = lattice_vals['a0']
            self.c0 = lattice_vals['c/a'] * self.a0
            
        if system_lat == 'FCC' and extra_sites is None:
            self.extra_atoms = None
            a0 = self.a0
            self.prim = Atoms(self.system_elements[0],
                              scaled_positions=[
                     [0.0,   0.0,   0.0],
                 ],
                 cell=[[a0/2, a0/2, 0],
                       [0, a0/2, a0/2],
                       [a0/2, 0, a0/2],
                       ], pbc=[1, 1, 1]
                        )
            
        if system_lat == 'BCC' and extra_sites is None:
            self.extra_atoms = None
            a0 = self.a0
            self.prim = Atoms(self.system_elements[0], 
                              scaled_positions=[
                [0.0,   0.0,   0.0],
                ],
                cell=[[a0/2, a0/2, -a0/2],
                      [-a0/2, a0/2, a0/2],
                      [a0/2, -a0/2, a0/2],
                      ],
                pbc=[1, 1, 1])
        
        if system_lat == 'HCP' and extra_sites is None:
            self.extra_atoms = None
            a0 = self.a0
            c0 = self.c0
            self.prim = Atoms(self.system_elements[0]*2,
                              scaled_positions=[
                    [1/3, 2/3, 1/4],    
                    [2/3, 1/3, 3/4]
                ],
                cell=[[np.abs(a0*np.cos(120*np.pi/180)), 
                       -np.abs(a0*np.sin(120*np.pi/180)), 0.0],
                      [np.abs(a0*np.cos(120*np.pi/180)), 
                       np.abs(a0*np.sin(120*np.pi/180)), 0.0],
                      [0.0, 0.0, c0],
                      ],
                pbc=[1, 1, 1])
            
        if system_lat == 'boride_191' and extra_sites is None:
            self.extra_atoms = ['B', 'B']
            a0 = self.a0
            c0 = self.c0
            self.prim = Atoms(self.system_elements[0]+'B'*2,
                              scaled_positions=[
                    [0.0, 0.0, 0.0],
                    [1/3, 2/3, 1/2],
                    [2/3, 1/3, 1/2]
                ],
                cell=[[np.abs(a0*np.cos(120*np.pi/180)),
                       -np.abs(a0*np.sin(120*np.pi/180)), 0.0],
                      [np.abs(a0*np.cos(120*np.pi/180)), 
                       np.abs(a0*np.sin(120*np.pi/180)), 0.0],
                      [0.0, 0.0, c0],
                      ],
                pbc=[1, 1, 1])
        print(self.prim)
    
    def _gen_icet(self, N):
        ti = time.time()
        for structure in enumerate_structures(self.prim, range(N + 1), 
                                              self.system_elements):
            self.structures.append(structure)
            self.sizes.append(len(structure))
        print('{} structures in {}s'.format(len(self.structures), 
              time.time() - ti))
        
    def _gen_higher(self, Ns):
        ti = time.time()
        self.structures_ord = []
        
        ELE_LIST = self.system_elements
        prim = self.prim.copy()
        prim_cell = prim.cell.array
        prim_pmg = ase_to_pmg(prim)
        extra_atoms = self.extra_atoms
        
        RMs = [i.rotation_matrix 
               for i in SpacegroupAnalyzer(prim_pmg).
               get_symmetry_operations(cartesian=True)]
        save_ases = []
        
        for n in Ns:
            hs, ss = return_HNF_n_SNF(n, prim_cell, RMs)

            lower = 2
            upper = 6
            sys_d = len(ELE_LIST)
            final_comp=np.empty([0,len(ELE_LIST)], float)

            for i in range(lower,upper+1):
                comps = []
                indices = np.ndindex(*[n-i for i in range(i)])
                j = 0

                for index in indices:
                    j += 1
                    comp = np.array(index)+1
                    if (np.sum(comp)==(n)):
                        comps.append(comp)
                comps = np.vstack(comps)
                for comb in combinations(range(sys_d),i):
                    dummy=np.zeros([comps.shape[0],sys_d],dtype=int)
                    dummy[:, comb]=comps
                    final_comp=np.vstack((final_comp,dummy))


            final_comp = final_comp.astype(int)
            # final_comp shows the number of atoms each element shows using this size of cell and 
            # number of constituents while only choosing a lower --> upper  number of constituents
            # at the same time, we allow lower = 2 to obtain higher size binaries
#             print(final_comp)
#             print(final_comp.shape)
            starters = []
            for f in final_comp:
                starter=[]
                for e, nE in enumerate(f):
                    starter=starter+[e]*nE
                starters.append(starter)

#             print(len(starters))
            n_select = 10000
            if len(starters) < n_select:
                mini_starters = starters
            else:
                mini_starters = np.array(starters)[
                    np.random.choice(np.arange(len(starters)), n_select, replace=False)]


            full_ases = []
            full_clusters = []

            for i_s, starter in enumerate(mini_starters):
                pool_ases = []
                pool_clusters = []
                diags = find_combinations_target(n)

                for i_h, h in enumerate(hs):
                    atoms_dummy = make_supercell(prim, h)
                    diff_structures = gen_only_starter(starter, 1)
                    diff_labels = [[
                        ELE_LIST[j] for j in i] for i in diff_structures]
                    ases = []

                    for i in diff_labels:

                        ite_ase = atoms_dummy.copy()
                        if extra_atoms:
                            ite_ase.set_chemical_symbols(
                                flatten([[j]+extra_atoms for j in i]))
                        else:
                            ite_ase.set_chemical_symbols(i)
                        ases.append(ite_ase.copy())
                        full_ases.append(ite_ase.copy())
                    np.random.seed(int(time.time() * 1e6) % (2**32 - 1))
#             print(n, len(full_ases))
            self.structures_ord.extend(full_ases)
        print('{} structures in {}s'.format(len(self.structures_ord), 
                                            time.time() - ti))
    
    def generate_structures(self, N_higher, N_lower):
        self._gen_icet(N_lower)
        self._gen_higher(np.arange(N_lower,N_higher))
        self.sizes_ord = [len(i) for i in self.structures_ord]
        self.total_sizes = self.sizes + self.sizes_ord
        self.total_strs = self .sizes + self.structures_ord