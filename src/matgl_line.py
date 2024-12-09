import numpy as np
from collections import Counter

def ase_checkrelax(ase_ini,ase_fin,cutoff=0.05):
    before = ase_ini.get_cell().array
    after  = ase_fin.get_cell().array
    before_res = before/np.power(np.linalg.det(before), 1./3.)
    after_res  = after/np.power(np.linalg.det(after), 1./3.)
    diff = np.dot(np.linalg.inv(before_res), after_res)

    out_mat = (diff + diff.T)/2 - np.eye(3)
    distorsion_val = np.linalg.norm(out_mat)
    # os.system(f"echo {distorsion_val} > distorsion")
    #if distorsion_val > cutoff:
        #os.system(f"touch error")
        #print('error')
    return distorsion_val

def row_comp(x):
    old_dict = Counter(x['structure_initial'].get_chemical_symbols())
    return {key: value / x['size'] for key, value in old_dict.items()}


def row_reference(x, dictionary_reference):
    value_reference = x['energy_atom']
    for i in dictionary_reference.keys():
        value_reference = value_reference- (x[i]*dictionary_reference[i])
    return value_reference

def row_strain(x):
    return ase_checkrelax(x['structure_initial'], x['structure_final'])

def row_volume(x):
    return x['structure_final'].get_volume()/x['size']