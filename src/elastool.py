
import numpy as np
from ase.spacegroup import get_spacegroup
import copy

def strain_matrix(input_dict,latt_system, up0):
    indict = input_dict
    strain_matrix_list = []
    if indict['strains_matrix'][0] == 'ohess':
        up = up0
        if indict['dimensional'][0] == '3D':
            if latt_system == 'Cubic':
                strain_matrix_1 = np.array([[up,    0.,    0.], 
                                            [0.,    0., up/2.], 
                                            [0.,  up/2.,   0.]])
                strain_matrix_list.append(strain_matrix_1)

            elif latt_system == 'Hexagonal':
                strain_matrix_1 = np.array([[up,    0.,    0.], 
                                            [0.,    0.,    0.], 
                                            [0.,    0.,    0.]])
                strain_matrix_2 = np.array([[0.,    0.,    0.], 
                                            [0.,    0., up/2.], 
                                            [0.,    up/2., up]])
                strain_matrix_list.append(strain_matrix_1)
                strain_matrix_list.append(strain_matrix_2)

            elif latt_system == 'Trigonal1' or latt_system == 'Trigonal2':
                strain_matrix_1 = np.array([[up,    0.,    0.], 
                                            [0.,    0.,    0.], 
                                            [0.,    0.,    0.]])
                strain_matrix_2 = np.array([[0.,    0.,    0.], 
                                            [0.,    0., up/2.], 
                                            [0.,    up/2.,up]])
                strain_matrix_list.append(strain_matrix_1)
                strain_matrix_list.append(strain_matrix_2)

            elif latt_system == 'Tetragonal1' or latt_system == 'Tetragonal2':
                strain_matrix_1 = np.array([[up,    0.,    0.], 
                                            [0.,    0.,    0.], 
                                            [0.,    0.,    0.]])
                strain_matrix_2 = np.array([[0.,    up/2., 0.], 
                                            [up/2., 0., up/2.], 
                                            [0.,    up/2., up]])
                strain_matrix_list.append(strain_matrix_1)
                strain_matrix_list.append(strain_matrix_2)

            elif latt_system == 'Orthorombic':
                strain_matrix_1 = np.array([[up,    0.,    0.], 
                                            [0.,    0.,    0.], 
                                            [0.,    0.,    0.]])
                strain_matrix_2 = np.array([[0.,    0.,    0.], 
                                            [0.,    up,    0.], 
                                            [0.,    0.,    0.]])
                strain_matrix_3 = np.array([[0.,  up/2.,up/2.], 
                                            [up/2., 0., up/2.], 
                                            [up/2., up/2., up]])
                strain_matrix_list.append(strain_matrix_1)
                strain_matrix_list.append(strain_matrix_2)
                strain_matrix_list.append(strain_matrix_3)

            elif latt_system == 'Monoclinic':
                strain_matrix_1 = np.array([[up,    0.,   0.], 
                                            [0.,    0.,   0.], 
                                            [0.,    0.,   0.]])
                strain_matrix_2 = np.array([[0.,    0.,   0.], 
                                            [0.,    up,   0.], 
                                            [0.,    0.,   0.]])
                strain_matrix_3 = np.array([[0.,    0.,   0.], 
                                            [0.,    0.,up/2.], 
                                            [0., up/2.,   up]])
                strain_matrix_4 = np.array([[0., up/2.,up/2.], 
                                            [up/2., 0.,   0.], 
                                            [up/2., 0.,   0.]])
                strain_matrix_list.append(strain_matrix_1)
                strain_matrix_list.append(strain_matrix_2)
                strain_matrix_list.append(strain_matrix_3)
                strain_matrix_list.append(strain_matrix_4)

            elif latt_system == 'Triclinic':
                strain_matrix_1 = np.array([[up,    0.,   0.], 
                                            [0.,    0.,   0.], 
                                            [0.,    0.,   0.]])
                strain_matrix_2 = np.array([[0.,    0.,   0.], 
                                            [0.,    up,   0.], 
                                            [0.,    0.,   0.]])
                strain_matrix_3 = np.array([[0.,    0.,   0.], 
                                            [0.,    0.,   0.], 
                                            [0.,    0.,   up]])
                strain_matrix_4 = np.array([[0.,    0.,   0.], 
                                            [0.,    0.,up/2.], 
                                            [0., up/2.,   0.]])
                strain_matrix_5 = np.array([[0.,    0.,up/2.], 
                                            [0.,    0.,   0.], 
                                            [up/2., 0.,   0.]])
                strain_matrix_6 = np.array([[0., up/2.,   0.], 
                                            [up/2., 0.,   0.], 
                                            [0.,    0.,   0.]])
                strain_matrix_list.append(strain_matrix_1)
                strain_matrix_list.append(strain_matrix_2)
                strain_matrix_list.append(strain_matrix_3)
                strain_matrix_list.append(strain_matrix_4)
                strain_matrix_list.append(strain_matrix_5)
                strain_matrix_list.append(strain_matrix_6)
        
        elif indict['dimensional'][0] == '2D':
            # 2D in-plane strains: in xy plane
            if latt_system == 'isotropy':
                strain_matrix_1 = array([[up,    0.,    0.], 
                                            [0.,    0.,    0.], 
                                            [0.,    0.,    0.]])
                strain_matrix_list.append(strain_matrix_1)

            elif latt_system == 'tetragonal':
                strain_matrix_1 = np.array([[up,  up/2,    0.], 
                                            [up/2,  0.,    0.], 
                                            [0.,    0.,    0.]])
                strain_matrix_list.append(strain_matrix_1)

            elif latt_system == 'orthotropy':
                strain_matrix_1 = np.array([[up,  up/2,    0.], 
                                            [up/2,  0.,    0.], 
                                            [0.,    0.,    0.]])
                strain_matrix_2 = np.array([[0.,    0.,    0.], 
                                            [0.,    up,    0.], 
                                            [0.,    0.,    0.]])                                        
                strain_matrix_list.append(strain_matrix_1)
                strain_matrix_list.append(strain_matrix_2)

            elif latt_system == 'anisotropy':
                strain_matrix_1 = np.array([[up,    0.,    0.], 
                                            [0.,    0.,    0.], 
                                            [0.,    0.,    0.]])
                strain_matrix_2 = np.array([[0.,    0.,    0.], 
                                            [0.,    up,    0.], 
                                            [0.,    0.,    0.]])                                        
                strain_matrix_3 = np.array([[0.,  up/2,    0.], 
                                            [up/2,  0.,    0.], 
                                            [0.,    0.,    0.]])                                        
                strain_matrix_list.append(strain_matrix_1)
                strain_matrix_list.append(strain_matrix_2)
                strain_matrix_list.append(strain_matrix_3)

    elif indict['strains_matrix'][0] == 'asess':
        if indict['dimensional'][0] == '3D':
            strain_matrix_1 = up0 * np.array([[1.,   0.,  0.], 
                                              [0.,   0.,  0.], 
                                              [0.,   0.,  0.]])
            strain_matrix_2 = up0 * np.array([[0.,   0.,  0.], 
                                              [0.,   1.,  0.], 
                                              [0.,   0.,  0.]])
            strain_matrix_3 = up0 * np.array([[0.,   0.,  0.], 
                                              [0.,   0.,  0.], 
                                              [0.,   0.,  1.]])
            strain_matrix_4 = up0 * np.array([[0.,   0,   0.], 
                                              [0.,  0.,1./2.], 
                                              [0.,  1./2.,0.]])
            strain_matrix_5 = up0 * np.array([[0., 0., 1./2.], 
                                              [0.,   0.,  0.], 
                                              [1./2.,0.,  0.]])
            strain_matrix_6 = up0 * np.array([[0., 1./2., 0.], 
                                              [1./2.,  0.,0.], 
                                              [0.,   0.,  0.]])
            strain_matrix_list = [strain_matrix_1, strain_matrix_2, strain_matrix_3, strain_matrix_4, strain_matrix_5, strain_matrix_6]
        
        elif indict['dimensional'][0] == '2D':
            strain_matrix_1 = up0 * np.array([[1.,    0.,  0.], 
                                              [0.,    0.,  0.], 
                                              [0.,    0.,  0.]])
            strain_matrix_2 = up0 * np.array([[0.,    0.,  0.], 
                                              [0.,    1.,  0.], 
                                              [0.,    0.,  0.]])
            strain_matrix_3 = up0 * np.array([[0., 1./2.,  0.], 
                                              [1./2., 0.,  0.], 
                                              [0.,    0.,  0.]])
            strain_matrix_list = [strain_matrix_1, strain_matrix_2, strain_matrix_3]

    elif indict['strains_matrix'][0] == 'ulics':
        if indict['dimensional'][0] == '3D':
            #up = 10. ** (-3.)
            up = up0 / 6.
            strain_matrix_1 = up * np.array([[1.,   6./2.,  5./2.], 
                                             [6./2.,   2.,  4./2.], 
                                             [5./2.,   4./2.,  3.]])
            strain_matrix_2 = up * np.array([[-2., -5./2.,  6./2.], 
                                             [-5./2.,  1., -3./2.], 
                                             [6./2.,  -3./2.,  4.]])
            strain_matrix_3 = up * np.array([[3.,  -4./2.,  2./2.], 
                                             [-4./2., -5.,  6./2.], 
                                             [2./2.,   6./2., -1.]])
            strain_matrix_4 = up * np.array([[-4., -2./2., -3./2.], 
                                             [-2./2.,  -6., 1./2.], 
                                             [-3./2.,  1./2.,  5.]])
            strain_matrix_5 = up * np.array([[5.,  -3./2., -1./2.], 
                                             [-3./2.,  4., -2./2.], 
                                             [-1./2., -2./2.,  6.]])
            strain_matrix_6 = up * np.array([[-6.,  1./2., -4./2.], 
                                             [1./2.,   3.,  5./2.], 
                                             [-4./2.,  5./2., -2.]])
            strain_matrix_list = [strain_matrix_1, strain_matrix_2, strain_matrix_3, strain_matrix_4, strain_matrix_5, strain_matrix_6]
            
            #### mod #########################
            if latt_system == 'Cubic':
                return strain_matrix_list[:1]
            elif latt_system == 'Hexagonal':
                return strain_matrix_list[:2]
            elif latt_system == 'Trigonal1':
                return strain_matrix_list[:2]
            elif latt_system == 'Trigonal2':
                return strain_matrix_list[:2]
            elif latt_system == 'Tetragonal1':
                return strain_matrix_list[:2]
            elif latt_system == 'Tetragonal2':
                return strain_matrix_list[:2]
            elif latt_system == 'Orthorombic':
                return strain_matrix_list[:3]
            elif latt_system == 'Monoclinic':
                return strain_matrix_list[:5]
            elif latt_system == 'Triclinic':
                return strain_matrix_list
            ##################################
            
            
            
            
            
        elif indict['dimensional'][0] == '2D':
            up = up0 / 3.
            strain_matrix_1 = up * np.array([[1.,  3./2.,   0.], 
                                             [3./2.,  2.,   0.], 
                                             [0.,     0.,   0.]])
            strain_matrix_2 = up * np.array([[2.,  1./2.,   0.], 
                                             [1./2,   3.,   0.], 
                                             [0.,     0.,   0.]])
            strain_matrix_3 = up * np.array([[3.,     1.,   0.], 
                                             [1.,     1.,   0.], 
                                             [0.,     0.,   0.]])
            strain_matrix_list = [strain_matrix_1, strain_matrix_2, strain_matrix_3]

    return strain_matrix_list



def deform_cell_asess_strains(input_dict,latt_system, cell, up):
    indict=input_dict
    deformed_cell_list = []
    deform_matrix_list = []
    I_ = np.identity(3)
    deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[0])
    deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[1])
    deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[2])

    if indict['dimensional'][0] == '3D':
        deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[3])
        deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[4])
        deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[5])
    
    for deform_matrix in deform_matrix_list:
        deformed_cell = np.dot(cell, deform_matrix)
        deformed_cell_list.append(deformed_cell)
    
    return deformed_cell_list

def deform_cell_ohess_strains(input_dict,latt_system, cell, up):
    indict=input_dict
    deformed_cell_list = []
    deform_matrix_list = []
    I_ = np.identity(3)

    if indict['dimensional'][0] == '3D':
        if latt_system == 'Cubic':
            deform_matrix = I_ + strain_matrix(indict,latt_system, up)[0]
            deformed_cell = np.dot(cell, deform_matrix)
            deformed_cell_list.append(deformed_cell)

        elif latt_system == 'Hexagonal':
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[0])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[1])
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)
                
        elif latt_system == 'Trigonal1' or latt_system == 'Trigonal2':
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[0])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[1])
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)

        elif latt_system == 'Tetragonal1' or latt_system == 'Tetragonal2':
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[0])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[1])
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)

        elif latt_system == 'Orthorombic':
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[0])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[1])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[2])
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)   

        elif latt_system == 'Monoclinic':
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[0])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[1])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[2])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[3])
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell) 

        elif latt_system == 'Triclinic':
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[0])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[1])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[2])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[3])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[4])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[5])
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)

    elif indict['dimensional'][0] == '2D':
        if latt_system == 'isotropy' or latt_system == 'tetragonal':
            deform_matrix = I_ + strain_matrix(indict,latt_system, up)[0]
            deformed_cell = np.dot(cell, deform_matrix)
            deformed_cell_list.append(deformed_cell)

        elif latt_system == 'orthotropy':
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[0])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[1])
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)

        elif latt_system == 'anisotropy':
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[0])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[1])
            deform_matrix_list.append(I_ + strain_matrix(indict,latt_system, up)[2])
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)

        else:
            print('Crystal system is not parsed correctly!!!')
            exit(1)

    return deformed_cell_list



def deform_cell_ulics(input_dict,latt_system, cell, up):
    indict=input_dict
    deformed_cell_list = []
    deform_matrix_list = []
    I_ = np.identity(3)
    if indict['dimensional'][0] == '3D':
        deform_matrix_1 = I_ + strain_matrix(indict,latt_system, up)[0]
        deform_matrix_2 = I_ + strain_matrix(indict,latt_system, up)[1]
        deform_matrix_3 = I_ + strain_matrix(indict,latt_system, up)[2]
        deform_matrix_4 = I_ + strain_matrix(indict,latt_system, up)[3]
        deform_matrix_5 = I_ + strain_matrix(indict,latt_system, up)[4]
        deform_matrix_6 = I_ + strain_matrix(indict,latt_system, up)[5]
                                                                                                                    
        if latt_system == 'Cubic':
            deformed_cell = np.dot(cell, deform_matrix_1)
            deformed_cell_list.append(deformed_cell)

        elif latt_system == 'Hexagonal':
            deform_matrix_list.append(deform_matrix_1)
            deform_matrix_list.append(deform_matrix_2)
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)
                
        elif latt_system == 'Trigonal1':
            deform_matrix_list.append(deform_matrix_1)
            deform_matrix_list.append(deform_matrix_2)
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)

        elif latt_system == 'Trigonal2':
            deform_matrix_list.append(deform_matrix_1)
            deform_matrix_list.append(deform_matrix_2)
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)

        elif latt_system == 'Tetragonal1':
            deform_matrix_list.append(deform_matrix_1)
            deform_matrix_list.append(deform_matrix_2)
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)

        elif latt_system == 'Tetragonal2':
            deform_matrix_list.append(deform_matrix_1)
            deform_matrix_list.append(deform_matrix_2)
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)

        elif latt_system == 'Orthorombic':
            deform_matrix_list.append(deform_matrix_1)
            deform_matrix_list.append(deform_matrix_2)
            deform_matrix_list.append(deform_matrix_3)
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)   

        elif latt_system == 'Monoclinic':
            deform_matrix_list.append(deform_matrix_1)
            deform_matrix_list.append(deform_matrix_2)
            deform_matrix_list.append(deform_matrix_3)
            deform_matrix_list.append(deform_matrix_4)
            deform_matrix_list.append(deform_matrix_5)
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell) 

        elif latt_system == 'Triclinic':
            deform_matrix_list.append(deform_matrix_1)
            deform_matrix_list.append(deform_matrix_2)
            deform_matrix_list.append(deform_matrix_3)
            deform_matrix_list.append(deform_matrix_4)
            deform_matrix_list.append(deform_matrix_5)
            deform_matrix_list.append(deform_matrix_6)
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)
                
        else:
            print('Crystal system is not parsed correctly!!!')
            exit(1)

    elif indict['dimensional'][0] == '2D':
        deform_matrix_1 = identity_matrix + strain_matrix(latt_system, up)[0]
        deform_matrix_2 = identity_matrix + strain_matrix(latt_system, up)[1]
        deform_matrix_3 = identity_matrix + strain_matrix(latt_system, up)[2]

        if latt_system == 'isotropy' or latt_system == 'tetragonal':
            deformed_cell = np.dot(cell, deform_matrix_1)
            deformed_cell_list.append(deformed_cell)

        elif latt_system == 'orthotropy':
            deform_matrix_list.append(deform_matrix_1)
            deform_matrix_list.append(deform_matrix_2)
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)

        elif latt_system == 'anisotropy':
            deform_matrix_list.append(deform_matrix_1)
            deform_matrix_list.append(deform_matrix_2)
            deform_matrix_list.append(deform_matrix_3)
            for deform_matrix in deform_matrix_list:
                deformed_cell = np.dot(cell, deform_matrix)
                deformed_cell_list.append(deformed_cell)
        else:
            print('Crystal system is not parsed correctly!!!')
            exit(1)
            
    return deformed_cell_list

def expanded_strain_ase_list(ase_struct, indct,lst_strain):
    res=strain_matrix_list_mag(ase_struct, indct, lst_strain)
    res.insert(0,
           [np.array([[0.0,0.0,0.0],
           [0.0,0.0,0.0],
           [0.0,0.0,0.0]])])
    return res

def group_ordering(ase_st):

    number = get_spacegroup(ase_st, symprec=1e-2).no
    #print(number)
    if number <= 2:
        return "Triclinic"
    if 2 < number <= 15:
        return "Monoclinic"
    if 15 < number <= 74:
        return "Orthorombic"
    if  74 < number <= 88:
        return "Tetragonal2"
    if 89 < number <= 142:
        return "Tetragonal1"
    if 142 < number <= 148:
        return "Trigonal2"
    if 149 < number <= 167:
        return "Trigonal1"
    if 167 < number <= 194:
        return "Hexagonal"
    if 194 < number:
        return "Cubic"
        

def strain_matrix_ase(ase_st,input_dict,mag):
    cell=ase_st.get_cell().array
    latt_system=group_ordering(ase_st)
    #print(latt_system)
    #print(strain_matrix(input_dict,latt_system,mag))
    return strain_matrix(input_dict,latt_system,mag)

def strain_matrix_list_mag(ase_st,input_dict,list_mag):
    cell=ase_st.get_cell().array
    latt_system=group_ordering(ase_st)
    
    s_l=[]
    for mag in list_mag:
        s_l.append(strain_matrix(input_dict,latt_system,mag))
    return s_l

def flatten_list(lst):
    flattened = []
    for item in lst:
        if isinstance(item, list):
            flattened.extend(flatten_list(item))
        else:
            flattened.append(item)
    return flattened




def Cubic(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        s = strain_matrix(indict,latt_system, up)[0]

        #print(s)
        stress_list = np.array(stress_set_dict[up][0])

        eplisons_now = np.array([[s[0][0], s[1][1]+s[2][2], 0.],
                                 [s[1][1], s[0][0]+s[2][2], 0.],
                                 [s[2][2], s[0][0]+s[1][1], 0.],
                                 [0.,      0.,        2*s[1][2]],
                                 [0.,      0.,        2*s[0][2]],
                                 [0.,      0.,        2*s[0][1]]])

        stresses_now = np.array([[stress_list[0]],
                                 [stress_list[1]],
                                 [stress_list[2]],
                                 [stress_list[3]],
                                 [stress_list[4]],
                                 [stress_list[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))

    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor

    c11 = cij[0][0]
    c12 = cij[1][0]
    c44 = cij[2][0]

    B_v = (c11+2*c12)/3.
    B_r = B_v

    G_v = (c11-c12+3*c44)/5.
    G_r = 5*(c11-c12)*c44/(4*c44+3*(c11-c12))

    B_vrh = (B_v+B_r)/2.
    G_vrh = (G_v+G_r)/2.
    E = 9*B_vrh*G_vrh/(3*B_vrh+G_vrh)
    v = (3*B_vrh-2*G_vrh)/(2*(3*B_vrh+G_vrh))

    elastic_constants_dict['c11'] = c11
    elastic_constants_dict['c12'] = c12
    elastic_constants_dict['c44'] = c44

    elastic_constants_dict['B_v'] = B_v
    elastic_constants_dict['B_r'] = B_r
    elastic_constants_dict['G_v'] = G_v
    elastic_constants_dict['G_r'] = G_r
    elastic_constants_dict['B_vrh'] = B_vrh
    elastic_constants_dict['G_vrh'] = G_vrh
    elastic_constants_dict['E'] = E
    elastic_constants_dict['v'] = v

    return elastic_constants_dict


def Hexagonal(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        if up == 0:
            s1 = strain_matrix(indict,latt_system, up)[0]
            stress_list1 = np.array(stress_set_dict[up][0])
            
            eplisons_now = np.array([[s1[0][0],  s1[1][1],  s1[2][2],  0., 0.],
                                     [s1[1][1],  s1[0][0],  s1[2][2],  0., 0.],
                                     [0., 0., s1[0][0]+s1[1][1], s1[2][2], 0.],
                                     [0.,    0.,     0.,     0.,   2*s1[1][2]],
                                     [0.,    0.,     0.,     0.,   2*s1[0][2]],
                                     [s1[0][1],   -s1[0][1], 0.,       0., 0.]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]]])
        else:
            s1 = strain_matrix(indict,latt_system, up)[0]
            s2 = strain_matrix(indict,latt_system, up)[1]
            stress_list1 = np.array(stress_set_dict[up][0])
            stress_list2 = np.array(stress_set_dict[up][1])

            eplisons_now = np.array([[s1[0][0],  s1[1][1],  s1[2][2],  0., 0.],
                                     [s1[1][1],  s1[0][0],  s1[2][2],  0., 0.],
                                     [0., 0., s1[0][0]+s1[1][1], s1[2][2], 0.],
                                     [0.,    0.,     0.,     0.,   2*s1[1][2]],
                                     [0.,    0.,     0.,     0.,   2*s1[0][2]],
                                     [s1[0][1],   -s1[0][1], 0.,       0., 0.],
                                     [s2[0][0],  s2[1][1],  s2[2][2],  0., 0.],
                                     [s2[1][1],  s2[0][0],  s2[2][2],  0., 0.],
                                     [0., 0., s2[0][0]+s2[1][1], s2[2][2], 0.],
                                     [0.,    0.,     0.,     0.,   2*s2[1][2]],
                                     [0.,    0.,     0.,     0.,   2*s2[0][2]],
                                     [s2[0][1],   -s2[0][1], 0.,       0., 0.]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]],
                                     [stress_list2[0]],
                                     [stress_list2[1]],
                                     [stress_list2[2]],
                                     [stress_list2[3]],
                                     [stress_list2[4]],
                                     [stress_list2[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))

    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor

    c11 = cij[0][0]
    c12 = cij[1][0]
    c13 = cij[2][0]
    c33 = cij[3][0]
    c44 = cij[4][0]

    M = c11+c12+2*c33-4*c13
    C2 = (c11+c12)*c33-2*c13*c13
    c66 = (c11-c12)/2.

    B_v = (2*(c11+c12)+4*c13+c33)/9.
    G_v = (M+12*c44+12*c66)/30.
    B_r = C2/M
    G_r = 2.5*(C2*c44*c66)/(3*B_v*c44*c66+C2*(c44+c66))

    B_vrh = (B_v+B_r)/2.
    G_vrh = (G_v+G_r)/2.
    E = 9*B_vrh*G_vrh/(3*B_vrh+G_vrh)
    v = (3*B_vrh-2*G_vrh)/(2*(3*B_vrh+G_vrh))

    elastic_constants_dict['c11'] = c11
    elastic_constants_dict['c12'] = c12
    elastic_constants_dict['c13'] = c13
    elastic_constants_dict['c33'] = c33
    elastic_constants_dict['c44'] = c44

    elastic_constants_dict['B_v'] = B_v
    elastic_constants_dict['B_r'] = B_r
    elastic_constants_dict['G_v'] = G_v
    elastic_constants_dict['G_r'] = G_r
    elastic_constants_dict['B_vrh'] = B_vrh
    elastic_constants_dict['G_vrh'] = G_vrh
    elastic_constants_dict['E'] = E
    elastic_constants_dict['v'] = v

    return elastic_constants_dict


def Trigonal1(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        if up == 0:
            s1 = strain_matrix(indict,latt_system, up)[0]
            stress_list1 = np.array(stress_set_dict[up][0])

            eplisons_now = np.array([[s1[0][0],  s1[1][1],  s1[2][2], 2*s1[1][2],  0., 0.],
                                     [s1[1][1],  s1[0][0],  s1[2][2],-2*s1[1][2],  0., 0.],
                                     [0.,    0., s1[0][0]+s1[1][1],    0.,   s1[2][2], 0.],
                                     [0.,    0.,  0.,  s1[0][0]-s1[1][1],  0., 2*s1[1][2]],
                                     [0.,    0.,     0., 2*s1[0][1],       0., 2*s1[0][2]],
                                     [s1[0][1],  -s1[0][1],  0.,   2*s1[0][2],   0.,   0.]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]]])
        else:
            s1 = strain_matrix(indict,latt_system, up)[0]
            s2 = strain_matrix(indict,latt_system, up)[1]
            stress_list1 = np.array(stress_set_dict[up][0])
            stress_list2 = np.array(stress_set_dict[up][1])

            eplisons_now = np.array([[s1[0][0],  s1[1][1],  s1[2][2], 2*s1[1][2],  0., 0.],
                                     [s1[1][1],  s1[0][0],  s1[2][2],-2*s1[1][2],  0., 0.],
                                     [0.,    0., s1[0][0]+s1[1][1],    0.,   s1[2][2], 0.],
                                     [0.,    0.,  0.,  s1[0][0]-s1[1][1],  0., 2*s1[1][2]],
                                     [0.,    0.,     0., 2*s1[0][1],       0., 2*s1[0][2]],
                                     [s1[0][1],  -s1[0][1],  0.,   2*s1[0][2],   0.,   0.],
                                     [s2[0][0],  s2[1][1],  s2[2][2], 2*s2[1][2],  0., 0.],
                                     [s2[1][1],  s2[0][0],  s2[2][2],-2*s2[1][2],  0., 0.],
                                     [0.,    0., s2[0][0]+s2[1][1],    0.,   s2[2][2], 0.],
                                     [0.,    0.,  0.,  s2[0][0]-s2[1][1],  0., 2*s2[1][2]],
                                     [0.,    0.,     0., 2*s2[0][1],       0., 2*s2[0][2]],
                                     [s2[0][1],  -s2[0][1],  0.,   2*s2[0][2],   0.,   0.]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]],
                                     [stress_list2[0]],
                                     [stress_list2[1]],
                                     [stress_list2[2]],
                                     [stress_list2[3]],
                                     [stress_list2[4]],
                                     [stress_list2[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))

    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor

    c11 = cij[0][0]
    c12 = cij[1][0]
    c13 = cij[2][0]
    c14 = cij[3][0]
    c33 = cij[4][0]
    c44 = cij[5][0]
    elastic_constants_dict['c11'] = c11
    elastic_constants_dict['c12'] = c12
    elastic_constants_dict['c13'] = c13
    elastic_constants_dict['c14'] = c14
    elastic_constants_dict['c33'] = c33
    elastic_constants_dict['c44'] = c44

    return elastic_constants_dict


def Trigonal2(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        if up == 0:
            s1 = strain_matrix(indict,latt_system, up)[0]
            stress_list1 = np.array(stress_set_dict[up][0])

            eplisons_now = np.array([[s1[0][0],  s1[1][1],  s1[2][2],  2*s1[1][2],  2*s1[0][2],  0., 0.],
                                     [s1[1][1],  s1[0][0],  s1[2][2], -2*s1[1][2], -2*s1[0][2],  0., 0.],
                                     [0.,    0., s1[0][0]+s1[1][1],      0.,      0.,      s1[2][2], 0.],
                                     [0.,    0.,     0., s1[0][0]-s1[1][1], -2*s1[0][1], 0., 2*s1[1][2]],
                                     [0.,    0.,     0., 2*s1[0][1], s1[0][0]-s1[1][1],  0., 2*s1[0][2]],
                                     [s1[0][1],  -s1[0][1],  0.,   2*s1[0][2],   -2*s1[1][2],   0.,  0.]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]]])
        else:
            s1 = strain_matrix(indict,latt_system, up)[0]
            s2 = strain_matrix(indict,latt_system, up)[1]
            stress_list1 = np.array(stress_set_dict[up][0])
            stress_list2 = np.array(stress_set_dict[up][1])

            eplisons_now = np.array([[s1[0][0],  s1[1][1],  s1[2][2],  2*s1[1][2],  2*s1[0][2],  0., 0.],
                                     [s1[1][1],  s1[0][0],  s1[2][2], -2*s1[1][2], -2*s1[0][2],  0., 0.],
                                     [0.,    0., s1[0][0]+s1[1][1],      0.,      0.,      s1[2][2], 0.],
                                     [0.,    0.,     0., s1[0][0]-s1[1][1], -2*s1[0][1], 0., 2*s1[1][2]],
                                     [0.,    0.,     0., 2*s1[0][1], s1[0][0]-s1[1][1],  0., 2*s1[0][2]],
                                     [s1[0][1],  -s1[0][1],  0.,   2*s1[0][2],   -2*s1[1][2],   0.,  0.],
                                     [s2[0][0],  s2[1][1],  s2[2][2],  2*s2[1][2],  2*s2[0][2],  0., 0.],
                                     [s2[1][1],  s2[0][0],  s2[2][2], -2*s2[1][2], -2*s2[0][2],  0., 0.],
                                     [0.,    0., s2[0][0]+s2[1][1],      0.,      0.,      s2[2][2], 0.],
                                     [0.,    0.,     0., s2[0][0]-s2[1][1], -2*s2[0][1], 0., 2*s2[1][2]],
                                     [0.,    0.,     0., 2*s2[0][1], s2[0][0]-s2[1][1],  0., 2*s2[0][2]],
                                     [s2[0][1],  -s2[0][1],  0.,   2*s2[0][2],   -2*s2[1][2],   0.,  0.]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]],
                                     [stress_list2[0]],
                                     [stress_list2[1]],
                                     [stress_list2[2]],
                                     [stress_list2[3]],
                                     [stress_list2[4]],
                                     [stress_list2[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))

    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor

    c11 = cij[0][0]
    c12 = cij[1][0]
    c13 = cij[2][0]
    c14 = cij[3][0]
    c15 = cij[4][0]
    c33 = cij[5][0]
    c44 = cij[6][0]
    elastic_constants_dict['c11'] = c11
    elastic_constants_dict['c12'] = c12
    elastic_constants_dict['c13'] = c13
    elastic_constants_dict['c14'] = c14
    elastic_constants_dict['c15'] = c15
    elastic_constants_dict['c33'] = c33
    elastic_constants_dict['c44'] = c44

    return elastic_constants_dict
    

def Tetragonal1(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        if up == 0:
            s1 = strain_matrix(indict,latt_system, up)[0]
            stress_list1 = np.array(stress_set_dict[up][0])

            eplisons_now = np.array([[s1[0][0],  s1[1][1],  s1[2][2],   0.,   0.,   0.],
                                     [s1[1][1],  s1[0][0],  s1[2][2],   0.,   0.,   0.],
                                     [0.,    0., s1[0][0]+s1[1][1], s1[2][2], 0.,   0.],
                                     [0.,    0.,     0.,     0.,      2*s1[1][2],   0.],
                                     [0.,    0.,     0.,     0.,      2*s1[0][2],   0.],
                                     [0.,    0.,     0.,     0.,       0.,  2*s1[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]]])
        else:
            s1 = strain_matrix(indict,latt_system, up)[0]
            s2 = strain_matrix(indict,latt_system, up)[1]
            stress_list1 = np.array(stress_set_dict[up][0])
            stress_list2 = np.array(stress_set_dict[up][1])

            eplisons_now = np.array([[s1[0][0],  s1[1][1],  s1[2][2],   0.,   0.,   0.],
                                     [s1[1][1],  s1[0][0],  s1[2][2],   0.,   0.,   0.],
                                     [0.,    0., s1[0][0]+s1[1][1], s1[2][2], 0.,   0.],
                                     [0.,    0.,     0.,     0.,      2*s1[1][2],   0.],
                                     [0.,    0.,     0.,     0.,      2*s1[0][2],   0.],
                                     [0.,    0.,     0.,     0.,       0.,  2*s1[0][1]],
                                     [s2[0][0],  s2[1][1],  s2[2][2],   0.,   0.,   0.],
                                     [s2[1][1],  s2[0][0],  s2[2][2],   0.,   0.,   0.],
                                     [0.,    0., s2[0][0]+s2[1][1], s2[2][2], 0.,   0.],
                                     [0.,    0.,     0.,     0.,      2*s2[1][2],   0.],
                                     [0.,    0.,     0.,     0.,      2*s2[0][2],   0.],
                                     [0.,    0.,     0.,     0.,       0.,  2*s2[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]],
                                     [stress_list2[0]],
                                     [stress_list2[1]],
                                     [stress_list2[2]],
                                     [stress_list2[3]],
                                     [stress_list2[4]],
                                     [stress_list2[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))

    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor

    c11 = cij[0][0]
    c12 = cij[1][0]
    c13 = cij[2][0]
    c33 = cij[3][0]
    c44 = cij[4][0]
    c66 = cij[5][0]

    M = c11+c12+2*c33-4*c13
    C2 = (c11+c12)*c33-2*c13*c13

    B_v = (2*(c11+c12)+c33+4*c13)/9.
    G_v = (M+3*c11-3*c12+12*c44+6*c66)/30.
    B_r = C2/M
    G_r = 15./(18*B_v/C2+6/(c11-c12)+6/c44+3/c66)

    B_vrh = (B_v+B_r)/2.
    G_vrh = (G_v+G_r)/2.
    E = 9*B_vrh*G_vrh/(3*B_vrh+G_vrh)
    v = (3*B_vrh-2*G_vrh)/(2*(3*B_vrh+G_vrh))

    elastic_constants_dict['c11'] = c11
    elastic_constants_dict['c12'] = c12
    elastic_constants_dict['c13'] = c13
    elastic_constants_dict['c33'] = c33
    elastic_constants_dict['c44'] = c44
    elastic_constants_dict['c66'] = c66

    elastic_constants_dict['B_v'] = B_v
    elastic_constants_dict['B_r'] = B_r
    elastic_constants_dict['G_v'] = G_v
    elastic_constants_dict['G_r'] = G_r

    elastic_constants_dict['B_vrh'] = B_vrh
    elastic_constants_dict['G_vrh'] = G_vrh
    elastic_constants_dict['E'] = E
    elastic_constants_dict['v'] = v

    return elastic_constants_dict


def Tetragonal2(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        if up == 0:
            s1 = strain_matrix(indict,latt_system, up)[0]
            stress_list1 = np.array(stress_set_dict[up][0])

            eplisons_now = np.array([[s1[0][0],  s1[1][1],  s1[2][2], 2*s1[0][1],  0.,   0.,   0.],
                                     [s1[1][1],  s1[0][0],  s1[2][2],-2*s1[0][1],  0.,   0.,   0.],
                                     [0.,    0., s1[0][0]+s1[1][1],      0.,   s1[2][2], 0.,   0.],
                                     [0.,    0.,     0.,     0.,     0.,     2*s1[1][2],       0.],
                                     [0.,    0.,     0.,     0.,     0.,     2*s1[0][2],       0.],
                                     [0.,    0.,     0., s1[0][0]-s1[1][1],   0.,   0.,2*s1[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]]])
        else:
            s1 = strain_matrix(indict,latt_system, up)[0]
            s2 = strain_matrix(indict,latt_system, up)[1]
            stress_list1 = np.array(stress_set_dict[up][0])
            stress_list2 = np.array(stress_set_dict[up][1])

            eplisons_now = np.array([[s1[0][0],  s1[1][1],  s1[2][2], 2*s1[0][1],  0.,   0.,   0.],
                                     [s1[1][1],  s1[0][0],  s1[2][2],-2*s1[0][1],  0.,   0.,   0.],
                                     [0.,    0., s1[0][0]+s1[1][1],      0.,   s1[2][2], 0.,   0.],
                                     [0.,    0.,     0.,     0.,     0.,     2*s1[1][2],       0.],
                                     [0.,    0.,     0.,     0.,     0.,     2*s1[0][2],       0.],
                                     [0.,    0.,     0., s1[0][0]-s1[1][1],   0.,   0.,2*s1[0][1]],
                                     [s2[0][0],  s2[1][1],  s2[2][2], 2*s2[0][1],  0.,   0.,   0.],
                                     [s2[1][1],  s2[0][0],  s2[2][2],-2*s2[0][1],  0.,   0.,   0.],
                                     [0.,    0., s2[0][0]+s2[1][1],      0.,   s2[2][2], 0.,   0.],
                                     [0.,    0.,     0.,     0.,     0.,     2*s2[1][2],       0.],
                                     [0.,    0.,     0.,     0.,     0.,     2*s2[0][2],       0.],
                                     [0.,    0.,     0., s2[0][0]-s2[1][1],   0.,   0.,2*s2[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]],
                                     [stress_list2[0]],
                                     [stress_list2[1]],
                                     [stress_list2[2]],
                                     [stress_list2[3]],
                                     [stress_list2[4]],
                                     [stress_list2[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))

    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor

    c11 = cij[0][0]
    c12 = cij[1][0]
    c13 = cij[2][0]
    c16 = cij[3][0]
    c33 = cij[4][0]
    c44 = cij[5][0]
    c66 = cij[6][0]

    M = c11+c12+2*c33-4*c13
    C2 = (c11+c12)*c33-2*c13*c13

    B_v = (2*(c11+c12)+c33+4*c13)/9.
    G_v = (M+3*c11-3*c12+12*c44+6*c66)/30.
    B_r = C2/M
    G_r = 15./(18*B_v/C2+6/(c11-c12)+6/c44+3/c66)

    B_vrh = (B_v+B_r)/2.
    G_vrh = (G_v+G_r)/2.
    E = 9*B_vrh*G_vrh/(3*B_vrh+G_vrh)
    v = (3*B_vrh-2*G_vrh)/(2*(3*B_vrh+G_vrh))

    elastic_constants_dict['c11'] = c11
    elastic_constants_dict['c12'] = c12
    elastic_constants_dict['c13'] = c13
    elastic_constants_dict['c16'] = c16
    elastic_constants_dict['c33'] = c33
    elastic_constants_dict['c44'] = c44
    elastic_constants_dict['c66'] = c66

    elastic_constants_dict['B_v'] = B_v
    elastic_constants_dict['B_r'] = B_r
    elastic_constants_dict['G_v'] = G_v
    elastic_constants_dict['G_r'] = G_r

    elastic_constants_dict['B_vrh'] = B_vrh
    elastic_constants_dict['G_vrh'] = G_vrh
    elastic_constants_dict['E'] = E
    elastic_constants_dict['v'] = v
    
    return elastic_constants_dict


def Orthorombic(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        if up == 0:
            s1 = strain_matrix(indict,latt_system, up)[0]
            #print(s1)
            stress_list1 = np.array(stress_set_dict[up][0])

            eplisons_now = np.array([[s1[0][0], s1[1][1], s1[2][2], 0., 0., 0., 0., 0., 0.],
                                     [0., s1[0][0], 0., s1[1][1], s1[2][2], 0., 0., 0., 0.],
                                     [0., 0., s1[0][0], 0., s1[1][1], s1[2][2], 0., 0., 0.],
                                     [0.,   0.,   0.,   0.,   0.,  0., 2*s1[1][2],  0., 0.],
                                     [0.,   0.,   0.,   0.,   0.,  0.,  0.,2*s1[0][2],  0.],
                                     [0.,   0.,   0.,   0.,   0.,  0.,  0.,  0.,2*s1[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]]])
        else:        
            s1 = strain_matrix(indict,latt_system, up)[0]
            s2 = strain_matrix(indict,latt_system, up)[1]
            s3 = strain_matrix(indict,latt_system, up)[2]
            stress_list1 = np.array(stress_set_dict[up][0])
            stress_list2 = np.array(stress_set_dict[up][1])
            stress_list3 = np.array(stress_set_dict[up][2])

            eplisons_now = np.array([[s1[0][0], s1[1][1], s1[2][2], 0., 0., 0., 0., 0., 0.],
                                     [0., s1[0][0], 0., s1[1][1], s1[2][2], 0., 0., 0., 0.],
                                     [0., 0., s1[0][0], 0., s1[1][1], s1[2][2], 0., 0., 0.],
                                     [0.,   0.,   0.,   0.,   0.,  0., 2*s1[1][2],  0., 0.],
                                     [0.,   0.,   0.,   0.,   0.,  0.,  0.,2*s1[0][2],  0.],
                                     [0.,   0.,   0.,   0.,   0.,  0.,  0.,  0.,2*s1[0][1]],
                                     [s2[0][0], s2[1][1], s2[2][2], 0., 0., 0., 0., 0., 0.],
                                     [0., s2[0][0], 0., s2[1][1], s2[2][2], 0., 0., 0., 0.],
                                     [0., 0., s2[0][0], 0., s2[1][1], s2[2][2], 0., 0., 0.],
                                     [0.,   0.,   0.,   0.,   0.,   0.,2*s2[1][2],  0., 0.],
                                     [0.,   0.,   0.,   0.,   0.,   0.,  0.,2*s2[0][2], 0.],
                                     [0.,   0.,   0.,   0.,   0.,   0.,  0., 0.,2*s2[0][1]],
                                     [s3[0][0], s3[1][1], s3[2][2], 0., 0., 0., 0., 0., 0.],
                                     [0., s3[0][0], 0., s3[1][1], s3[2][2], 0., 0., 0., 0.],
                                     [0., 0., s3[0][0], 0., s3[1][1], s3[2][2], 0., 0., 0.],
                                     [0.,   0.,   0.,   0.,   0.,   0., 2*s3[1][2], 0., 0.],
                                     [0.,   0.,   0.,   0.,   0.,   0., 0., 2*s3[0][2], 0.],
                                     [0.,   0.,   0.,   0.,   0.,   0., 0., 0., 2*s3[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]],
                                     [stress_list2[0]],
                                     [stress_list2[1]],
                                     [stress_list2[2]],
                                     [stress_list2[3]],
                                     [stress_list2[4]],
                                     [stress_list2[5]],
                                     [stress_list3[0]],
                                     [stress_list3[1]],
                                     [stress_list3[2]],
                                     [stress_list3[3]],
                                     [stress_list3[4]],
                                     [stress_list3[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))

    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor
    c11 = cij[0][0]
    c12 = cij[1][0]
    c13 = cij[2][0]
    c22 = cij[3][0]
    c23 = cij[4][0]
    c33 = cij[5][0]
    c44 = cij[6][0]
    c55 = cij[7][0]
    c66 = cij[8][0]

    D = c13*(c12*c23-c13*c22)+c23*(c12*c13-c23*c11)+c33*(c11*c22-c12*c12)
    B_v = (c11+c22+c33+2*(c12+c13+c23))/9.
    G_v = (c11+c22+c33+3*(c44+c55+c66)-(c12+c13+c23))/15.
    B_r = D/(c11*(c22+c33-2*c23)+c22*(c33-2*c13)-2*c33*c12+c12*(2*c23-c12)+c13*(2*c12-c13)+c23*(2*c13-c23))
    G_r = 15/(4*(c11*(c22+c33+c23)+c22*(c33+c13)+c33*c12-c12*(c23+c12)-c13*(c12+c13)-c23*(c13+c23))/D+3*(1/c44+1/c55+1/c66))

    B_vrh = (B_v+B_r)/2.
    G_vrh = (G_v+G_r)/2.
    E = 9*B_vrh*G_vrh/(3*B_vrh+G_vrh)
    v = (3*B_vrh-2*G_vrh)/(2*(3*B_vrh+G_vrh))

    elastic_constants_dict['c11'] = c11
    elastic_constants_dict['c12'] = c12
    elastic_constants_dict['c13'] = c13
    elastic_constants_dict['c22'] = c22
    elastic_constants_dict['c23'] = c23
    elastic_constants_dict['c33'] = c33
    elastic_constants_dict['c44'] = c44
    elastic_constants_dict['c55'] = c55
    elastic_constants_dict['c66'] = c66

    elastic_constants_dict['B_v'] = B_v
    elastic_constants_dict['B_r'] = B_r
    elastic_constants_dict['G_v'] = G_v
    elastic_constants_dict['G_r'] = G_r
    elastic_constants_dict['B_vrh'] = B_vrh
    elastic_constants_dict['G_vrh'] = G_vrh
    elastic_constants_dict['E'] = E
    elastic_constants_dict['v'] = v

    return elastic_constants_dict


def Monoclinic(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        if up == 0:
            s1 = strain_matrix(indict,latt_system, up)[0]
            stress_list1 = np.array(stress_set_dict[up][0])

            eplisons_now = np.array([[s1[0][0], s1[1][1], s1[2][2], 2*s1[0][2], 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., s1[0][0], 0., 0., s1[1][1], s1[2][2], 2*s1[0][2], 0., 0., 0., 0., 0., 0.],
                                     [0., 0., s1[0][0], 0.,  0., s1[1][1], 0.,s1[2][2], 2*s1[0][2], 0., 0., 0., 0.],
                                     [0., 0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  2*s1[1][2],   2*s1[0][1],  0., 0.],
                                     [0., 0.,  0.,s1[0][0], 0.,  0.,s1[1][1],  0., s1[2][2],  0., 0.,2*s1[0][2],0.],
                                     [0., 0.,  0., 0.,  0., 0.,  0.,   0.,  0.,  0.,   2*s1[1][2], 0.,  2*s1[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]]])
        else:        
            s1 = strain_matrix(indict,latt_system, up)[0]
            s2 = strain_matrix(indict,latt_system, up)[1]
            s3 = strain_matrix(indict,latt_system, up)[2]
            s4 = strain_matrix(indict,latt_system, up)[3]
            #s5 = strain_matrix(indict,latt_system, up)[4]
            stress_list1 = np.array(stress_set_dict[up][0])
            stress_list2 = np.array(stress_set_dict[up][1])
            stress_list3 = np.array(stress_set_dict[up][2])
            stress_list4 = np.array(stress_set_dict[up][3])
            #stress_list5 = np.array(stress_set_dict[up][4])

            eplisons_now = np.array([[s1[0][0], s1[1][1], s1[2][2], 2*s1[0][2], 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., s1[0][0], 0., 0., s1[1][1], s1[2][2], 2*s1[0][2], 0., 0., 0., 0., 0., 0.],
                                     [0., 0., s1[0][0], 0.,  0., s1[1][1], 0.,s1[2][2], 2*s1[0][2], 0., 0., 0., 0.],
                                     [0., 0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  2*s1[1][2],   2*s1[0][1],  0., 0.],
                                     [0., 0.,  0.,s1[0][0], 0.,  0.,s1[1][1],  0., s1[2][2],  0., 0.,2*s1[0][2],0.],
                                     [0., 0.,  0., 0.,  0., 0.,  0.,   0.,  0.,  0.,   2*s1[1][2], 0.,  2*s1[0][1]],
                                     [s2[0][0], s2[1][1], s2[2][2], 2*s2[0][2], 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., s2[0][0], 0., 0., s2[1][1], s2[2][2], 2*s2[0][2], 0., 0., 0., 0., 0., 0.],
                                     [0., 0., s2[0][0], 0.,  0., s2[1][1], 0.,s2[2][2], 2*s2[0][2], 0., 0., 0., 0.],
                                     [0., 0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  2*s2[1][2],   2*s2[0][1],  0., 0.],
                                     [0., 0.,  0.,s2[0][0], 0.,  0.,s2[1][1],  0., s2[2][2],  0., 0.,2*s2[0][2],0.],
                                     [0., 0.,  0., 0.,  0., 0.,  0.,   0.,  0.,  0.,   2*s2[1][2], 0.,  2*s2[0][1]],
                                     [s3[0][0], s3[1][1], s3[2][2], 2*s3[0][2], 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., s3[0][0], 0., 0., s3[1][1], s3[2][2], 2*s3[0][2], 0., 0., 0., 0., 0., 0.],
                                     [0., 0., s3[0][0], 0.,  0., s3[1][1], 0.,s3[2][2], 2*s3[0][2], 0., 0., 0., 0.],
                                     [0., 0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  2*s3[1][2],   2*s3[0][1],  0., 0.],
                                     [0., 0.,  0.,s3[0][0], 0.,  0.,s3[1][1],  0., s3[2][2],  0., 0.,2*s3[0][2],0.],
                                     [0., 0.,  0., 0.,  0., 0.,  0.,   0.,  0.,  0.,   2*s3[1][2], 0.,  2*s3[0][1]],
                                     [s4[0][0], s4[1][1], s4[2][2], 2*s4[0][2], 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., s4[0][0], 0., 0., s4[1][1], s4[2][2], 2*s4[0][2], 0., 0., 0., 0., 0., 0.],
                                     [0., 0., s4[0][0], 0.,  0., s4[1][1], 0.,s4[2][2], 2*s4[0][2], 0., 0., 0., 0.],
                                     [0., 0.,  0.,  0., 0.,  0.,  0.,  0.,  0.,  2*s4[1][2],   2*s4[0][1],  0., 0.],
                                     [0., 0.,  0.,s4[0][0], 0.,  0.,s4[1][1],  0., s4[2][2],  0., 0.,2*s4[0][2],0.],
                                     [0., 0.,  0., 0.,  0., 0.,  0.,   0.,  0.,  0.,   2*s4[1][2], 0.,  2*s4[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]],
                                     [stress_list2[0]],
                                     [stress_list2[1]],
                                     [stress_list2[2]],
                                     [stress_list2[3]],
                                     [stress_list2[4]],
                                     [stress_list2[5]],
                                     [stress_list3[0]],
                                     [stress_list3[1]],
                                     [stress_list3[2]],
                                     [stress_list3[3]],
                                     [stress_list3[4]],
                                     [stress_list3[5]],
                                     [stress_list4[0]],
                                     [stress_list4[1]],
                                     [stress_list4[2]],
                                     [stress_list4[3]],
                                     [stress_list4[4]],
                                     [stress_list4[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))
            
    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor
    c11 = cij[0][0]
    c12 = cij[1][0]
    c13 = cij[2][0]
    c15 = cij[3][0]
    c22 = cij[4][0]
    c23 = cij[5][0]
    c25 = cij[6][0]
    c33 = cij[7][0]
    c35 = cij[8][0]
    c44 = cij[9][0]
    c46 = cij[10][0]
    c55 = cij[11][0]
    c66 = cij[12][0]

    a = c33*c55-c35*c35
    b = c23*c55-c25*c35
    c = c13*c35-c15*c33
    d = c13*c55-c15*c35
    e = c13*c25-c15*c23
    f = c11*(c22*c55-c25*c25)-c12*(c12*c55-c15*c25)+c15*(c12*c25-c15*c22)+c25*(c23*c35-c25*c33)
    g = c11*c22*c33-c11*c23*c23-c22*c13*c13-c33*c12*c12+2*c12*c13*c23
    O = 2*(c15*c25*(c33*c12-c13*c23)+c15*c35*(c22*c13-c12*c23)+c25*c35*(c11*c23-c12*c13))-(c15*c15*(c22*c33-c23*c23)+c25*c25*(c11*c33-c13*c13)+c35*c35*(c11*c22-c12*c12))+g*c55

    B_v = (c11+c22+c33+2*(c12+c13+c23))/9.
    G_v = (c11+c22+c33+3*(c44+c55+c66)-(c12+c13+c23))/15.
    B_r = O/(a*(c11+c22-2*c12)+b*(2*c12-2*c11-c23)+c*(c15-2*c25)+d*(2*c12+2*c23-c13-2*c22)+2*e*(c25-c15)+f)
    G_r = 15/(4*(a*(c11+c22+c12)+b*(c11-c12-c23)+c*(c15+c25)+d*(c22-c12-c23-c13)+e*(c15-c25)+f)/O+3*(g/O+(c44+c66)/(c44*c66-c46*c46)))

    B_vrh = (B_v+B_r)/2.
    G_vrh = (G_v+G_r)/2.
    E = 9*B_vrh*G_vrh/(3*B_vrh+G_vrh)
    v = (3*B_vrh-2*G_vrh)/(2*(3*B_vrh+G_vrh))

    elastic_constants_dict['c11'] = c11
    elastic_constants_dict['c12'] = c12
    elastic_constants_dict['c13'] = c13
    elastic_constants_dict['c15'] = c15
    elastic_constants_dict['c22'] = c22
    elastic_constants_dict['c23'] = c23
    elastic_constants_dict['c25'] = c25
    elastic_constants_dict['c33'] = c33
    elastic_constants_dict['c35'] = c35
    elastic_constants_dict['c44'] = c44
    elastic_constants_dict['c46'] = c46
    elastic_constants_dict['c55'] = c55
    elastic_constants_dict['c66'] = c66

    elastic_constants_dict['B_v'] = B_v
    elastic_constants_dict['B_r'] = B_r
    elastic_constants_dict['G_v'] = G_v
    elastic_constants_dict['G_r'] = G_r
    elastic_constants_dict['B_vrh'] = B_vrh
    elastic_constants_dict['G_vrh'] = G_vrh
    elastic_constants_dict['E'] = E
    elastic_constants_dict['v'] = v


    return elastic_constants_dict


def Triclinic(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        if up == 0:
            s1 = strain_matrix(indict,latt_system, up)[0]
            stress_list1 = np.array(stress_set_dict[up][0])

            eplisons_now = np.array([[s1[0][0], s1[1][1], s1[2][2],2*s1[1][2],2*s1[0][2],2*s1[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., s1[0][0], 0., 0., 0., 0., s1[1][1], s1[2][2],2*s1[1][2],2*s1[0][2],2*s1[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0., s1[0][0], 0., 0., 0., 0., s1[1][1], 0., 0., 0., s1[2][2],2*s1[1][2],2*s1[0][2],2*s1[0][1], 0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., s1[0][0], 0., 0., 0., 0., s1[1][1], 0., 0., 0., s1[2][2], 0., 0.,2*s1[1][2],2*s1[0][2],2*s1[0][1], 0., 0., 0.],
                                     [0., 0., 0., 0., s1[0][0], 0., 0., 0., 0., s1[1][1], 0., 0., 0., s1[2][2], 0., 0.,2*s1[1][2], 0.,2*s1[0][2],2*s1[0][1], 0.],
                                     [0., 0., 0., 0., 0., s1[0][0], 0., 0., 0., 0., s1[1][1], 0., 0., 0., s1[2][2], 0.,0.,2*s1[1][2], 0.,2*s1[0][2], 2*s1[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]]])
        else:        
            s1 = strain_matrix(indict,latt_system, up)[0]
            s2 = strain_matrix(indict,latt_system, up)[1]
            s3 = strain_matrix(indict,latt_system, up)[2]
            s4 = strain_matrix(indict,latt_system, up)[3]
            s5 = strain_matrix(indict,latt_system, up)[4]
            s6 = strain_matrix(indict,latt_system, up)[5]
            stress_list1 = np.array(stress_set_dict[up][0])
            stress_list2 = np.array(stress_set_dict[up][1])
            stress_list3 = np.array(stress_set_dict[up][2])
            stress_list4 = np.array(stress_set_dict[up][3])
            stress_list5 = np.array(stress_set_dict[up][4])
            stress_list6 = np.array(stress_set_dict[up][5])

            eplisons_now = np.array([[s1[0][0], s1[1][1], s1[2][2],2*s1[1][2],2*s1[0][2],2*s1[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., s1[0][0], 0., 0., 0., 0., s1[1][1], s1[2][2],2*s1[1][2],2*s1[0][2],2*s1[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0., s1[0][0], 0., 0., 0., 0., s1[1][1], 0., 0., 0., s1[2][2],2*s1[1][2],2*s1[0][2],2*s1[0][1], 0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., s1[0][0], 0., 0., 0., 0., s1[1][1], 0., 0., 0., s1[2][2], 0., 0.,2*s1[1][2],2*s1[0][2],2*s1[0][1], 0., 0., 0.],
                                     [0., 0., 0., 0., s1[0][0], 0., 0., 0., 0., s1[1][1], 0., 0., 0., s1[2][2], 0., 0.,2*s1[1][2], 0.,2*s1[0][2],2*s1[0][1], 0.],
                                     [0., 0., 0., 0., 0., s1[0][0], 0., 0., 0., 0., s1[1][1], 0., 0., 0., s1[2][2], 0.,0.,2*s1[1][2], 0.,2*s1[0][2], 2*s1[0][1]],
                                     [s2[0][0], s2[1][1], s2[2][2],2*s2[1][2],2*s2[0][2],2*s2[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., s2[0][0], 0., 0., 0., 0., s2[1][1], s2[2][2],2*s2[1][2],2*s2[0][2],2*s2[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0., s2[0][0], 0., 0., 0., 0., s2[1][1], 0., 0., 0., s2[2][2],2*s2[1][2],2*s2[0][2],2*s2[0][1], 0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., s2[0][0], 0., 0., 0., 0., s2[1][1], 0., 0., 0., s2[2][2], 0., 0.,2*s2[1][2],2*s2[0][2],2*s2[0][1], 0., 0., 0.],
                                     [0., 0., 0., 0., s2[0][0], 0., 0., 0., 0., s2[1][1], 0., 0., 0., s2[2][2], 0., 0.,2*s2[1][2], 0.,2*s2[0][2],2*s2[0][1], 0.],
                                     [0., 0., 0., 0., 0., s2[0][0], 0., 0., 0., 0., s2[1][1], 0., 0., 0., s2[2][2], 0.,0.,2*s2[1][2], 0.,2*s2[0][2], 2*s2[0][1]],
                                     [s3[0][0], s3[1][1], s3[2][2],2*s3[1][2],2*s3[0][2],2*s3[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., s3[0][0], 0., 0., 0., 0., s3[1][1], s3[2][2],2*s3[1][2],2*s3[0][2],2*s3[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0., s3[0][0], 0., 0., 0., 0., s3[1][1], 0., 0., 0., s3[2][2],2*s3[1][2],2*s3[0][2],2*s3[0][1], 0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., s3[0][0], 0., 0., 0., 0., s3[1][1], 0., 0., 0., s3[2][2], 0., 0.,2*s3[1][2],2*s3[0][2],2*s3[0][1], 0., 0., 0.],
                                     [0., 0., 0., 0., s3[0][0], 0., 0., 0., 0., s3[1][1], 0., 0., 0., s3[2][2], 0., 0.,2*s3[1][2], 0.,2*s3[0][2],2*s3[0][1], 0.],
                                     [0., 0., 0., 0., 0., s3[0][0], 0., 0., 0., 0., s3[1][1], 0., 0., 0., s3[2][2], 0.,0.,2*s3[1][2], 0.,2*s3[0][2], 2*s3[0][1]],
                                     [s4[0][0], s4[1][1], s4[2][2],2*s4[1][2],2*s4[0][2],2*s4[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., s4[0][0], 0., 0., 0., 0., s4[1][1], s4[2][2],2*s4[1][2],2*s4[0][2],2*s4[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0., s4[0][0], 0., 0., 0., 0., s4[1][1], 0., 0., 0., s4[2][2],2*s4[1][2],2*s4[0][2],2*s4[0][1], 0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., s4[0][0], 0., 0., 0., 0., s4[1][1], 0., 0., 0., s4[2][2], 0., 0.,2*s4[1][2],2*s4[0][2],2*s4[0][1], 0., 0., 0.],
                                     [0., 0., 0., 0., s4[0][0], 0., 0., 0., 0., s4[1][1], 0., 0., 0., s4[2][2], 0., 0.,2*s4[1][2], 0.,2*s4[0][2],2*s4[0][1], 0.],
                                     [0., 0., 0., 0., 0., s4[0][0], 0., 0., 0., 0., s4[1][1], 0., 0., 0., s4[2][2], 0.,0.,2*s4[1][2], 0.,2*s4[0][2], 2*s4[0][1]],
                                     [s5[0][0], s5[1][1], s5[2][2],2*s5[1][2],2*s5[0][2],2*s5[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., s5[0][0], 0., 0., 0., 0., s5[1][1], s5[2][2],2*s5[1][2],2*s5[0][2],2*s5[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0., s5[0][0], 0., 0., 0., 0., s5[1][1], 0., 0., 0., s5[2][2],2*s5[1][2],2*s5[0][2],2*s5[0][1], 0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., s5[0][0], 0., 0., 0., 0., s5[1][1], 0., 0., 0., s5[2][2], 0., 0.,2*s5[1][2],2*s5[0][2],2*s5[0][1], 0., 0., 0.],
                                     [0., 0., 0., 0., s5[0][0], 0., 0., 0., 0., s5[1][1], 0., 0., 0., s5[2][2], 0., 0.,2*s5[1][2], 0.,2*s5[0][2],2*s5[0][1], 0.],
                                     [0., 0., 0., 0., 0., s5[0][0], 0., 0., 0., 0., s5[1][1], 0., 0., 0., s5[2][2], 0.,0.,2*s5[1][2], 0.,2*s5[0][2], 2*s5[0][1]],
                                     [s6[0][0], s6[1][1], s6[2][2],2*s6[1][2],2*s6[0][2],2*s6[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., s6[0][0], 0., 0., 0., 0., s6[1][1], s6[2][2],2*s6[1][2],2*s6[0][2],2*s6[0][1], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                     [0., 0., s6[0][0], 0., 0., 0., 0., s6[1][1], 0., 0., 0., s6[2][2],2*s6[1][2],2*s6[0][2],2*s6[0][1], 0., 0., 0., 0., 0., 0.],
                                     [0., 0., 0., s6[0][0], 0., 0., 0., 0., s6[1][1], 0., 0., 0., s6[2][2], 0., 0.,2*s6[1][2],2*s6[0][2],2*s6[0][1], 0., 0., 0.],
                                     [0., 0., 0., 0., s6[0][0], 0., 0., 0., 0., s6[1][1], 0., 0., 0., s6[2][2], 0., 0.,2*s6[1][2], 0.,2*s6[0][2],2*s6[0][1], 0.],
                                     [0., 0., 0., 0., 0., s6[0][0], 0., 0., 0., 0., s6[1][1], 0., 0., 0., s6[2][2], 0.,0.,2*s6[1][2], 0.,2*s6[0][2], 2*s6[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[2]],
                                     [stress_list1[3]],
                                     [stress_list1[4]],
                                     [stress_list1[5]],
                                     [stress_list2[0]],
                                     [stress_list2[1]],
                                     [stress_list2[2]],
                                     [stress_list2[3]],
                                     [stress_list2[4]],
                                     [stress_list2[5]],
                                     [stress_list3[0]],
                                     [stress_list3[1]],
                                     [stress_list3[2]],
                                     [stress_list3[3]],
                                     [stress_list3[4]],
                                     [stress_list3[5]],
                                     [stress_list4[0]],
                                     [stress_list4[1]],
                                     [stress_list4[2]],
                                     [stress_list4[3]],
                                     [stress_list4[4]],
                                     [stress_list4[5]],
                                     [stress_list5[0]],
                                     [stress_list5[1]],
                                     [stress_list5[2]],
                                     [stress_list5[3]],
                                     [stress_list5[4]],
                                     [stress_list5[5]],
                                     [stress_list6[0]],
                                     [stress_list6[1]],
                                     [stress_list6[2]],
                                     [stress_list6[3]],
                                     [stress_list6[4]],
                                     [stress_list6[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))
            
    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor

    c11 = cij[0][0]
    c12 = cij[1][0]
    c13 = cij[2][0]
    c14 = cij[3][0]
    c15 = cij[4][0]
    c16 = cij[5][0]
    c22 = cij[6][0]
    c23 = cij[7][0]
    c24 = cij[8][0]
    c25 = cij[9][0]
    c26 = cij[10][0]
    c33 = cij[11][0]
    c34 = cij[12][0]
    c35 = cij[13][0]
    c36 = cij[14][0]
    c44 = cij[15][0]
    c45 = cij[16][0]
    c46 = cij[17][0]
    c55 = cij[18][0]
    c56 = cij[19][0]
    c66 = cij[20][0]

    elastic_constants_dict['c11'] = c11
    elastic_constants_dict['c12'] = c12
    elastic_constants_dict['c13'] = c13
    elastic_constants_dict['c14'] = c14
    elastic_constants_dict['c15'] = c15
    elastic_constants_dict['c16'] = c16
    elastic_constants_dict['c22'] = c22
    elastic_constants_dict['c23'] = c23
    elastic_constants_dict['c24'] = c24
    elastic_constants_dict['c25'] = c25
    elastic_constants_dict['c26'] = c26
    elastic_constants_dict['c33'] = c33
    elastic_constants_dict['c34'] = c34
    elastic_constants_dict['c35'] = c35
    elastic_constants_dict['c36'] = c36
    elastic_constants_dict['c44'] = c44
    elastic_constants_dict['c45'] = c45
    elastic_constants_dict['c46'] = c46
    elastic_constants_dict['c55'] = c55
    elastic_constants_dict['c56'] = c56
    elastic_constants_dict['c66'] = c66

    return elastic_constants_dict


def isotropy(indict,pos_optimized, latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        s = strain_matrix(indict,latt_system, up)[0]
        #print(s)

        stress_list = np.array(stress_set_dict[up][0])

        eplisons_now = np.array([[s[0][0],   s[1][1]],
                                 [s[1][1],   s[0][0]],
                                 [s[0][1],  -s[0][1]]])

        stresses_now = np.array([[stress_list[0]],
                                 [stress_list[1]],
                                 [stress_list[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))

    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor
    lenth_angl = pos_optimized.get_cell_lengths_and_angles()

    elastic_constants_dict['c11'] = cij[0][0] * lenth_angl[2] / 10.
    elastic_constants_dict['c12'] = cij[1][0] * lenth_angl[2] / 10.

    return elastic_constants_dict


def tetragonal(indict,pos_optimized, latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        s = strain_matrix(indict,latt_system, up)[0]

        stress_list = np.array(stress_set_dict[up][0])

        eplisons_now = np.array([[s[0][0], s[1][1], 0.],
                                 [s[1][1], s[0][0], 0.],
                                 [0.,   0.,  2*s[0][1]]])

        stresses_now = np.array([[stress_list[0]],
                                 [stress_list[1]],
                                 [stress_list[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))

    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor
    lenth_angl = pos_optimized.get_cell_lengths_and_angles()

    elastic_constants_dict['c11'] = cij[0][0] * lenth_angl[2] / 10.
    elastic_constants_dict['c12'] = cij[1][0] * lenth_angl[2] / 10.
    elastic_constants_dict['c66'] = cij[2][0] * lenth_angl[2] / 10.

    return elastic_constants_dict


def orthotropy(indict,pos_optimized, latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        if up == 0:
            s1 = strain_matrix(indict,latt_system, up)[0]
            stress_list1 = np.array(stress_set_dict[up][0])
            eplisons_now = np.array([[s1[0][0], s1[1][1], 0., 0.],
                                     [0., s1[0][0], s1[1][1], 0.],
                                     [0.,   0.,  0.,  2*s1[0][1]]])
            
            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[5]]])
        else:
            s1 = strain_matrix(indict,latt_system, up)[0]
            s2 = strain_matrix(indict,latt_system, up)[1]

            stress_list1 = np.array(stress_set_dict[up][0])
            stress_list2 = np.array(stress_set_dict[up][1])

            eplisons_now = np.array([[s1[0][0], s1[1][1], 0., 0.],
                                     [0., s1[0][0], s1[1][1], 0.],
                                     [0.,   0.,  0.,  2*s1[0][1]],
                                     [s2[0][0], s2[1][1], 0., 0.],
                                     [0., s2[0][0], s2[1][1], 0.],
                                     [0.,   0.,  0.,  2*s2[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[5]],
                                     [stress_list2[0]],
                                     [stress_list2[1]],
                                     [stress_list2[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))

    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor
    lenth_angl = pos_optimized.get_cell_lengths_and_angles()

    elastic_constants_dict['c11'] = cij[0][0] * lenth_angl[2] / 10.
    elastic_constants_dict['c12'] = cij[1][0] * lenth_angl[2] / 10.
    elastic_constants_dict['c22'] = cij[2][0] * lenth_angl[2] / 10.
    elastic_constants_dict['c66'] = cij[3][0] * lenth_angl[2] / 10.

    return elastic_constants_dict


def anisotropy(indict,pos_optimized, latt_system, elastic_constants_dict, stress_set_dict, convertion_factor):
    for n, up in enumerate(stress_set_dict.keys()):
        if up == 0:
            s1 = strain_matrix(indict,latt_system, up)[0]
            stress_list1 = np.array(stress_set_dict[up][0])

            eplisons_now = np.array([[s1[0][0], s1[1][1], 2*s1[0][1], 0., 0., 0.],
                                     [0., s1[0][0], 0., s1[1][1], 2*s1[0][1], 0.],
                                     [0., 0., s1[0][0], 0., s1[1][1], 2*s1[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[5]]])
        else:
            s1 = strain_matrix(indict,latt_system, up)[0]
            s2 = strain_matrix(indict,latt_system, up)[1]
            s3 = strain_matrix(indict,latt_system, up)[2]

            stress_list1 = np.array(stress_set_dict[up][0])
            stress_list2 = np.array(stress_set_dict[up][1])
            stress_list3 = np.array(stress_set_dict[up][2])

            eplisons_now = np.array([[s1[0][0], s1[1][1], 2*s1[0][1], 0., 0., 0.],
                                     [0., s1[0][0], 0., s1[1][1], 2*s1[0][1], 0.],
                                     [0., 0., s1[0][0], 0., s1[1][1], 2*s1[0][1]],
                                     [s2[0][0], s2[1][1], 2*s2[0][1], 0., 0., 0.],
                                     [0., s2[0][0], 0., s2[1][1], 2*s2[0][1], 0.],
                                     [0., 0., s2[0][0], 0., s2[1][1], 2*s2[0][1]],
                                     [s3[0][0], s3[1][1], 2*s3[0][1], 0., 0., 0.],
                                     [0., s3[0][0], 0., s3[1][1], 2*s3[0][1], 0.],
                                     [0., 0., s3[0][0], 0., s3[1][1], 2*s3[0][1]]])

            stresses_now = np.array([[stress_list1[0]],
                                     [stress_list1[1]],
                                     [stress_list1[5]],
                                     [stress_list2[0]],
                                     [stress_list2[1]],
                                     [stress_list2[5]],
                                     [stress_list3[0]],
                                     [stress_list3[1]],
                                     [stress_list3[5]]])
        if n == 0:
            eplisons = copy.deepcopy(eplisons_now)
            stresses = copy.deepcopy(stresses_now)
        else:
            eplisons = np.vstack((eplisons, eplisons_now))
            stresses = np.vstack((stresses, stresses_now))

    cij = np.linalg.lstsq(eplisons, stresses, rcond=None)[0] * convertion_factor
    lenth_angl = pos_optimized.get_cell_lengths_and_angles()

    elastic_constants_dict['c11'] = cij[0][0] * lenth_angl[2] / 10.
    elastic_constants_dict['c12'] = cij[1][0] * lenth_angl[2] / 10.
    elastic_constants_dict['c16'] = cij[2][0] * lenth_angl[2] / 10.
    elastic_constants_dict['c22'] = cij[3][0] * lenth_angl[2] / 10.
    elastic_constants_dict['c26'] = cij[4][0] * lenth_angl[2] / 10.
    elastic_constants_dict['c66'] = cij[5][0] * lenth_angl[2] / 10.

    return elastic_constants_dict


def calc_elastic_constants(indict,pos_optimized, latt_system, elastic_constants_dict, stress_set_dict, convertion_factor=1):
    # kB to GPa
    # convertion_factor = -0.1
    strains_matrix = indict['strains_matrix'][0]
    if strains_matrix != 'asess':
        if latt_system == 'Cubic':
            elastic_constants_dict = Cubic(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)

        elif latt_system == 'Hexagonal':
            elastic_constants_dict = Hexagonal(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)
                
        elif latt_system == 'Trigonal1':
            elastic_constants_dict = Trigonal1(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)

        elif latt_system == 'Trigonal2':
            elastic_constants_dict = Trigonal2(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)

        elif latt_system == 'Tetragonal1':
            elastic_constants_dict = Tetragonal1(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)

        elif latt_system == 'Tetragonal2':
            elastic_constants_dict = Tetragonal2(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)

        elif latt_system == 'Orthorombic':
            elastic_constants_dict = Orthorombic(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)

        elif latt_system == 'Monoclinic':
            elastic_constants_dict = Monoclinic(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)
                
        elif latt_system == 'Triclinic':
            elastic_constants_dict = Triclinic(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)
        
        elif latt_system == 'isotropy':
            elastic_constants_dict = isotropy(indict,pos_optimized, latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)

        elif latt_system == 'tetragonal':
            elastic_constants_dict = tetragonal(indict,pos_optimized, latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)

        elif latt_system == 'orthotropy':
            elastic_constants_dict = orthotropy(indict,pos_optimized, latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)

        elif latt_system == 'anisotropy':
            elastic_constants_dict = anisotropy(indict,pos_optimized, latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)                        
        else:
            print('Crystal system is not parsed correctly!!!')
            exit(1)

    elif strains_matrix == 'asess':
        if indict['dimensional'][0] == '3D':
            elastic_constants_dict = Triclinic(indict,latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)
        elif indict['dimensional'][0] == '2D':
            elastic_constants_dict = anisotropy(indict,pos_optimized, latt_system, elastic_constants_dict, stress_set_dict, convertion_factor)                        

    return elastic_constants_dict


def stress_tens_to_voigt(tensor,conversion_ratio=-1.60218E2):
    #print(tensor)
    cr=conversion_ratio
    return np.array([tensor[0,0],tensor[1,1],tensor[2,2],tensor[1,2],tensor[0,1],tensor[0,2]])*cr

def stress_accom(strains,stress_flat):
    stress=[]
    c=0
    
    for i in range(len(strains)):
        stress.append([])
        for j in range(len(strains[i])):
            stress[i].append(stress_flat[c])
            c=c+1
    return stress

def stress_to_dict(lst_mag, strs_per_mag):
    dic={}
    dic[0] = strs_per_mag[0]
    for ni,i in enumerate(lst_mag):
        dic[i] = strs_per_mag[ni+1]
    return dic