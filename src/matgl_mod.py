import os
import numpy as np
import matplotlib.pyplot as plt
import colorsys
from sklearn.model_selection import train_test_split
import pickle

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from matgl.ext.ase import Relaxer

from pymatgen.io import ase as pgase

ase_to_pmg = pgase.AseAtomsAdaptor.get_structure
pmg_to_ase = pgase.AseAtomsAdaptor.get_atoms


def obtain_sizes(list_structures):
    return [len(i) for i in list_structures]


def generate_distinct_colors(n):
    n = max(n, 1)
    hues = [i / n for i in range(n)]
    colors = [colorsys.hls_to_rgb(hue, 0.5, 1) for hue in hues]
    colors = [(int(r * 255), int(g * 255), int(b * 255)) 
              for (r, g, b) in colors]
    hex_colors = ['#%02x%02x%02x' % (r, g, b) for (r, g, b) in colors]
    return hex_colors


# def sanity_index(dft_rr):
#     res_index = []
#     strcts = []
#     for i in range(len(dft_rr)):
#         if len(dft_rr[i]) < 1:
#             print('HERE, empty: {}'.format(i))
#             continue
#         print(i)
#         if dft_rr[i][0]['energies'][-1] > 0.0:
#             print('HERE, last is positive: {}'.format(i))
#             continue
#         # else:
#         #     pass
#         res_index.append(i)
#         strcts.append(dft_rr[i][0]['trajectories'][0].copy())
        
#     return res_index, strcts


def getting_indset_first_with_n_frac(dft_rr: list, n_size, fraction,
                                     random_state=42):
    indexes = []
    sizes = []
    for i, ii in enumerate(dft_rr):
        if (len(dft_rr[i]) > 0) and (dft_rr[i][0] != "error") and \
                (len(dft_rr[i][0]) > 0):
            
            if dft_rr[i][0]['energies'][-1] > 0.0:
                print('HERE, last is positive: {}'.format(i))
                continue
            indexes.append(i)
            sizes.append(len(dft_rr[i][0]['trajectories'][0]))
        else:
            print('HERE, error/empty: {}'.format(i))
            continue

    uni = list(set(sizes))
    uni.sort()
    uni_i = []
    train = []
    test = []
    np.random.seed(random_state)
    for i in uni:
        iter_li = [index for index, item in enumerate(sizes) if item == i]
        uni_i.append(iter_li)
        if i >= n_size:
            sel_n = int(fraction*len(iter_li))
            s_li = np.random.choice(iter_li, sel_n, replace=False)
            s_li.sort()
            n_li = [i for i in iter_li if i not in s_li]
            train.extend(s_li)
            test.extend(n_li)
        else:
            train.extend(iter_li)
    train = np.array(indexes)[train]
    test = np.array(indexes)[test]
    return train, test


def getting_data_set_clean_and_index(dft_rr, sets_index, format='pmg'):
    energies = []
    forces = []
    stresses = []
    structures = []

    CONVERSION_to_ = -1.60218E2    
    
    len_sets = len(sets_index)
    len_full = len(dft_rr)

    index_sets = [[] for i in range(len_sets)]

    c_ = 0
    
    for i in range(len_full):

        if i not in np.concatenate(sets_index):
            continue
        ite_index = []
        
        num_ites = len(dft_rr[i][0]['energies'])
        print('STRUC index: {}, and size: {}'.format(i, num_ites))
        
        for j in range(num_ites):
            if len(dft_rr[i][0]['internal_energies'][j]) == 60:
                print('HERE (not converged) {},{}'.format(i, j))
                continue
            energies.append(dft_rr[i][0]['energies'][j])
            forces.append(dft_rr[i][0]['forces'][j].tolist())
            stresses.append((
                dft_rr[i][0]['stresses'][j]*CONVERSION_to_).tolist())
            if format == 'pmg':
                structures.append(ase_to_pmg(dft_rr[i][0]['trajectories'][j]))
            elif format == 'ase':
                structures.append(dft_rr[i][0]['trajectories'][j])

            ite_index.append(c_)
            c_ += 1
            
        for j, jj in enumerate(sets_index):
            if i in jj:
                index_sets[j].append(ite_index)
                
    return structures, {'energies': np.array(energies).tolist(), 
                        'forces': forces, 
                        'stresses': stresses}, index_sets


def get_lims(xs, ys, panf=0.05):
    
    h_ = np.append(xs, ys)
    mi, ma = np.min(h_), np.max(h_)
    pan = panf*(ma-mi)
    return mi-pan, ma+pan


def get_index_data_partition(dft_list,
                             train_partition=0.70, 
                             test_fraction=0.55,
                             test_start=3,
                             random_state=42):
    train_index, val_test_index = getting_indset_first_with_n_frac(
        dft_list, 
        test_start, 
        train_partition, 
        random_state=random_state)
    val_index, test_index, _, _ = train_test_split(val_test_index,
                                                   val_test_index, 
                                                   test_size=test_fraction, 
                                                   shuffle=True, 
                                                   random_state=random_state)
    val_index, test_index = np.sort(val_index), np.sort(test_index)
    return train_index, test_index, val_index


def get_static_data(dft_list, 
                    train_partition=0.70, 
                    test_fraction=0.55, 
                    test_start=3, 
                    random_state=42,
                    directory=""):
    train_index, val_index, test_index = get_index_data_partition(
        dft_list, 
        train_partition, 
        test_fraction, 
        test_start,
        random_state)

    train_set = [dft_list[i][0] for i in train_index]
    val_set = [dft_list[i][0] for i in val_index]
    test_set = [dft_list[i][0] for i in test_index]

    all_static_structures, \
        all_static_labels, \
        all_static_index_sets = getting_data_set_clean_and_index(
            dft_list, 
            [train_index, val_index, test_index])
    train_static_index_set, \
        val_static_index_set, \
        test_static_index_set = all_static_index_sets
    train_static_index = [
        i for sublist in train_static_index_set for i in sublist]
    val_static_index = [
        i for sublist in val_static_index_set for i in sublist]
    test_static_index = [
        i for sublist in test_static_index_set for i in sublist]
    print('train size {} relaxations, {} static'.format(
          len(train_index), len(train_static_index)))
    print('val size {} relaxations, {} static'.format(
          len(val_index), len(val_static_index)))
    print('test size {} relaxations, {} static'.format(
          len(test_index), len(test_static_index)))
    train_structures = [all_static_structures[i] for i in train_static_index]
    val_structures = [all_static_structures[i] for i in val_static_index]
    test_structures = [all_static_structures[i] for i in test_static_index]

    train_labels = [[all_static_labels['energies'][i], 
                    all_static_labels['forces'][i], 
                    all_static_labels['stresses'][i]
                     ] for i in train_static_index]
    val_labels = [[all_static_labels['energies'][i], 
                   all_static_labels['forces'][i], 
                   all_static_labels['stresses'][i]
                   ] for i in val_static_index]
    test_labels = [[all_static_labels['energies'][i], 
                    all_static_labels['forces'][i], 
                    all_static_labels['stresses'][i]
                    ] for i in test_static_index]

    if directory == "":
        print()
    else:
        os.makedirs(directory, exist_ok=True)
        # with open('{}/train_structures.pkl'.format(directory), 'wb') as f:
        #     pickle.dump(train_structures, f)
        # with open('{}/val_structures.pkl'.format(directory), 'wb') as f:
        #     pickle.dump(val_structures, f)
        # with open('{}/test_structures.pkl'.format(directory), 'wb') as f:
        #     pickle.dump(test_structures, f)
        # with open('{}/train_labels.pkl'.format(directory), 'wb') as f:
        #     pickle.dump(train_labels, f)
        # with open('{}/val_labels.pkl'.format(directory), 'wb') as f:
        #     pickle.dump(val_labels, f)
        # with open('{}/test_labels.pkl'.format(directory), 'wb') as f:
        #     pickle.dump(test_labels, f)
        with open('{}/train_set.pkl'.format(directory), 'wb') as f:
            pickle.dump(train_set, f)
        with open('{}/val_set.pkl'.format(directory), 'wb') as f:
            pickle.dump(val_set, f)
        with open('{}/test_set.pkl'.format(directory), 'wb') as f:
            pickle.dump(test_set, f)
        with open('{}/all_static_structures.pkl'.format(directory), 'wb') as f:
            pickle.dump(all_static_structures, f)
        with open('{}/all_static_labels.pkl'.format(directory), 'wb') as f:
            pickle.dump(all_static_labels, f)
        with open('{}/train_static_index.pkl'.format(directory), 'wb') as f:
            pickle.dump(train_static_index, f)
        with open('{}/val_static_index.pkl'.format(directory), 'wb') as f:
            pickle.dump(val_static_index, f)
        with open('{}/test_static_index.pkl'.format(directory), 'wb') as f:
            pickle.dump(test_static_index, f)
    return [train_structures, train_labels], [
        val_structures, val_labels], [
        test_structures, test_labels]


def parity_plot_Static(arr_list, ase_sizes, units, per_atom,
                       size_marker=10,
                       alpha=0.4, 
                       color_sized=True, 
                       color_unique='red', 
                       edgecolors='white', 
                       savefig=False,
                       savefile_name='output', 
                       format='svg',
                       fontsize=14):

    no_plots = len(arr_list)
    unique_sizes = list(set(ase_sizes))
    unique_sizes.sort()
    color_sizes = generate_distinct_colors(len(unique_sizes))
    
    if color_sized:
        colors = []
        for i in range(no_plots):
            colors.append([])
            if per_atom[i] == 0:
                colors[i] = [color_sizes[unique_sizes.index(j)] 
                             for j in ase_sizes]
            if per_atom[i] == 3:
                dummy_list = [[color_sizes[unique_sizes.index(j)]]*3*j 
                              for j in ase_sizes]
                colors[i] = np.concatenate(dummy_list)
            if per_atom[i] == -6:
                dummy_list = [[color_sizes[unique_sizes.index(j)]]*6 
                              for j in ase_sizes]
                colors[i] = np.concatenate(dummy_list)
            
            # dummy_list = [color_sizes[unique_sizes.index(i)] 
            # for i in ase_sizes]
            # colors[i] = np.concatenate(dummy_list)
    else:
        colors = [color_unique for i in range(no_plots)]

    fig, axs = plt.subplots(1, no_plots, figsize=(5*no_plots, 5)) 
    
    for i in range(no_plots):
        axs[i].set_axisbelow(True)
        axs[i].grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
        ii = arr_list[i][0]
        jj = arr_list[i][1]
        axs[i].scatter(ii, jj, edgecolors=edgecolors, linewidth=0.5, 
                       s=size_marker, color=colors[i], 
                       alpha=alpha, rasterized=True)

        mi, ma = get_lims(ii, jj)
        axs[i].set_xlim(mi, ma)
        axs[i].set_ylim(mi, ma)
        
        axs[i].annotate(r'$r^2=$'+'{:.4f}'.format(r2_score(ii, jj)),
                        xy=(0.1, 0.9), xycoords='axes fraction',
                        fontsize=fontsize)
        axs[i].annotate(r'$RMSE=$'+'{:.4f} {}'.format(np.sqrt(
                        mean_squared_error(ii, jj)), units[i]),
                        xy=(0.1, 0.84), xycoords='axes fraction',
                        fontsize=fontsize)
        axs[i].annotate(r'$MAE=$'+'{:.4f} {}'.format(
                        mean_absolute_error(ii, jj), units[i]),
                        xy=(0.1, 0.78), xycoords='axes fraction',
                        fontsize=fontsize)
    axs[0].set_ylabel(r'M3Gnet Energy per atom ({})'.format(units[0]))
    axs[1].set_ylabel(r'M3Gnet Forces ({})'.format(units[1]))
    axs[2].set_ylabel(r'M3Gnet Stresses ({})'.format(units[2]))

    axs[0].set_xlabel(r'DFT Energy per atom ({})'.format(units[0]))
    axs[1].set_xlabel(r'DFT Forces ({})'.format(units[1]))
    axs[2].set_xlabel(r'DFT Stresses ({})'.format(units[2]))

    if color_sized:
        custom_legend = [
            plt.scatter([-999], [-999], color=color_sizes[i], 
                        edgecolors=edgecolors, linewidth=0.5,
                        label='{} atom size'.format(
                            unique_sizes[i])) for i, _ in enumerate(
                                unique_sizes)]
        # Add the legend to the plot
        axs[2].legend(handles=custom_legend, loc='lower right', 
                      fontsize=fontsize)

    plt.subplots_adjust(wspace=0.25)
    if savefig:
        
        fig.savefig('{}.{}'.format(savefile_name, format), format=format, 
                    bbox_inches='tight')


def validation_static(list_static, calculator, plot_bool=True):
    
    static_structs = list_static[0]
    static_labels = list_static[1]
    
    res_arr = [[[], []] for i in range(3)]
    ase_sizes = []
    
    for i, i_ase in enumerate(static_structs):
        
        ite_ase = pmg_to_ase(i_ase)
        ite_ase.calc = calculator

        ase_sizes.append(len(i_ase))
        res_arr[0][0].append(static_labels[i][0] / len(i_ase))
        res_arr[0][1].append(ite_ase.get_potential_energy() / len(i_ase))
        
        res_arr[1][0].extend(np.array(static_labels[i][1]).ravel())
        res_arr[1][1].extend(ite_ase.get_forces().ravel())
        
        stress_pred_ite = np.array(static_labels[i][2])
        
        res_arr[2][0].extend([stress_pred_ite[0, 0], stress_pred_ite[1, 1], 
                              stress_pred_ite[2, 2], stress_pred_ite[1, 2], 
                              stress_pred_ite[0, 2], stress_pred_ite[0, 1]])
        res_arr[2][1].extend(ite_ase.get_stress())
    
    units = [r'eV/atom', r'eV/$\AA$', r'GPa']
    per_atom = [0, 3, -6]
    
    for i in range(len(res_arr)):
        
        ii = np.array(res_arr[i][0]).reshape(1, -1)[0]
        jj = np.array(res_arr[i][1]).reshape(1, -1)[0]
        print(r'r2 = {:.4f}'.format(r2_score(ii, jj)))
        print(r'RMSE = {:.4f} {}'.format(np.sqrt(
            mean_squared_error(ii, jj)), units[i]))
        print(r'MAE = {:.4f} {}'.format(mean_absolute_error(ii, jj), units[i]))

    if plot_bool:
        parity_plot_Static(res_arr, ase_sizes, units, per_atom)
        
    # return res_arr, ase_sizes


def parity_plot_Relax(arr_list, ase_sizes, units, per_atom,
                      size_marker=10,
                      alpha=0.4, 
                      color_sized=True, 
                      color_unique='red', 
                      edgecolors='white', 
                      savefig=False,
                      savefile_name='output', 
                      format='svg',
                      fontsize=14):

    no_plots = len(arr_list)
    unique_sizes = list(set(ase_sizes))
    unique_sizes.sort()
    color_sizes = generate_distinct_colors(len(unique_sizes))
    
    if color_sized:
        colors = []
        for i in range(no_plots):
            colors.append([])
            if per_atom[i] == 0:
                colors[i] = [color_sizes[unique_sizes.index(j)] 
                             for j in ase_sizes]
            if per_atom[i] == 3:
                dummy_list = [[color_sizes[unique_sizes.index(j)]]*3*j 
                              for j in ase_sizes]
                colors[i] = np.concatenate(dummy_list)
            if per_atom[i] == -6:
                dummy_list = [[color_sizes[unique_sizes.index(j)]]*6 
                              for j in ase_sizes]
                colors[i] = np.concatenate(dummy_list)
            
            # dummy_list = [color_sizes[unique_sizes.index(i)] 
            # for i in ase_sizes]
            # colors[i] = np.concatenate(dummy_list)
    else:
        colors = [color_unique for i in range(no_plots)]

    fig, axs = plt.subplots(1, no_plots, figsize=(5*no_plots, 5)) 
    
    for i in range(no_plots):
        axs[i].set_axisbelow(True)
        axs[i].grid(True, linestyle='--', alpha=0.5, linewidth=1.5)
        ii = arr_list[i][0]
        jj = arr_list[i][1]
        axs[i].scatter(ii, jj, edgecolors=edgecolors, linewidth=0.5, 
                       s=size_marker, color=colors[i], 
                       alpha=alpha, rasterized=True)

        mi, ma = get_lims(ii, jj)
        axs[i].set_xlim(mi, ma)
        axs[i].set_ylim(mi, ma)
        
        axs[i].annotate(r'$r^2=$'+'{:.4f}'.format(r2_score(ii, jj)),
                        xy=(0.1, 0.9), xycoords='axes fraction',
                        fontsize=fontsize)
        axs[i].annotate(r'$RMSE=$'+'{:.4f} {}'.format(
                        np.sqrt(mean_squared_error(ii, jj)), units[i]),
                        xy=(0.1, 0.84), xycoords='axes fraction', 
                        fontsize=fontsize)
        axs[i].annotate(r'$MAE=$'+'{:.4f} {}'.format(
                        mean_absolute_error(ii, jj), units[i]),
                        xy=(0.1, 0.78), xycoords='axes fraction',
                        fontsize=fontsize)
    axs[0].set_ylabel(r'M3Gnet Energy per atom ({})'.format(units[0]))
    axs[1].set_ylabel(r'M3Gnet Volume ({})'.format(units[1]))
    
    axs[0].set_xlabel(r'DFT Energy per atom ({})'.format(units[0]))
    axs[1].set_xlabel(r'DFT Volume ({})'.format(units[1]))
    
    if color_sized:
        custom_legend = [
            plt.scatter([-999], [-999], color=color_sizes[i], 
                        edgecolors=edgecolors, linewidth=0.5,
                        label='{} atom size'.format(
                            unique_sizes[i])) for i, _ in enumerate(
                            unique_sizes)]
        # Add the legend to the plot
        axs[1].legend(handles=custom_legend, loc='lower right', 
                      fontsize=fontsize)

    plt.subplots_adjust(wspace=0.25)
    if savefig:
        
        fig.savefig('{}.{}'.format(savefile_name, format), 
                    format=format, bbox_inches='tight')


def validation_relax(list_set, calculator, plot_bool=True):
    
    relax_structs = [i['trajectories'][0] for i in list_set]
    
    res_arr = [[[], []] for i in range(2)]
    ase_sizes = []
    relaxer = Relaxer(calculator.potential)
    
    for i, i_ase in enumerate(relax_structs):
        
        ase_sizes.append(len(i_ase))    
        res_it = relaxer.relax(i_ase)
        
        res_arr[0][0].append(list_set[i]['energies'][-1] / ase_sizes[-1])
        res_arr[0][1].append(res_it["trajectory"].energies[-1]/ase_sizes[-1])

        res_arr[1][0].append(
            list_set[i]['trajectories'][-1].get_volume()/ase_sizes[-1])
        cell = res_it["trajectory"].cells[-1]
        res_arr[1][1].append(
            np.abs(np.dot(cell[0], np.cross(cell[1], cell[2])))/ase_sizes[-1])
    
    units = [r'eV/atom', r'$\AA^3$',]
    per_atom = [0, 0,]
    
    for i in range(len(res_arr)):
        
        ii = np.array(res_arr[i][0]).reshape(1, -1)[0]
        jj = np.array(res_arr[i][1]).reshape(1, -1)[0]
        print(r'r2 = {:.4f}'.format(r2_score(ii, jj)))
        print(r'RMSE = {:.4f} {}'.format(
            np.sqrt(mean_squared_error(ii, jj)), units[i]))
        print(r'MAE = {:.4f} {}'.format(mean_absolute_error(ii, jj), units[i]))

    if plot_bool:
        parity_plot_Relax(res_arr, ase_sizes, units, per_atom)
        
    # # return res_arr, ase_sizes


def get_static_data_from_all(all_static_structures, all_static_labels, 
                             train_static_index,
                             val_static_index,
                             test_static_index):
    train_structures = [all_static_structures[i] for i in train_static_index]
    val_structures = [all_static_structures[i] for i in val_static_index]
    test_structures = [all_static_structures[i] for i in test_static_index]
    
    train_labels = [[all_static_labels['energies'][i], 
                    all_static_labels['forces'][i], 
                    all_static_labels['stresses'][i]
                     ] for i in train_static_index]
    val_labels = [[all_static_labels['energies'][i], 
                   all_static_labels['forces'][i], 
                   all_static_labels['stresses'][i]
                   ] for i in val_static_index]
    test_labels = [[all_static_labels['energies'][i], 
                    all_static_labels['forces'][i], 
                    all_static_labels['stresses'][i]
                    ] for i in test_static_index]
    return [train_structures, train_labels], [
        val_structures, val_labels], [
        test_structures, test_labels]
