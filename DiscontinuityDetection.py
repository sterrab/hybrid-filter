import numpy as np
from Multiwavelets import *
import matplotlib.pyplot as plt

def compute_data_multiwavelets(modes, modes_type='synthetic', N=128):
    
    num_samples = modes.shape[0]
    multiwavelets = np.zeros((num_samples, N))
    
    for sample in range(num_samples):

        # DG modes per sample 
        if modes_type == 'synthetic':
            p = len(modes[sample, 0]) - 1
            modes_sample = np.vstack(modes[sample, 1:-1]).reshape((N, p+1))
            modes_sample = modes_sample.T
        elif modes_type == 'cfd':
            modes_sample = np.array(modes[sample]).reshape((-1, N))
            p = modes_sample.shape[0]-1

        # Compute multiwavelets
        multiwavelets[sample, :] = calculate_multiwavelet_coeffs(modes_sample, p, N=N)

    return multiwavelets


def mask_thresholding_multiwavelets(multiwavelet_coeffs, threshold, N=128,  pts_per_cell=4, seal_cell_gap=1): 

    num_samples = multiwavelet_coeffs.shape[0]
    mask_thresh_mw = np.zeros((num_samples, N*pts_per_cell))
    troubled_cells_final = np.zeros(num_samples, dtype=object)
    for sample in range(num_samples): 

        # Interior cells identified as troubled: {0,..., N-1}
        troubled_cells = list(np.where(abs(multiwavelet_coeffs[sample, :]) > threshold * np.max(abs(multiwavelet_coeffs[sample, :])))[0])

        # Seal any gaps of size seal_cell_gap
        for flag in range(len(troubled_cells)-1): 
            if (troubled_cells[flag+1] - troubled_cells[flag] > 1) and (troubled_cells[flag+1] - troubled_cells[flag] <=seal_cell_gap+1): 
                troubled_cells.append(troubled_cells[flag]+1)
        
        troubled_cells_final[sample] = sorted(troubled_cells)

        # Generating pointwise Mask 
        for cell in troubled_cells:         
            mask_thresh_mw[sample, pts_per_cell*cell:pts_per_cell*(cell+1)]= np.ones(pts_per_cell)

    # Computing the Negative Mask: 1 -> 0, 0 -> 1
    not_mask_thresh_mw = np.logical_not(mask_thresh_mw).astype(int)

    return troubled_cells_final, mask_thresh_mw, not_mask_thresh_mw

def mask_boxplot_multiwavelets(multiwavelet_coeffs, N=128,  fold_len=32, whisker_len=3, pts_per_cell=4, bc='dirichlet', seal_cell_gap=1):

    if N < fold_len:
        fold_len = N

    num_folds = N//fold_len

    num_samples = multiwavelet_coeffs.shape[0]

    mask_boxplot_mw = np.zeros((num_samples, N*pts_per_cell))
    troubled_cells_final = np.zeros(num_samples, dtype=object)

    for sample in range(num_samples): 

        lower_bounds = np.zeros(num_folds + 2)
        upper_bounds = np.zeros(num_folds + 2)

        coeff_index = np.zeros((N, 2))
        for cell in range(N): 
            coeff_index[cell, :] = np.array([multiwavelet_coeffs[sample, cell], cell])     

        global_mean = np.mean(abs(multiwavelet_coeffs[sample, :]))

        sorted_folds = np.zeros((num_folds+2, fold_len+1, 2))

        for fold in range(num_folds):
            # Adding One Overlapped Cell between Adjacent Subdomains 
            if fold == num_folds -1: 
                final_coeffs = list(coeff_index[fold*fold_len:(fold+1)*fold_len])
                final_coeffs.append(coeff_index[fold*fold_len-1]) #Overlapped cell with left-adjacent subdomain
                sorted_coeffs = sorted(final_coeffs, key=lambda x: x[0])
                sorted_folds[fold+1, :, :] = np.array(sorted_coeffs)
            else: 
                sorted_coeffs = sorted(list(coeff_index[fold*fold_len:(fold+1)*fold_len+1]), key=lambda x: x[0])
                sorted_folds[fold+1, :, :] = np.array(sorted_coeffs)
            
            boundary_index = fold_len // 4
            balance_factor = fold_len / 4.0 - boundary_index

            first_quartile = (1 - balance_factor) * sorted_folds[fold + 1, boundary_index - 1, 0] \
                             + balance_factor * sorted_folds[fold + 1, boundary_index, 0]
            third_quartile = (1 - balance_factor) * sorted_folds[fold + 1, 3 * boundary_index - 1, 0] \
                             + balance_factor * sorted_folds[fold + 1, 3 * boundary_index, 0]

            lower_bound = first_quartile - whisker_len * (third_quartile - first_quartile)
            upper_bound = third_quartile + whisker_len * (third_quartile - first_quartile)

            # Adjusted Outer Fences method
            lower_bounds[fold + 1] = min(-global_mean, lower_bound)
            upper_bounds[fold + 1] = max(global_mean, upper_bound)

        # Ghost Local subdomains
        if bc == 'dirichlet':
            sorted_folds[0, :, :] = sorted_folds[1, :, :]
            sorted_folds[-1, :, :] = sorted_folds[-2, :, :]
            lower_bounds[0] = lower_bounds[1]
            upper_bounds[0] = upper_bounds[1]
            lower_bounds[-1] = lower_bounds[-2]
            upper_bounds[-1] = upper_bounds[-2]

        elif bc == 'periodic':
            sorted_folds[0, :, :] = sorted_folds[-2, :, :]
            sorted_folds[-1, :, :] = sorted_folds[1, :, :]
            lower_bounds[0] = lower_bounds[-2]
            upper_bounds[0] = upper_bounds[-2]
            lower_bounds[-1] = lower_bounds[1]
            upper_bounds[-1] = upper_bounds[1]

        troubled_cells = []
        for fold in range(num_folds):
            # Check for lower extreme outliers and add respective cells
            for cell in sorted_folds[fold+1, :, :]:
                if cell[0] < lower_bounds[fold + 1]:
                    if cell[0] < lower_bounds[fold] and cell[0] < lower_bounds[fold + 2]:
                        if int(cell[1]) not in troubled_cells:
                            troubled_cells.append(int(cell[1]))
                else:
                    break
            
            # Check for upper extreme outliers and add respective cells
            for cell in sorted_folds[fold + 1, ::-1, :]:
                if cell[0] > upper_bounds[fold + 1]:
                    if cell[0] > upper_bounds[fold] and cell[0] > upper_bounds[fold + 2]:
                        if int(cell[1]) not in troubled_cells:
                            troubled_cells.append(int(cell[1]))
                else:
                    break
        
        # Seal any gaps of size seal_cell_gap
        for flag in range(len(troubled_cells)-1): 
            if (troubled_cells[flag+1] - troubled_cells[flag] > 1) and (troubled_cells[flag+1] - troubled_cells[flag] <=seal_cell_gap+1): 
                troubled_cells.append(troubled_cells[flag]+1)
        
        troubled_cells_final[sample] = sorted(troubled_cells)

        # Generating pointwise Mask 
        for cell in troubled_cells:         
            mask_boxplot_mw[sample, pts_per_cell*cell:pts_per_cell*(cell+1)]= np.ones(pts_per_cell)

    # Computing the Negative Mask: 1 -> 0, 0 -> 1
    not_mask_boxplot_mw = np.logical_not(mask_boxplot_mw).astype(int)

    return troubled_cells_final, mask_boxplot_mw, not_mask_boxplot_mw


def center_tc_data(data, multiwavelets, exact, eval_pts, tcd_type='threshold', N=128, half_window=4, pts_per_cell=4, threshold=0.4, fold_len=32, whisker_len=3,  bc='dirichlet',  debug=False):
    centered_data = []
    exact_centered_data = []
    if tcd_type == 'threshold':
        troubled_cells, mask_thresh, not_mask_thresh = mask_thresholding_multiwavelets(multiwavelets, threshold=threshold, N=N, seal_cell_gap=0)
    elif tcd_type == 'boxplot':
        troubled_cells, mask_thresh, not_mask_thresh = mask_boxplot_multiwavelets(multiwavelets, fold_len=fold_len, N=N, whisker_len=whisker_len, bc=bc, seal_cell_gap=0)
    for sample in range(data.shape[0]):
        sample_tc = troubled_cells[sample]
        if sample_tc != []:
            if debug:
                for tc in sample_tc:
                    plt.figure(1)
                    plt.plot(eval_pts, data[sample, :])
                    plt.plot(eval_pts[tc*pts_per_cell], data[sample, tc*pts_per_cell], 'ro')
                    
        for tc in sample_tc:
            upper_bound = int(tc+half_window+1)*pts_per_cell 
            lower_bound = int(tc-half_window)*pts_per_cell 
            if (lower_bound >= 0 and upper_bound <= data.shape[1]):
                centered_data.append(data[sample, lower_bound:upper_bound])
                exact_centered_data.append(exact[sample, lower_bound:upper_bound])

    return np.array(centered_data), np.array(exact_centered_data) 


def normalize_clean_disc_data(disc_data, exact_disc_data, pts_per_cell=4, flat_val_threshold=0.01, max_val_threshold=0.1, disc_type='all'):

    # ---- Normalize data
    min_vals = np.min(disc_data, axis=1)
    max_vals = np.max(disc_data, axis=1)
    disc_data_norm = (disc_data - min_vals[:, np.newaxis])/(max_vals - min_vals)[:, np.newaxis]
    exact_disc_data_norm = (exact_disc_data - min_vals[:, np.newaxis])/(max_vals - min_vals)[:, np.newaxis]

    final_disc_data = []
    final_exact_disc_data = []
    min_vals_samples = []
    max_vals_samples = []
    sample_ids = []
    
    for sample in range(disc_data_norm.shape[0]):
        # ---- Clean data
        diff_values = np.diff(disc_data_norm[sample, :])
        left_diff = np.max(abs(diff_values[:int(2*pts_per_cell)]))
        right_diff = np.max(abs(diff_values[-int(2*pts_per_cell):]))
        max_diff = np.max(abs(diff_values))
        if disc_type == 'all':
            if (left_diff <= flat_val_threshold and right_diff <= flat_val_threshold and max_diff > max_val_threshold):
                final_disc_data.append(disc_data_norm[sample, :])
                final_exact_disc_data.append(exact_disc_data_norm[sample, :])
                min_vals_samples.append(min_vals[sample])
                max_vals_samples.append(max_vals[sample])
                sample_ids.append(sample)
        elif disc_type == 'shock':
            idx = np.where(abs(diff_values) == max_diff)[0]
            if diff_values[idx] < 0:
                if (left_diff <= flat_val_threshold and right_diff <= flat_val_threshold and max_diff > max_val_threshold):
                    final_disc_data.append(disc_data_norm[sample, :])
                    final_exact_disc_data.append(exact_disc_data_norm[sample, :])
                    min_vals_samples.append(min_vals[sample])
                    max_vals_samples.append(max_vals[sample])
                    sample_ids.append(sample)
        elif disc_type == 'contact':
            idx = np.where(abs(diff_values) == max_diff)[0]
            if diff_values[idx] > 0:
                if (left_diff <= flat_val_threshold and right_diff <= flat_val_threshold and max_diff > max_val_threshold):
                    final_disc_data.append(disc_data_norm[sample, :])
                    final_exact_disc_data.append(exact_disc_data_norm[sample, :])
                    min_vals_samples.append(min_vals[sample])
                    max_vals_samples.append(max_vals[sample])
                    sample_ids.append(sample)

    return np.array(final_disc_data), np.array(final_exact_disc_data), np.array(min_vals_samples), np.array(max_vals_samples), sample_ids


def find_filtering_windows(troubled_cells, gap=4, kernel_size_by_cell=2, N=128):
    if len(troubled_cells) == 1:
        return [troubled_cells], [np.arange(max(0, troubled_cells[0]-kernel_size_by_cell), min(troubled_cells[0]+kernel_size_by_cell+1, N+1))]
    else: 
        diff = np.diff(troubled_cells)
        ind_set = list(np.where(diff > gap)[0])
        if ind_set == []:
            tc_sets = [troubled_cells]
            # filtering windoes shifted by 1 to account for TCs in 0,..., N-1 range vs 1,..., N range in full data with ghost cells
            filtering_windows = [np.arange(max(1, troubled_cells[0]-kernel_size_by_cell), min(troubled_cells[-1]+kernel_size_by_cell+1, N))]

        else:
            tc_sets = []
            filtering_windows = []

            prev_ind = 0
            for i in range(len(ind_set)): 
                next_ind = ind_set[i]+1
                tc_subset = troubled_cells[prev_ind:next_ind]
                tc_sets.append(tc_subset)

                disc_window = np.arange(max(1, troubled_cells[prev_ind]-kernel_size_by_cell), min(troubled_cells[next_ind-1]+kernel_size_by_cell+1, N))
                filtering_windows.append(disc_window)

                prev_ind = next_ind

                if i == len(ind_set)-1:
                    tc_subset = troubled_cells[prev_ind:]
                    tc_sets.append(tc_subset)

                    disc_window = np.arange(troubled_cells[prev_ind]-kernel_size_by_cell, min(troubled_cells[-1]+kernel_size_by_cell+1, N))
                    filtering_windows.append(disc_window)
    
    return tc_sets, filtering_windows

