import numpy as np
import scipy as sp
from scipy.special import eval_legendre
import matplotlib.pyplot as plt
import time
import seaborn as sns

sns.set()
sns.set(rc={'figure.figsize':(6,4)})
sns.color_palette('bright')
sns.set_style("white")
custom_params = {"axes.spines.right": False, "axes.spines.top": False}
sns.set_style("ticks", rc=custom_params)
sns.set_context("talk")

from Grid import *
from InitialConditions import *

# # ------ Define GLOBAL VARIABLES  ------------------
CFL_DICT = {
    0:  0.4,
    1:  0.4,
    2:  0.2,
    3:  0.13,
    4:  0.089
    } 

ADVECTION_SPEED = 1
X_LEFT = -5
X_RIGHT = 5

MAX_DEGREE = 4
PTS_PER_ELEMENT = 4

# Set stiffness matrix
STIFFNESS_MATRIX = []
for degree in range(MAX_DEGREE + 1): 
    matrix = [] 
    for i in range(degree + 1):
        new_row = []
        for j in range(degree + 1):
            new_entry = -1.0
            if (j < i) & ((i + j) % 2 == 1):
                new_entry = 1.0
            new_row.append(new_entry * np.sqrt((i + 0.5) * (j + 0.5)))
        matrix.append(new_row)
    STIFFNESS_MATRIX.append(matrix)
STIFFNESS_MATRIX = np.array(STIFFNESS_MATRIX, dtype=object)  

# Set boundary matrix
BOUNDARY_MATRIX = []
for degree in range(MAX_DEGREE + 1):
    matrix = []
    for i in range(degree + 1):
        new_row = []
        for j in range(degree + 1):
            new_entry = np.sqrt((i + 0.5) * (j + 0.5)) * (-1.0) ** i
            new_row.append(new_entry)
        matrix.append(new_row)
    BOUNDARY_MATRIX.append(matrix)
BOUNDARY_MATRIX = np.array(BOUNDARY_MATRIX, dtype=object)  


# Compute QMF scaling matrix 
QMF_H0=[]
QMF_H1=[]
for degree in range(MAX_DEGREE + 1):
    # Guass-Legendre Quadrature Nodes and Weights
    num_of_eval_pts= degree+1
    xi_eval, weights_eval = np.polynomial.legendre.leggauss(num_of_eval_pts)

    matrix0 = []
    matrix1 = []
    for i in range(degree + 1):
        row0 = []
        row1 = []
        for j in range(degree + 1):
            entry0 = 1/np.sqrt(2)*sum(weights_eval*(eval_legendre(j, xi_eval)* np.sqrt(j+0.5)*eval_legendre(i, 0.5 * (xi_eval - np.ones(num_of_eval_pts)))* np.sqrt(i+0.5)))
            entry1 = 1/np.sqrt(2)*sum(weights_eval*(eval_legendre(j, xi_eval)* np.sqrt(j+0.5)*eval_legendre(i, 0.5 * (xi_eval + np.ones(num_of_eval_pts)))* np.sqrt(i+0.5)))
            row0.append(np.float64(entry0))
            row1.append(np.float64(entry1))
        matrix0.append(row0)
        matrix1.append(row1)
    QMF_H0.append(matrix0)
    QMF_H1.append(matrix1)
QMF_H0 = np.array(QMF_H0, dtype=object)  
QMF_H1 = np.array(QMF_H1, dtype=object)  

# # ---------------DEFINE KEY Functions ----------------

def generateData(num_samples, wavespeed=1, grid_level_lo=6, grid_level_hi=6, degree_lo=1, degree_hi=MAX_DEGREE,  uniformgrid_ratio=1, nonsmooth_ratio=1, data_type='quadrature', ic = 'any', show_plot=False, save_modal=False): 
    data = []
    exact = []
    data_other = []
    exact_other = []
    modes = []
    for sample in range(num_samples): 
        start_time = time.time()
        
        # Create mesh 
        grid, grid_spacing = generateGrid(grid_level_lo, grid_level_hi, uniformgrid_ratio, x_left=X_LEFT, x_right=X_RIGHT)

        # Select initial condition with given parameters
        bias = np.random.uniform(low=0.1, high=0.5)
        coeff = np.random.uniform(low=0.1, high=1)
        if bias - coeff < 0: 
            coeff = bias
        power = 1
        jump = np.random.uniform(low=0.1, high=1)
        freq = np.random.randint(2,5)
        amplitude = np.random.uniform(low=0.2, high=0.6)
        params = {'bias': bias, 'coeff': coeff, 'power': power, 'jump': jump, 'freq': freq, 'amplitude': amplitude}

        random_draw = np.random.random_sample()
        if random_draw < nonsmooth_ratio: 
            initial_condition = nonSmooth(params, ic=ic)
        else: 
            initial_condition = smooth(params, ic=ic)

        # Generate random parameters
        degree = np.random.randint(degree_lo, high=degree_hi+1)
        final_time = np.random.uniform(low=2.1, high=9.5)

        # Generate modes 
        initial_modes = computeInitialModes(degree, grid, grid_spacing, initial_condition)
        
        # Generate data 
        a =  wavespeed
        if data_type == 'both': 
            evaluation_pts_initial, initial_approx = computeApprox(initial_modes, grid, grid_spacing, data_type='even')
            evolved_modes = evolveModes(initial_modes, grid_spacing, final_time, a)
            evaluation_pts, approx = computeApprox(evolved_modes, grid, grid_spacing, data_type='even')
            exact_soln = computeExact(evaluation_pts, initial_condition, final_time, a)
            evaluation_pts_quad, approx_quad = computeApprox(evolved_modes, grid, grid_spacing, data_type='quadrature')
            exact_quad = computeExact(evaluation_pts_quad, initial_condition, final_time, a)
            
            # Add to data files
            data.append(approx)
            exact.append(exact_soln)
            modes.append(list(evolved_modes))
            data_other.append(approx_quad)
            exact_other.append(exact_quad)

        else:
            evaluation_pts_initial, initial_approx = computeApprox(initial_modes, grid, grid_spacing, data_type=data_type)
            evolved_modes = evolveModes(initial_modes, grid_spacing, final_time, a)
            evaluation_pts, approx = computeApprox(evolved_modes, grid, grid_spacing, data_type=data_type)
            exact_soln = computeExact(evaluation_pts, initial_condition, final_time, a)

            # Add to data files
            data.append(approx)
            exact.append(exact_soln)
            modes.append(list(evolved_modes))

        if show_plot: 
            plt.plot(evaluation_pts, exact_soln[0,:], '-k', alpha=1, linewidth=5,label='Exact')
            plt.plot(evaluation_pts, approx[0,:], '-', color='0.55', alpha=0.9, linewidth=5, label='Unfiltered')
            plt.plot(evaluation_pts_initial, initial_approx[0,:], ':', color='0.3', alpha=1, linewidth=2, label='Initial Condition')
            plt.legend()
            ax = plt.gca()
            sns.move_legend(
                ax, "lower center",
                bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False,
            )
            # plt.savefig('final_plots/IC00'+str(sample)+'.pdf', bbox_inches='tight', pad_inches=0.15)
            plt.show()

        end_time = time.time()
        print('Sample: ', sample, '| Degree: ', degree, '| Wavespeed: ', a, ' T_f:', final_time,  '| Time per sample: ', end_time-start_time)

    data = np.array(data)
    exact = np.array(exact)
    modes = np.array(modes, dtype=object)
    if data_type =='both': 
        data_other = np.array(data_other)
        exact_other = np.array(exact_other)

    if save_modal: 
        if data_type == 'both': 
            return data, exact, evaluation_pts, data_other, exact_other, evaluation_pts_quad, modes, grid, grid_spacing
        else: 
            return data, exact, evaluation_pts, modes, grid, grid_spacing
    else: 
        if data_type == 'both':
            return data, exact, evaluation_pts, data_other, exact_other, evaluation_pts_quad
        else: 
            return data, exact, evaluation_pts
        


def splitData(data, exact, train_ratio, validation_ratio): 
    # random shuffle of the data
    num_samples = data.shape[0]
    order = np.random.permutation(num_samples)
    data = data[order]
    exact = exact[order]

    # splitting the data 
    train_sample_cutoff = int(train_ratio*num_samples)
    validation_sample_cutoff = train_sample_cutoff + int(validation_ratio*num_samples)

    data_train = data[:train_sample_cutoff]
    exact_train = exact[:train_sample_cutoff]

    data_validation = data[train_sample_cutoff:validation_sample_cutoff]
    exact_validation = exact[train_sample_cutoff:validation_sample_cutoff]

    data_test = data[validation_sample_cutoff:]
    exact_test = exact[validation_sample_cutoff:]

    return data_train, exact_train, data_validation, exact_validation, data_test, exact_test


def computeInitialModes(degree, grid, grid_spacing, initial_condition):
    # Guass-Legendre Quadrature Nodes and Weights
    num_of_eval_pts= degree+1
    xi_eval, weights_eval = np.polynomial.legendre.leggauss(num_of_eval_pts)
    
    # Compute the evaluation points 
    N = len(grid)
    nodes_eval = np.zeros(N*num_of_eval_pts)
    for cell in range(N): 
        cell_center = grid[cell] + 0.5*grid_spacing[cell]
        # Local quadrature nodes:
        nodes_eval[cell*num_of_eval_pts: (cell+1)*num_of_eval_pts] = 0.5*grid_spacing[cell]*xi_eval + cell_center*np.ones(num_of_eval_pts)

    # Evaluate initial condition at all nodes at once
    initcond_eval = initial_condition(nodes_eval)

    # Compute the (N+2) x (degree + 1) L2-Projection modes matrix, includes ghost cells
    modes = np.zeros((N+2, degree + 1))
    for cell in range(1, N+1): # internal cells only, not ghost cells
        for k in range(degree + 1):
            integrand = initcond_eval[(cell-1)*num_of_eval_pts: cell*num_of_eval_pts] *eval_legendre(k, xi_eval)* np.sqrt(k+0.5)
            modes[cell, k]= sum(integrand*weights_eval)

    # Apply periodic boundary conditions to ghost cells
    modes[0, :] = modes[N, :]
    modes[N+1, :] = modes[1, :]

    return modes 



def evolveModes(initial_modes, grid_spacing, final_time, a=1): 
    degree = initial_modes.shape[1]-1
    cfl = CFL_DICT[degree]
    h = min(grid_spacing)
    time_step = abs(cfl * h / a)

    # Begin time iterations
    evolved_modes = initial_modes
    current_time = 0
    while current_time < final_time:
        # Adjust for final time-step
        if current_time+time_step > final_time:
            time_step = final_time - current_time
            cfl = abs(a * time_step / h)

        # Update modes
        evolved_modes = timeStep(evolved_modes, cfl)

        current_time += time_step

    return evolved_modes

def evolveModesNumSteps(initial_modes, grid_spacing, max_iterations, a=1): 
    degree = initial_modes.shape[1]-1
    cfl = CFL_DICT[degree]
    h = min(grid_spacing)
    time_step = abs(cfl * h / a)

    # Begin time iterations
    evolved_modes = initial_modes
    current_time = 0
    iteration = 0
    while iteration < max_iterations:
        # Update modes
        evolved_modes = timeStep(evolved_modes, cfl)
        iteration += 1
        current_time += time_step

    return evolved_modes, current_time


def timeStep(modes, cfl_number):
    modes = enforce_boundary_condition(modes)
    original_modes = modes

    current_modes = apply_first_step(original_modes, cfl_number)
    current_modes = enforce_boundary_condition(current_modes)

    current_modes = apply_second_step(original_modes, current_modes, cfl_number)
    current_modes = enforce_boundary_condition(current_modes)

    current_modes = apply_third_step(original_modes, current_modes, cfl_number)
    current_modes = enforce_boundary_condition(current_modes)

    return current_modes

def enforce_boundary_condition(modes): 
    new_modes = modes.copy()

    # Apply periodic boundary conditions 
    new_modes[0, :] = new_modes[-2, :]
    new_modes[-1, :] = new_modes[1, :]

    return new_modes

def apply_first_step(original_modes, cfl_number): 
    right_hand_side = update_right_hand_side(original_modes)
    return original_modes + (cfl_number * right_hand_side)

def apply_second_step(original_modes, current_modes, cfl_number):
    right_hand_side = update_right_hand_side(current_modes)
    return 1 / 4 * (3 * original_modes + (current_modes + cfl_number * right_hand_side))

def apply_third_step(original_modes, current_modes, cfl_number):  
    right_hand_side = update_right_hand_side(current_modes)
    return 1 / 3 * (original_modes + 2 * (current_modes + cfl_number * right_hand_side))

def update_right_hand_side(modes): 
    degree = modes.shape[1]-1
    N = modes.shape[0]-2

    # Stiffness and Boundary matrices for given degree
    stiffness_matrix = np.array(STIFFNESS_MATRIX[degree]).reshape((degree+1,degree+1))
    boundary_matrix = np.array(BOUNDARY_MATRIX[degree]).reshape((degree+1,degree+1))

    # Computing right-hand side
    rhs = 0*modes
    for cell in range(1, N+1): 
        rhs[cell, :] = 2 *(stiffness_matrix @ modes[cell, :]  + boundary_matrix @ modes[cell-1, :])

    # Apply periodic boundary conditions 
    rhs[0, :] = rhs[N, :]
    rhs[N+1, :] = rhs[1, :]

    return rhs

def computeApprox(modes, grid, grid_spacing, data_type='quadrature', num_pts_per_cell=PTS_PER_ELEMENT): 
    degree = modes.shape[1]-1
    N = len(grid)
    # Internal cells only 
    if modes.shape[0] == N+2: 
        modes = modes[1:-1, :]
    # Evaluation points 
    if data_type =='quadrature':
        local_nodes, _ = np.polynomial.legendre.leggauss(num_pts_per_cell)
    elif data_type =='even': 
        local_nodes = np.array([-1, -0.5, 0, 0.5])
    
    grid_pts = np.zeros(N*num_pts_per_cell)
    for cell in range(N): 
        cell_center = grid[cell] + 0.5*grid_spacing[cell]
        grid_pts[cell*num_pts_per_cell: (cell+1)*num_pts_per_cell] = 0.5*grid_spacing[cell]*local_nodes + cell_center*np.ones(num_pts_per_cell)
    
    # Compute the Approximation at evaluation points
    approx = np.zeros(N*num_pts_per_cell)
    for cell in range(N): 
        approx[cell*num_pts_per_cell: (cell+1)*num_pts_per_cell] = sum(modes[cell, k] *eval_legendre(k, local_nodes)* np.sqrt(k+0.5) for k in range(degree + 1))

    return grid_pts, approx.reshape((1, N*num_pts_per_cell))


def computeExact(evaluation_points, initial_condition, final_time, a=1): 
    num_periods = np.floor(abs(a*final_time/(X_RIGHT-X_LEFT)))
    new_points = evaluation_points + (num_periods*(X_RIGHT-X_LEFT)- a * final_time) * np.ones(len(evaluation_points))
    for point in range(len(new_points)): 
        while new_points[point] < X_LEFT:
            new_points[point] += (X_RIGHT-X_LEFT)
        while new_points[point] > X_RIGHT:
            new_points[point] -=(X_RIGHT-X_LEFT)

    exact = initial_condition(new_points)

    return exact.reshape((1, len(new_points)))


## ---------------SAVE DATA ----------------

def saveData(num_samples, file_label, grid_level_lo=6, grid_level_hi=6, degree_lo=1, degree_hi=MAX_DEGREE, uniformgrid_ratio=1, nonsmooth_ratio=0.7, data_type='quadrature', ic='any', save_modal=False):
    tic = time.process_time()
    if save_modal: 
        data, exact, evaluation_pts, modes, grid, grid_spacing = generateData(num_samples, grid_level_lo, grid_level_hi, degree_lo, degree_hi, uniformgrid_ratio, nonsmooth_ratio, data_type=data_type, ic=ic, save_modal=save_modal)
    else: 
        data, exact, evaluation_pts = generateData(num_samples, grid_level_lo, grid_level_hi, degree_lo, degree_hi, uniformgrid_ratio, nonsmooth_ratio,  data_type=data_type, ic=ic, save_modal=save_modal)

    # Split into Training and Test datasets
    train_ratio = 0.8
    validation_ratio = 0.1
    data_train, exact_train, data_validation, exact_validation, data_test, exact_test = splitData(data, exact, train_ratio, validation_ratio)

    # Save the data 
    if save_modal: 
        data_label = file_label + "_Uniform"+ str(uniformgrid_ratio)+ "_Nonsmooth"+ str(int(10*nonsmooth_ratio))+ "_Ntotal"+ str(num_samples) + "_Train2Validation" + str(int(10*train_ratio))+ "-" + str(int(10*validation_ratio))
        np.savez(data_label, eval_pts=evaluation_pts, modes=modes, grid=grid, grid_spacing=grid_spacing, xtrain=data_train, ytrain=exact_train, xvalidation=data_validation, yvalidation=exact_validation, xtest=data_test, ytest=exact_test)
    else: 
        data_label = file_label + "_Uniform"+ str(uniformgrid_ratio)+ "_Nonsmooth"+ str(int(10*nonsmooth_ratio))+ "_Ntotal"+ str(num_samples) + "_Train2Validation" + str(int(10*train_ratio))+ "-" + str(int(10*validation_ratio))
        np.savez(data_label, eval_pts=evaluation_pts, xtrain=data_train, ytrain=exact_train, xvalidation=data_validation, yvalidation=exact_validation, xtest=data_test, ytest=exact_test)

    toc = time.process_time()
    print("CPU Time to generate and save data: ", toc-tic)

def saveDataNoSplit(num_samples, file_label, grid_level_lo=6, grid_level_hi=6, degree_lo=1, degree_hi=MAX_DEGREE, uniformgrid_ratio=1, nonsmooth_ratio=0.7, data_type='quadrature', ic='any', save_modal=False):
    tic = time.process_time()

    if save_modal: 
        data, exact, evaluation_pts, modes, grid, grid_spacing = generateData(num_samples, grid_level_lo, grid_level_hi, degree_lo, degree_hi, uniformgrid_ratio, nonsmooth_ratio,  data_type=data_type, ic=ic, save_modal=save_modal)
        data_label = file_label + "_Uniform"+ str(uniformgrid_ratio)+ "_Nonsmooth"+ str(int(10*nonsmooth_ratio))+ "_Ntotal"+ str(num_samples)
        np.savez(data_label, eval_pts=evaluation_pts, modes=modes, grid=grid, grid_spacing=grid_spacing, data=data, exact=exact)
    else: 
        data, exact, evaluation_pts = generateData(num_samples, grid_level_lo, grid_level_hi, degree_lo, degree_hi, uniformgrid_ratio, nonsmooth_ratio,  data_type=data_type, ic=ic, save_modal=save_modal)
        data_label = file_label + "_Uniform"+ str(uniformgrid_ratio)+ "_Nonsmooth"+ str(int(10*nonsmooth_ratio))+ "_Ntotal"+ str(num_samples)
        np.savez(data_label, eval_pts=evaluation_pts, data=data, exact=exact)

    toc = time.process_time()
    print("CPU Time to generate and save data: ", toc-tic)
    

def computeModes(approx, degree, num_pts_per_cell=PTS_PER_ELEMENT): 
    num_samples = approx.shape[0]
    N = int(approx.shape[2]/num_pts_per_cell)
    # Guass-Legendre Quadrature Nodes and Weights
    xi_eval, weights_eval = np.polynomial.legendre.leggauss(num_pts_per_cell)

    # Compute the (N) x (degree + 1) L2-Projection modes matrix
    filtered_modes = []
    for sample in range(num_samples):
        sample_modes = np.zeros((N, int(degree[sample] + 1)))
        for cell in range(N):
            for k in range(int(degree[sample] + 1)):
                integrand = approx[sample, 0, cell*num_pts_per_cell: (cell+1)*num_pts_per_cell] * eval_legendre(k, xi_eval)* np.sqrt(k+0.5)
                sample_modes[cell, k]= sum(integrand*weights_eval)

        filtered_modes.append(list(sample_modes))
    
    return np.array(filtered_modes, dtype=object)
