import os
import math
from Data import *
from Multiwavelets import *

cwd = os.getcwd()
os.chdir('SIACPythonCode')
from symmetricpp import symmetricpp
os.chdir(cwd)


def apply_siac_filter(modes_dg, smoothness, filter_size='full', modes_type='synthetic', bc='dirichlet', ghost_cells=0, pts_per_cell=4, N=128, xL=-5, xR=5):
    
    # Domain 
    h = (xR-xL)/N

    # SIAC Kernel: B-spline order
    spline_order = smoothness+2

    # Compute SIAC relevant data: modes
    num_samples = modes_dg.shape[0]
    filtered_siac = np.zeros((num_samples, N*pts_per_cell))
    modes_siac = np.zeros(num_samples, dtype=object)
    multiwavelets_siac = np.zeros((num_samples, N))
    
    for sample in range(num_samples):

        # DG modes per sample 
        if modes_type == 'synthetic':
            p = len(modes_dg[sample, 0]) - 1
            modes_sample = np.vstack(modes_dg[sample, 1:-1]).reshape((N, p+1))
        elif modes_type == 'cfd':
            modes_sample = np.array(modes_dg[sample]).reshape((-1, N))
            p = modes_sample.shape[0]-1
            modes_sample = modes_sample.T

        # Update DG data if Dirichlet BCs instead of periodic BCs are used
        if ghost_cells > 0: 
            if bc == 'dirichlet':
                ghost_modes_left = np.reshape(modes_sample[0, :], (1, p+1))*np.ones((ghost_cells, p+1))
                ghost_modes_right = np.reshape(modes_sample[-1, :], (1, p+1))*np.ones((ghost_cells, p+1))
            elif bc == 'periodic':
                ghost_modes_left = modes_sample[-ghost_cells:, :]
                ghost_modes_right = modes_sample[:ghost_cells, :]
            modes_ext = np.concatenate((ghost_modes_left, modes_sample, ghost_modes_right), axis=0)
        else:
            modes_ext = modes_sample
        
        # SIAC filter size
        if filter_size == 'full':
            RS = p
        elif filter_size == 'half':
            RS = int(np.ceil(0.5*(p+spline_order-1)))
        elif filter_size == '0':
            RS = 0
        else:
            raise ValueError('Invalid filter size')

        # Apply SIAC filter and extract SIAC data
        _, filtered_approx = np.array(convolution_SIAC_1d(modes_ext, pts_per_cell, int(N+2*ghost_cells), p, smoothness, RS, xleft=xL-(ghost_cells*h), xright=xR+(ghost_cells*h)))
        if ghost_cells > 0:
            filtered_siac[sample, :] = filtered_approx[ghost_cells*pts_per_cell:-ghost_cells*pts_per_cell]
        else:
            filtered_siac[sample, :] = filtered_approx
        modes_filtered = computeModes(filtered_siac[sample, :].reshape((1,1, -1)), np.array([p]), num_pts_per_cell=pts_per_cell)[0]
        modes_siac[sample] = list(modes_filtered.T)
        multiwavelets_siac[sample, :] = calculate_multiwavelet_coeffs(modes_filtered.T, p, N=N)
    
    return filtered_siac, modes_siac, multiwavelets_siac


def convolution_SIAC_1d(uhat, evalPoints, Nx, p, smoothness, RS, xleft=-1, xright=1): 
    # Domain Parameters
    hx = (xright-xleft)/Nx
    x_grid = np.array([xleft +cell*hx for cell in range(Nx+1)])

    # Evaluation Points
    zEval, wEval = np.polynomial.legendre.leggauss(evalPoints)
    
    # Kernel Parameters: B-splines, support
    order = smoothness + 2
    # RS = int(max(math.ceil(0.5*(p+order-1)),math.ceil(0.5*p)))
    # RS = int(math.ceil(0.5*(p+order-1)))
    kwide = math.ceil(RS + 0.5*order)

    
    # SIAC symmetric post-processing matrix
    symcc = symmetricpp(p, order, RS, zEval)

    # Post-process approximation 
    PPxEval=[]
    PPfApprox=[]

    for nelx in range(Nx):  
        for ix in range(evalPoints):
            xrg = 0.5*hx*(zEval[ix]+1.0) + x_grid[nelx]
            PPxEval.append(xrg)
            
            # set indices to form post-processed solution
            upost = 0.0
            if kwide <= nelx <= Nx-2-kwide:
                for kkx in range(2*kwide+1):
                    kk2x = kkx - kwide
                    xindex = nelx + kk2x

                    # interior elements, use symmetric filter
                    for mx in range(p+1):
                        upost = upost + symcc[kkx][mx][ix]*uhat[xindex, mx]*np.sqrt(mx+0.5)

            # boundary filtering, either turn of boundary filtering or use this bit.
            elif nelx < kwide:
                for kkx in range(2*kwide+1):
                    kk2x = kkx - kwide
                    if nelx+kk2x <0:
                        xindex = Nx+nelx+kk2x
                    else:
                        xindex = nelx+kk2x
                    
                    for mx in range(p+1):
                            upost = upost + symcc[kkx][mx][ix]*uhat[xindex, mx]*np.sqrt(mx+0.5)
            else:
                for kkx in range(2*kwide+1):
                    kk2x = kkx - kwide
                    if kk2x <= 0:
                        xindex = nelx + kk2x
                    else:
                        xindex = nelx-Nx+kk2x
                    
                    for mx in range(p+1):
                            upost = upost + symcc[kkx][mx][ix]*uhat[xindex][mx]*np.sqrt(mx+0.5)

            PPfApprox.append(upost)


    return PPxEval, PPfApprox

        


