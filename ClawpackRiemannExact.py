# Clawpack Exact Solution as implemented in  https://github.com/clawpack/riemann_book/tree/FA16:
# Riemann Problems and Jupyter Solutions
#     Theory and Approximate Solvers for Hyperbolic PDEs
# by David I. Ketcheson, Randall J. LeVeque, and Mauricio J. del Razo
# SIAM, 2020.   ISBN: 978-1-611976-20-5
# ebook: DOI 10.1137/1.9781611976212

import os 
import numpy as np

cwd = os.getcwd()
os.chdir('riemann_book')
from exact_solvers import euler 
os.chdir(cwd)

def generate_exact_soln(x, rho_l=0.445,u_l=0.698,p_l=3.528,
                rho_r=0.5,u_r=0.,p_r=0.571,gamma=1.4,t=1.3):
        
    # 
    q_l = euler.primitive_to_conservative(rho_l,u_l,p_l)
    q_r = euler.primitive_to_conservative(rho_r,u_r,p_r)
        
    states, speeds, reval, wave_types = euler.exact_riemann_solution(q_l, q_r, gamma=gamma)
    if t == 0:
        q = np.zeros((3,len(x)))
        q[0,:] = q_l[0]*(x<=0) + q_r[0]*(x>0)
        q[1,:] = q_l[1]*(x<=0) + q_r[1]*(x>0)
        q[2,:] = q_l[2]*(x<=0) + q_r[2]*(x>0)
    else:
        q = reval(x/t)
    primitive = euler.conservative_to_primitive(q[0],q[1],q[2])

    return primitive