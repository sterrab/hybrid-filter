import numpy as np
from sympy import Symbol, integrate, legendre
from scipy.special import eval_legendre

x = Symbol('x')
z = Symbol('z')

MAX_DEGREE = 4

def multiwavelet(degree, eval_pt):
    if degree == 0:
        return [np.sqrt(0.5) + eval_pt * 0]
    if degree == 1:
        return [np.sqrt(1.5) * (-1 + 2 * eval_pt), np.sqrt(0.5) * (-2 + 3 * eval_pt)]
    if degree == 2:
            return [1/3 * np.sqrt(0.5) * (1 - 24*eval_pt + 30*(eval_pt**2)),
                    1/2 * np.sqrt(1.5) * (3 - 16*eval_pt + 15*(eval_pt**2)),
                    1/3 * np.sqrt(2.5) * (4 - 15*eval_pt + 12*(eval_pt**2))]
    if degree == 3:
        return [np.sqrt(15/34) * (1 + 4*eval_pt - 30*(eval_pt**2) + 28*(eval_pt**3)),
                np.sqrt(1/42) * (-4 + 105*eval_pt - 300*(eval_pt**2) + 210*(eval_pt**3)),
                1/2 * np.sqrt(35/34) * (-5 + 48*eval_pt - 105*(eval_pt**2) + 64*(eval_pt**3)),
                1/2 * np.sqrt(5/42) * (-16 + 105*eval_pt - 192*(eval_pt**2) + 105*(eval_pt**3))]
    if degree == 4:
        return [np.sqrt(1/186) * (1 + 30*eval_pt + 210*(eval_pt**2)
                                    - 840*(eval_pt**3) + 630*(eval_pt**4)),
                0.5 * np.sqrt(1/38) * (-5 - 144*eval_pt + 1155*(eval_pt**2)
                                        - 2240*(eval_pt**3) + 1260*(eval_pt**4)),
                np.sqrt(35/14694) * (22 - 735*eval_pt + 3504*(eval_pt**2)
                                        - 5460*(eval_pt**3) + 2700*(eval_pt**4)),
                1/8 * np.sqrt(21/38) * (35 - 512*eval_pt + 1890*(eval_pt**2)
                                        - 2560*(eval_pt**3) + 1155*(eval_pt**4)),
                0.5 * np.sqrt(7/158) * (32 - 315*eval_pt + 960*(eval_pt**2)
                                        - 1155*(eval_pt**3) + 480*(eval_pt**4))]


def quadrature_mirror_filters(degree):
    H = np.zeros((degree + 1, degree + 1, 2))
    G = np.zeros((degree + 1, degree + 1, 2))

    for i in range(degree + 1):
        for j in range(degree + 1):
            H[i, j, 0] = 1 / np.sqrt(2) * np.sqrt(i + 0.5) * np.sqrt(j + 0.5) * \
                         integrate(legendre(i, 0.5 * (z - 1)) * legendre(j, z), (z, -1, 1))
            H[i, j, 1] = 1 / np.sqrt(2) * np.sqrt(i + 0.5) * np.sqrt(j + 0.5) * \
                         integrate(legendre(i, 0.5 * (z + 1)) * legendre(j, z), (z, -1, 1))
            G[i, j, 0] = 1 / np.sqrt(2) * np.sqrt(j + 0.5) * \
                         integrate(multiwavelet(degree, -0.5 * (z - 1))[i] * (-1) ** (i + degree + 1) * legendre(j, z),
                                   (z, -1, 1))
            G[i, j, 1] = 1 / np.sqrt(2) * np.sqrt(j + 0.5) * \
                         integrate(multiwavelet(degree, 0.5 * (z + 1))[i] * legendre(j, z), (z, -1, 1))

    return H, G

# Compute QMF scaling matrix 
QMF_G0=[]
QMF_G1=[]
for degree in range(MAX_DEGREE + 1):
    _, G = quadrature_mirror_filters(degree)
    QMF_G0.append(G[:, :, 0])
    QMF_G1.append(G[:, :, 1])
QMF_G0 = np.array(QMF_G0, dtype=object)  
QMF_G1 = np.array(QMF_G1, dtype=object)  


def calculate_multiwavelet_coeffs(modes, degree, N=128):
    # Scaling Coefficients
    scaling_coeffs = 1 / (np.sqrt(N)) * modes

    # QMF matrix G for multiwavelets
    G = np.zeros((degree+1, degree+1, 2))
    G[:, :, 0] = np.array(QMF_G0[degree]).reshape((degree+1,degree+1))
    G[:, :, 1] = np.array(QMF_G1[degree]).reshape((degree+1,degree+1))

    # Computing Multiwavelets Coefficients
    d_coeffs = np.zeros(N)
    for cell in range(int(N/2)):
        d_coeffs[2 * cell] = sum(sum(G[degree, r, itilde]* scaling_coeffs[r, 2 * cell + itilde] 
                                    for itilde in range(2)) 
                                for r in range(degree + 1))

        if cell == int(N / 2) - 1:
            d_coeffs[2 * cell + 1] = d_coeffs[2 * cell]
        else:
            d_coeffs[2 * cell + 1] = sum(sum(G[degree, r, itilde]* scaling_coeffs[r, 2 * cell + itilde + 1]
                                                                    for itilde in range(2))
                                                            for r in range(degree + 1))


    return d_coeffs

