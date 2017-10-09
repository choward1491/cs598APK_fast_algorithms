import numpy as np
import numpy.linalg as la

multi_indices = []
order = 6
for i in range(order+1):
    for j in range(order+1):
        if i+j <= order:
            multi_indices.append((i, j))
ncoeffs = len(multi_indices)

nsources = 15
nxtgts = 200
nytgts = 200

angles = np.linspace(0, 2*np.pi, nsources, endpoint=False)
r = 1 + 0.3 * np.sin(3*angles)
sources = np.array([
    r*np.cos(angles),
    r*np.sin(angles),
    ])

center_old = np.random.randn(2) * 0.2 - 0.3
center_new = np.random.randn(2) * 0.2 + 0.3

charges = 0.5+2*np.sin(3*angles)

left, right, bottom, top = extent = (-4, 4, -4, 4)
targets = np.mgrid[left:right:nxtgts*1j, bottom:top:nytgts*1j]

coefficients_old = np.zeros((ncoeffs,))
for icoeff, mi in enumerate(multi_indices):
    dist = center_old.reshape(-1, 1) - sources
    coefficients_old[icoeff] += (
        np.product(dist ** np.array(mi).reshape(-1, 1), axis=0)
        * charges
        ).sum()

def mi_derivative(expr, vector, mi):
    for mi_i, vec_i in zip(mi, vector):
        expr = expr.diff(vec_i, mi_i)
    return expr

def kernel_deriv(deriv_orders, x):
    import sympy as sp
    assert len(x) <= 4
    d = len(x)
    vec = [sp.Symbol("x%d" % i) for i in range(d)]
    r2 = sum(v_i**2 for v_i in vec)

    knl = sp.log(sp.sqrt(r2))
    deriv_knl = mi_derivative(knl, vec, deriv_orders)
    func = sp.lambdify(tuple(vec), deriv_knl, "numpy")
    return func(*x)

# compute reference potential
dist_vecs = sources.reshape(2, -1, 1, 1) - targets.reshape(2, 1, targets.shape[-1], -1)
dists = np.sqrt(np.sum(dist_vecs**2, axis=0))
pot_ref = np.sum(charges.reshape(-1, 1, 1) * np.log(dists), axis=0)

def plot_pot(pot):
    import matplotlib.pyplot as pt
    pt.imshow(pot.T[::-1], extent=extent)
    pt.colorbar()



#
#
# MY CODE NOW
#
#

import math
import numpy as np

def binomial(x, y):
    try:
        binom = math.factorial(x) // math.factorial(y) // math.factorial(x - y)
    except ValueError:
        binom = 0
    return binom

def mpow(v,midx):
    out = 1
    (d,) = v.shape
    for i in range(0,d):
        out = out*v[i]**midx[i]
    return out

def getNewAlpha(alpha, midx, cold, cnew):

    # get size of list
    (ncoefs) = alpha.shape

    # make copy of alpha into the new alpha vector
    alpha_n = np.zeros(alpha.shape)

    # take difference of dc
    dc = cold - cnew

    # compute new alpha values
    for p in range(0, ncoefs[0]):
        (m,n) = midx[p]
        for k in range(0,m+1):
            b1 = binomial(m,k)
            for l in range(0,n+1):
                b2 = binomial(n,l)
                idx = [x for x, y in enumerate(midx) if y[0] == (m-k) and y[1] == (n-l)][0]
                alpha_n[p] += b1*b2*mpow(dc,(k,l))*alpha[idx]

    # return new alpha
    return alpha_n


def mfact(midx):
    value = math.factorial(midx[0]) * math.factorial(midx[1])
    return value

coefficients_new2 = np.zeros((ncoeffs,))
for icoeff, mi in enumerate(multi_indices):
    dist = center_new.reshape(-1, 1) - sources
    coefficients_new2[icoeff] += (
        np.product(dist ** np.array(mi).reshape(-1, 1), axis=0)
        * charges
        ).sum()

coefficients_new = getNewAlpha(coefficients_old,
                               multi_indices,
                               center_new, center_old)

print(coefficients_new)
print(coefficients_new2)

# compute potentials
(ncoefs) = coefficients_new.shape
(d, nx, ny) = targets.shape
pot_old = np.zeros((nx, ny))
pot_new = np.zeros((nx, ny))
d1 = targets - center_old.reshape(2,1,1)
d2 = targets - center_new.reshape(2,1,1)
for k in range(0, ncoefs[0]):
    pot_old += kernel_deriv(multi_indices[k], d1) * (coefficients_old[k]/mfact(multi_indices[k]))
    pot_new += kernel_deriv(multi_indices[k], d2) * (coefficients_new[k]/mfact(multi_indices[k]))
