
# Setup Code
import numpy as np
nsources = 200
sources = np.random.rand(2, nsources)

center = np.array([0.5, 0.5])
proxy_rad = 1.5

targets = np.random.randn(2, 200)
rtgt = np.sqrt(np.sum(targets**2, axis=0))
new_rtgt = 3+0.3*np.arctan(rtgt)
targets = targets * new_rtgt/rtgt + center.reshape(-1, 1)

weights = np.random.rand(nsources)

def interaction_mat(t, s):
    all_distvecs = s.reshape(2, 1, -1) - t.reshape(2, -1, 1)
    dists = np.sqrt(np.sum(all_distvecs**2, axis=0))
    return np.log(dists)

ranks = list(range(5, 40, 5))


# My solution code
# Need to generate code to do the following:
#
# Create a labeled, semi-(y-)logarithmic plot that, for each value
# rank in the list ranks shows the L2 approximation error for the
# 2D Laplace potential originating from sources with weights using
# an outgoing skeleton using rank points and a complex Taylor
# multipole expansion about center using rank terms.

import math
import scipy.linalg.interpolative as sinterp
import matplotlib.pyplot as plot

def complex_alpha(weights, w, c, deriv_order):
    t0 = w-c
    t1 = t0**deriv_order
    t2 = weights*t1
    return np.sum(t2,dtype=complex)

def log_deriv( z, deriv_order ):
    if deriv_order == 0:
        return np.log(z)
    else:
        c1 = math.factorial(deriv_order-1)*(1 - 2*( (1+deriv_order) % 2))
        c2 = z**(-deriv_order)
        return c1*c2

def series_term(z, w, c, weights, deriv_order):
    c1 = (1 - 2*(deriv_order % 2))/math.factorial(deriv_order)
    c2 = complex_alpha(weights,w,c,deriv_order)
    c3 = log_deriv(z-c,deriv_order)[0][0]
    return c1*c2*c3

def expansion_soln(targets, sources, center, weights, rank):
    # function to compute approximate solution using complex
    # taylor expansion

    # init complex variables
    c = np.empty((1,1),dtype=complex)
    z = np.empty(targets.shape[1:], dtype=complex)
    w = np.empty(sources.shape[1:], dtype=complex)
    c.real = center[0]
    c.imag = center[1]
    z.real = targets[0,:]
    z.imag = targets[1,:]
    w.real = sources[0,:]
    w.imag = sources[1,:]

    # compute taylor series approx for each target
    series_approx = np.zeros((z.shape[0],))
    for t in range(0, z.shape[0]):
        for k in range(0,rank):
            delta   = series_term(z[t],w,c,weights,k)
            dr      = delta.real
            series_approx[t] += dr.reshape(1,)

    # return the approximate solution
    return series_approx

def proxy_soln( targets, sources, center, weights, rank, proxy_radius):
    # method to compute approximate solution using proxies
    # set the proxy radius
    r = proxy_radius

    # create the set of proxies
    theta   = np.linspace(0, 2*math.pi, rank).reshape(1,rank)
    param   = np.concatenate((np.cos(theta),np.sin(theta)),0)
    p       = r*param + center.reshape(2,1)

    # compute the interaction matrix using proxies as "targets"
    A = interaction_mat(p,sources)

    # do column ID to find which sources are most useful
    idx, proj = sinterp.interp_decomp(A,rank)

    # use important sources, construct a temporary A and then use the ID stuff to fill it back in
    An = sinterp.reconstruct_matrix_from_id(interaction_mat(targets,sources[:,idx[:rank]]),idx,proj)

    # using approximate interaction matrix, compute solution
    return An@weights



def true_soln( targets, sources, weights ):
    # method to compute exact solution
    return interaction_mat(targets,sources)@weights


# get the true solution
psi_t = true_soln(targets,sources,weights)

# initialize arrays for storing error
num_rank = len(ranks)
err_prox = np.zeros((num_rank,1))
err_expn = np.zeros((num_rank,1))

# loop through rank values and compute error in estimates using
# proxies and using the taylor expansion
for idx, rank in enumerate(ranks):
    psi_p = proxy_soln(targets, sources, center, weights, rank, proxy_rad)
    psi_e = expansion_soln(targets, sources, center, weights, rank)
    err_prox[idx] = np.linalg.norm(psi_p - psi_t)
    err_expn[idx] = np.linalg.norm(psi_e - psi_t)

# plot the error results
fig = plot.figure()
ax  = fig.add_subplot(111)
rank_arr = np.array(ranks)
h1 = plot.semilogy(rank_arr,err_prox,linestyle='--', marker='o',color='b',label='Proxy')
h2 = plot.semilogy(rank_arr,err_expn,linestyle='--', marker='o',color=[0.7,0,1.0],label='Expansion')
plot.xlabel('Rank')
plot.ylabel('L2 Error')
plot.legend()
plot.grid()
