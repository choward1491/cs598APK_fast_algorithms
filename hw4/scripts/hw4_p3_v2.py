
#
# Code provided for initialization
#

import numpy as np
import numpy.linalg as la

# {{{ particles and ranges

nranges = 30
nparticles = 1500
assert nparticles % nranges == 0

t = np.linspace(0, 2*np.pi, nparticles)

wave = 1+0.25*np.sin(5*t)

x = np.vstack([
    np.cos(t)*wave,
    np.sin(t)*wave,
    ])

ranges = np.arange(0, nparticles+1, nparticles//nranges)

# }}}

# {{{ block matrix assembly

def interaction_mat(t, s, is_self=False):
    all_distvecs = s.reshape(2, 1, -1) - t.reshape(2, -1, 1)
    dists = np.sqrt(np.sum(all_distvecs**2, axis=0))
    if is_self:
        np.fill_diagonal(dists, 1)
    return np.log(dists)

a = np.zeros((nranges, nranges), dtype=object)


for i in range(nranges):
    tgt_start = ranges[i]
    tgt_stop = ranges[i+1]

    for j in range(nranges):
        src_start = ranges[j]
        src_stop = ranges[j+1]
        a[i, j] = interaction_mat(
            x[:, tgt_start:tgt_stop],
            x[:, src_start:src_stop],
            is_self=i == j)
        if i == j:
            a[i, j] += 100 * np.eye(tgt_stop-tgt_start)

# }}}

nproxies = 25
extra_rank = 5

rhs = np.random.randn(nparticles)

#
# Code created by me to solve problem
#

import numpy                        as np
import scipy.linalg.interpolative   as ID
#import scipy.linalg                 as sla
#import matplotlib.pyplot            as plot

# compute optimal proxy point distance based on desired error
# using the (2D) error estimator for Local Expansion/Multipole Expansion
# by using the associated furthest target/source distance
def optRange( eps, rank, distance_furthest ):
    return distance_furthest*eps**(-1/(rank+1))

# perform row and column ID for some source and target particles
# Row ID on target particles and then Column ID on source particles.
# This function will return the compressed interaction matrix chunk associated
# with the provided source-target pair.
def boxCompression(allParticles, allRanges, range_pair_tuple, rank, eps = 1e-8):
    '''
    :param allParticles: This represents all the particles in the problem
    :param allRanges:    This represents the list of range indices used to slice up the particles into chunks
    :param range_pair_tuple: This represents a tuple of two indices, the first representing the
        target range index and the second representing the source range index
    :param rank: The total rank being used for the interpolative decomposition
    :param eps: An error tolerance that can be set to manage the radius for proxies
    :return: (P, Ahat, PI) where P is the interpolation matrix from the Row ID, and Ahat is
        a compressed version of the interaction matrix after the various row/column ID operations,
        and PI is the interpolation matrix from the Column ID. The original interaction matrix A
        based on the chunk of targets and sources being looked at can be approximated by the following relationship:

        A \approx P * Ahat * PI
    '''

    # get the indices representing which range chunks are getting paired
    (idx_t, idx_s) = range_pair_tuple

    # get the mean location for the sources and largest distance from center to source
    sparticles  = allParticles[:, allRanges[idx_s]:allRanges[idx_s + 1]]
    ms          = np.mean(sparticles, axis=1).reshape(sparticles.shape[0],1)
    maxds       = np.max(np.linalg.norm(sparticles-ms,axis=0))

    # get the mean location for the targets and largest distance from center to target
    tparticles  = allParticles[:,allRanges[idx_t]:allRanges[idx_t+1]]
    mt          = np.mean(tparticles, axis=1).reshape(tparticles.shape[0],1)
    maxdt       = np.max(np.linalg.norm(tparticles-mt,axis=0))

    # compute parametrization variable for proxy circle
    s = np.linspace(0,2*np.pi,rank)

    # construct the source proxy set
    rs      = optRange(eps, rank, maxds)
    sproxy  = mt + rs * np.vstack((np.cos(s), np.sin(s)))

    # add to proxies based on particles within the radius
    # of the proxies yet outside of the range being looked at
    k1 = allRanges[idx_s]-1
    k2 = allRanges[idx_s+1]
    if k1 < 0: k1 = allRanges[-2]
    if k2 == allRanges[-1]: k2 = 0

    # add particles to the left of the first set index
    # that are within the proxy circle
    while True:
        xt = allParticles[:,k1].reshape(2,1)
        dist = np.linalg.norm(xt - ms)
        if dist <= rs:
            sproxy = np.hstack((sproxy,xt))
            k1 -= 1
        else:
            break

    # add particles to the left of the first set index
    # that are within the proxy circle
    while True:
        xt = allParticles[:, k2].reshape(2,1)
        dist = np.linalg.norm(xt - ms)
        if dist <= rs:
            sproxy = np.hstack((sproxy, xt))
            k2 = (k2 + 1) % nparticles
        else:
            break

    # construct the target proxy set
    rt      = optRange(eps, rank, maxdt)
    tproxy  = mt + rt * np.vstack((np.cos(s),np.sin(s)))

    # add to proxies based on particles within the radius
    # of the proxies yet outside of the range being looked at
    k1 = allRanges[idx_t] - 1
    k2 = allRanges[idx_t + 1]
    if k1 < 0: k1 = allRanges[-2]
    if k2 == allRanges[-1]: k2 = 0

    # add particles to the left of the first set index
    # that are within the proxy circle
    while True:
        xt = allParticles[:, k1].reshape(2, 1)
        dist = np.linalg.norm(xt - mt)
        if dist <= rt:
            tproxy = np.hstack((tproxy, xt))
            k1 -= 1
        else:
            break

    # add particles to the right of the last set index
    # that are within the proxy circle
    while True:
        xt = allParticles[:, k2].reshape(2, 1)
        dist = np.linalg.norm(xt - mt)
        if dist <= rt:
            tproxy = np.hstack((tproxy, xt))
            k2 = (k2 + 1) % nparticles
        else:
            break

    # find column indices using column ID on source particles
    (idx_s, proj_s) = ID.interp_decomp(interaction_mat(sproxy,sparticles),rank)

    # find row indices using row ID on target particles. Note that
    # this ID implementation does column ID, so need to transpose stuff
    (idx_t, proj_t) = ID.interp_decomp(interaction_mat(tparticles, tproxy).T,rank)
    proj_t = proj_t.T

    # construct the approximation matrix factors
    Ahat = interaction_mat(tparticles[:, idx_t[:rank]], sparticles[:, idx_s[:rank]])
    P = np.hstack((np.eye(rank), proj_t.T))[:, np.argsort(idx_t)].T
    PI = np.hstack((np.eye(rank), proj_s))[:, np.argsort(idx_s)]

    # For debugging, see how the approximation compares to the exact
    #A = interaction_mat(tparticles,sparticles)
    #err = np.linalg.norm(A - P@Ahat@PI)
    # print('Test error is {0}.'.format(err))

    # return the resulting compression
    return (P, Ahat, PI)


# construct the block diagonal matrix
D    = dict()
Dinv = dict()
for k in range(0,nranges):
    D[k] = a[k,k]
    Dinv[k] = np.linalg.inv(a[k,k])

# construct B matrix and PI matrix
rank = nproxies + extra_rank
PI = dict()
P  = dict()
Ah = np.zeros((nranges*rank,nranges*rank))
PIn = np.zeros((nranges*rank, nparticles))
Pn  = np.zeros((nparticles,ranges*rank))

for r in range(0,nranges):
    for c in range(0,nranges):
        if r != c:
            (P_, Ahat, PI_ ) = boxCompression(x, ranges, (r,c), rank)
            an[r,c] = Ahat
            if r not in P:
                P[r]  = 1
                Pn[r*]
            if c not in PI:
                PI[c] = 1

# finish construction of Ahat block matrix
for r in range(0,nranges):
    an[r,r] = np.linalg.inv(PI[r]@Dinv[r]@P[r])

# construct overall Ahat, PI matrix, (both without using blocks) and new b vector to solve
Ah = np.zeros((nranges*rank,nranges*rank))
PIn = np.zeros((nranges*rank, nparticles))
rhsn = np.zeros((nranges*rank,))
for r in range(0,nranges):
    nchunk = Dinv[r].shape[1]
    rhsn[r*rank:(r+1)*rank] = an[r,r]@PI[r]@Dinv[r]@rhs[r*nchunk:(r+1)*nchunk]
    PIn[r * rank:(r + 1)*rank, r*nchunk:(r + 1)*nchunk] = PI[r]
    for c in range(0,nranges):
        Ah[r*rank:(r+1)*rank, c*rank:(c+1)*rank] = an[r,c][:,:]


# solve the system for xhat
xhat = np.linalg.solve(Ah,rhsn)

# construct overall A matrix and solve for the solution exactly
A = np.zeros((nparticles,nparticles))
for r in range(0,nranges):
    for c in range(0, nranges):
        A[ranges[r]:ranges[r+1],ranges[c]:ranges[c+1]] = a[r,c][:,:]

# solve the system for x
solution = np.linalg.pinv(PIn)@xhat

# solve system exactly
solution_t = np.linalg.solve(A,rhs)

# error
err = np.linalg.norm(solution - solution_t)
print(err)

'''# solve the system for x
P, L, U  = sla.lu(PIn)
solution = '''