# setup code for problem
import numpy as np

cos = np.cos
sin = np.sin
pi = np.pi

def curve(t):
    return np.array([
        (3/4)*cos(t-pi/4.)*(1 + sin(2*t)/2),
        sin(t-pi/4)*(1 + sin(2*t)/2)
        ])

def dcurve_dt(t):
    return np.array([
        -(3./4.)*sin(t-pi/4.)*(1 + sin(2*t)/2.) + (3./4.)*cos(t-pi/4.)*cos(2*t),
        cos(t-pi/4.)*(1 + sin(2*t)/2.) + sin(t-pi/4.)*cos(2*t)
        ])

def u_exact(points):
    x, y = points
    return (-1./(2*np.pi))*np.log(np.sqrt((x-0.1)**2 + (y-0.7)**2))

test_targets = np.array([
    [-0.2, 0],
    [0.2, 0],
    [0, -0.2],
    [0, 0.2]
    ]).T

npanels1 = 10
npanels2 = 20

# provided code
import math
import numpy as np
import numpy.linalg as la
import scipy.special
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spla

# This data structure helps you get started by setting up geometry
# and Gauss quadrature panels for you.

class QuadratureInfo:
    def __init__(self, nintervals):
        self.nintervals = nintervals
        par_length = 2*np.pi
        intervals = np.linspace(0, 2*np.pi, nintervals+1)
        self.npoints = 7+1
        self.shape = (nintervals, self.npoints)

        ref_info = scipy.special.legendre(self.npoints).weights
        ref_nodes = ref_info[:,0]
        ref_weights = ref_info[:,2]

        par_intv_length = intervals[1] - intervals[0]

        self.par_nodes = np.zeros((nintervals, self.npoints))
        for i in range(nintervals):
            a, b = intervals[i:i+2]

            assert abs((b-a) - par_intv_length) < 1e-10
            self.par_nodes[i] = ref_nodes*par_intv_length*0.5 + (b+a)*0.5

        self.curve_nodes = curve(self.par_nodes.reshape(-1)).reshape(2, nintervals, -1)
        self.curve_deriv = dcurve_dt(self.par_nodes.reshape(-1)).reshape(2, nintervals, -1)

        self.curve_speed = la.norm(self.curve_deriv, 2, axis=0)

        tangent = self.curve_deriv / self.curve_speed
        tx, ty = tangent
        self.normals = np.array([ty, -tx])

        self.curve_weights = self.curve_speed * ref_weights * par_intv_length / 2
        self.panel_lengths = np.sum(self.curve_weights, 1)

        if 0:
            plt.plot(
                self.curve_nodes[0].reshape(-1),
                self.curve_nodes[1].reshape(-1), "x-")
            plt.show()

# my code
def gradG_interExpan(x, y, nx, r, term_idx):
    # evaluate gradient expansion using interior limit
    # based on values for x, y, x normal (nx), expansion radius, and the expansion term idx
    nx1 =nx[0]
    nx2 =nx[1]
    x1  = x[0]
    x2  = x[1]
    y1  = y[0]
    y2  = y[1]
    Pi  = np.pi
    return np.array({
        0 : [-(nx1*r + x1 - y1)/(2.*Pi*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2)),-(nx2*r + x2 - y2)/(2.*Pi*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2))],
        1 : [((nx1*r + x1 - y1)*(-2*nx1*r*(nx1*r + x1 - y1) - 2*nx2*r*(nx2*r + x2 - y2)) + nx1*r*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2))/(2.*Pi*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2)**2),(nx2*r*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2) + (-2*nx1*r*(nx1*r + x1 - y1) - 2*nx2*r*(nx2*r + x2 - y2))*(nx2*r + x2 - y2))/(2.*Pi*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2)**2)],
        2 : [-((r**2*((nx1*r + x1 - y1)*(4*(nx1*(nx1*r + x1 - y1) + nx2*(nx2*r + x2 - y2))**2 - (nx1**2 + nx2**2)*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2)) - 2*nx1*(nx1*(nx1*r + x1 - y1) + nx2*(nx2*r + x2 - y2))*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2)))/(Pi*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2)**3)), -((r**2*(-2*nx2*(nx1*(nx1*r + x1 - y1) + nx2*(nx2*r + x2 - y2))*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2) + (4*(nx1*(nx1*r + x1 - y1) + nx2*(nx2*r + x2 - y2))**2 - (nx1**2 + nx2**2)*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2))*(nx2*r + x2 - y2)))/(Pi*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2)**3))],
        3 : [(-3*(2*(nx1*r + x1 - y1)*((2*nx1*r*(nx1*r + x1 - y1) + 2*nx2*r*(nx2*r + x2 - y2))**3 -4*(nx1**2 + nx2**2)*r**3*(nx1*(nx1*r + x1 - y1) + nx2*(nx2*r + x2 - y2))*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2)) -2*nx1*r**3*(4*(nx1*(nx1*r + x1 - y1) + nx2*(nx2*r + x2 - y2))**2 - (nx1**2 + nx2**2)*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2))*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2)))/(2.*Pi*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2)**4),(3*(2*nx2*r**3*(4*(nx1*(nx1*r + x1 - y1) + nx2*(nx2*r + x2 - y2))**2 - (nx1**2 + nx2**2)*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2))*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2) - 2*((2*nx1*r*(nx1*r + x1 - y1) + 2*nx2*r*(nx2*r + x2 - y2))**3 -4*(nx1**2 + nx2**2)*r**3*(nx1*(nx1*r + x1 - y1) + nx2*(nx2*r + x2 - y2))*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2))*(nx2*r + x2 - y2)))/(2.*Pi*((nx1*r + x1 - y1)**2 + (nx2*r + x2 - y2)**2)**4)]
    }[term_idx])

def gradG_exterExpan(x, y, nx, r, term_idx):
    # evaluate gradient expansion using exterior limit
    # based on values for x, y, x normal (nx), expansion radius, and the expansion term idx
    nx1 =nx[0]
    nx2 =nx[1]
    x1  = x[0]
    x2  = x[1]
    y1  = y[0]
    y2  = y[1]
    Pi  = np.pi
    return np.array({
        0 : [-(-(nx1*r) + x1 - y1)/(2.*Pi*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2)),-(-(nx2*r) + x2 - y2)/(2.*Pi*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2))],
        1 : [(2*r*(nx1*r - x1 + y1)*(nx1**2*r + nx1*(-x1 + y1) + nx2*(nx2*r - x2 + y2)) - nx1*r*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2))/(2.*Pi*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2)**2),(r*(2*nx1*(x1 - y1)*(x2 - y2) + nx2*(nx2*r + x1 - x2 - y1 + y2)*(nx2*r - x1 - x2 + y1 + y2) + nx1**2*r*(nx2*r - 2*x2 + 2*y2)))/(2.*Pi*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2)**2)],
        2 : [-((r**2*(-2*nx1*(-(nx1*(nx1*r - x1 + y1)) - nx2*(nx2*r - x2 + y2))*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2) - (nx1*r - x1 + y1)*(4*(nx1**2*r + nx1*(-x1 + y1) + nx2*(nx2*r - x2 + y2))**2 - (nx1**2 + nx2**2)*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2))))/(Pi*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2)**3)),-((r**2*(-2*nx2*(-(nx1*(nx1*r - x1 + y1)) - nx2*(nx2*r - x2 + y2))*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2) - (nx2*r - x2 + y2)*(4*(nx1**2*r + nx1*(-x1 + y1) + nx2*(nx2*r - x2 + y2))**2 - (nx1**2 + nx2**2)*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2))))/(Pi*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2)**3))],
        3 : [(-3*r**3*(nx1*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2)*(4*(nx1**2*r + nx1*(-x1 + y1) + nx2*(nx2*r - x2 + y2))**2 - (nx1**2 + nx2**2)*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2)) +4*(nx1*r - x1 + y1)*(-2*(nx1**2*r + nx1*(-x1 + y1) + nx2*(nx2*r - x2 + y2))**3 -(nx1**2 + nx2**2)*(-(nx1**2*r) + nx1*(x1 - y1) - nx2*(nx2*r - x2 + y2))*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2))))/(Pi*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2)**4),(-3*r**3*(nx2*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2)*(4*(nx1**2*r + nx1*(-x1 + y1) + nx2*(nx2*r - x2 + y2))**2 - (nx1**2 + nx2**2)*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2)) +4*(nx2*r - x2 + y2)*(-2*(nx1**2*r + nx1*(-x1 + y1) + nx2*(nx2*r - x2 + y2))**3 -(nx1**2 + nx2**2)*(-(nx1**2*r) + nx1*(x1 - y1) - nx2*(nx2*r - x2 + y2))*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2))))/(Pi*((nx1*r - x1 + y1)**2 + (nx2*r - x2 + y2)**2)**4)]
    }[term_idx])

def dlayerPotentialEvalAtBoundary(density, quad_info):
    # function to evaluate double layer potential on boundary for  use in solving
    # the integral equation

    # define QBX accuracy order
    qbx_order = 3

    # get size for discretization
    (npanels, npoints) = quad_info.shape

    # get the panel radii
    radii = quad_info.panel_lengths / 2

    # compute the double layer potential from interior limit
    phi_inter = np.zeros((npanels, npoints))
    for kpan in range(0, npanels):  # loop through target points to evaluate at
        for kpt in range(0, npoints):
            for i in range(0, qbx_order):  # loop through expansion
                term_val = 0.0  # temporary value for storing taylor expansion term
                for pnl in range(0, npanels):  # loop through source points to eval at
                    for pt in range(0, npoints):
                        # compute the QBX one dimensional expansion term
                        # based on the formula:
                        # f^(i)(0) = int_{\Gamma} ny \dot \partial_{t^i} grad_y G(x + r(1-t)nx,y) sigma(y) dy
                        # which gets approximated as
                        # f^{i}(0) \approx sum_k w_k * ny(y_k) \dot \partial_{t^i} grad_y G(x + r(1-t)nx,y_k) * sigma(y_k)
                        term_val += quad_info.curve_weights[pnl, pt] \
                                    * np.dot(gradG_interExpan(quad_info.curve_nodes[:, kpan, kpt],
                                                              quad_info.curve_nodes[:, pnl, pt],
                                                              quad_info.normals[:, kpan, kpt],
                                                              radii[pnl],
                                                              i),
                                             quad_info.normals[:, pnl, pt]) * density[pnl, pt]

                # divide expansion term by appropriate coefficient based on factorial
                term_val /= math.factorial(i)

                # add this term contribution to net double layer potential
                phi_inter[kpan, kpt] += term_val

    # compute the double layer potential from exterior limit
    phi_exter = np.zeros((npanels, npoints))
    for kpan in range(0, npanels): # loop through target points to evaluate at
        for kpt in range(0, npoints):
            for i in range(0, qbx_order): # loop through expansion
                term_val = 0.0 # temporary value for storing taylor expansion term
                for pnl in range(0, npanels): # loop through source points to eval at
                    for pt in range(0, npoints):

                        # compute the QBX one dimensional expansion term
                        # based on the formula:
                        # f^(i)(0) = int_{\Gamma} ny \dot \partial_{t^i} grad_y G(x + r(1-t)nx,y) sigma(y) dy
                        # which gets approximated as
                        # f^{i}(0) \approx sum_k w_k * ny(y_k) \dot \partial_{t^i} grad_y G(x + r(1-t)nx,y_k) * sigma(y_k)
                        term_val += quad_info.curve_weights[pnl, pt] \
                                    * np.dot(gradG_exterExpan(quad_info.curve_nodes[:, kpan, kpt],
                                                              quad_info.curve_nodes[:, pnl, pt],
                                                              quad_info.normals[:, kpan, kpt],
                                                              radii[pnl],
                                                              i),
                                             quad_info.normals[:, pnl, pt]) * density[pnl, pt]

                # divide expansion term by appropriate coefficient based on factorial
                term_val /= math.factorial(i)

                # add this term contribution to net double layer potential
                phi_exter[kpan, kpt] += term_val

    # return average for potential
    return (phi_inter + phi_exter)*0.5

def dlayerPotentialEvalInterior(x, density, quad_info):
    # function to evaluate double layer potential on locations x not on the boundary

    # get size for input
    (nd, npt) = x.shape

    # get size for discretization
    (npanels, npoints) = quad_info.shape

    # define function for grad_y G(x,y)
    def gradG(x,y):
        (x1,x2)=x
        (y1,y2)=y
        Pi = np.pi
        return np.array([-(x1 - y1) / (2. * Pi * ((x1 - y1) ** 2 + (x2 - y2) ** 2)), -(x2 - y2) / (2. * Pi * ((x1 - y1) ** 2 + (x2 - y2) ** 2))])

    # compute the double layer potential from interior limit
    phi = np.zeros((npt,))
    for tpt in range(0, npt):  # loop through target points to evaluate at
        for pnl in range(0, npanels):  # loop through source points to eval at
            for pt in range(0, npoints):
                # compute double layer potential integral
                phi[tpt] += quad_info.curve_weights[pnl, pt] * \
                           np.dot(gradG(x[:,tpt],quad_info.curve_nodes[:, pnl, pt]), quad_info.normals[:, pnl, pt]) \
                           * density[pnl, pt]

    # return the output potential
    return phi

def bvp(n):
    # function to solve dirichlet boundary value problem
    # n: number of panels to break boundary into

    # define callback to be used in GMRES, taken from:
    # https://stackoverflow.com/questions/33512081/getting-the-number-of-iterations-of-scipys-gmres-iterative-method
    class gmres_counter(object):
        def __init__(self, disp=True):
            self._disp = disp
            self.niter = 0

        def __call__(self, rk=None):
            self.niter += 1
            if self._disp:
                print('Residual norm({0}): '.format(self.niter), np.linalg.norm(rk))

    # setup geometry and quadrature info
    qinfo = QuadratureInfo(nintervals=n)
    (npan,npts) = qinfo.shape

    # define linear operator that will be used in GMRES
    def dlOperator(sigma):
        (npan,npts) = qinfo.shape
        return 0.5*sigma + dlayerPotentialEvalAtBoundary(sigma.reshape(npan,npts), qinfo).reshape(npan*npts,)
    operator = scipy.sparse.linalg.LinearOperator(shape=(npan*npts,npan*npts),matvec=dlOperator)

    # define the values that the solution should have on the boundary
    u_boundary  = u_exact(qinfo.curve_nodes).reshape(npan*npts,)

    if 0: # check if the constant density case produces expected result of density/2 for operator
        sigma = -np.ones((npan*npts,))
        print(dlOperator(sigma))

    # solve using GMRES
    counter = gmres_counter()
    sigma1, info1 = scipy.sparse.linalg.gmres(A=operator,b=u_boundary,tol=1e-13,maxiter=50,callback=counter)

    # return the useful info
    return (sigma1.reshape(npan,npts),qinfo,dlOperator,counter.niter)


#solve with each set of panels
(s1,q1,op1,niter1) = bvp(npanels1)
(s2,q2,op2,niter2) = bvp(npanels2)

# compute the max norm between the reference solution and the QBX solution
print('Max-Norm Error for npanels1 after {0} GMRES iterations: '.format(niter1),
      np.linalg.norm(u_exact(test_targets) - dlayerPotentialEvalInterior(test_targets,s1,q1),ord=np.inf))
print('Max-Norm Error for npanels2 after {0} GMRES iterations: '.format(niter2),
      np.linalg.norm(u_exact(test_targets) - dlayerPotentialEvalInterior(test_targets,s2,q2),ord=np.inf))