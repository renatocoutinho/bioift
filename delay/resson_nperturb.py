# -*- coding: utf-8 -*-
'''Computes the coefficients of a multiple scales perturbation expansion for the Hutchinson equation.

The (rescaled) Hutchinson equation, or delayed logistic model, is given by:
    d_t u = -u(t-tau) [1 - \epsilon u(t)]

We expand $u(t-tau)$ around $tau = \pi/2$ and construct a perturbation series:
    u(t - tau) = u(t - \pi/2 - \sigma \epsilon^2) = u(t - \pi/2) - \sigma
    \epsilon^2 d_t u(t - \pi/2) + \dots
with
    u = u_0 + \epsilon u_1 + \epsilon^2 u_2 + \dots

Also, we employ several time scales:
    t = t_0 + \epsilon^2 t_2 + \epsilon^4 t_4 + \dots
so that the sinusoidal solution in the zeroth order has amplitude depending on
the slow time scales t_2, t_4, \dots

Using the ansatz:
    u = c_0 + c_1 e^{it} + c_2 e^{2it} + \dots + c_n e^{nit} + c. c. terms
we are able to apply the operations on a vector representing a solution and
come up with a (hopefully) solvable linear system for each order in $\epsilon$
(in terms of the solutions in the preceding orders). Also, we are able to
compute the coefficients of the (differential) equations for the zeroth order
coefficients, which leads to a description of the envelope: equations for the
amplitude and phase of the leading order solution.
'''
from __future__ import division
import sympy as S
import numpy as N
from sympy.matrices import Matrix
from itertools import product as iproduct

def prod(u, v):
    '''Calculates the product between two state vectors.'''
    r = N.zeros(u.shape, dtype=object)
    for i, x in enumerate(u):
        for j, y in enumerate(v):
            if i+j < len(u):
                r[i+j] += x * y
            if i-j >= 0:
                r[i-j] += x * y.conjugate()
            else:
                r[j-i] += x.conjugate() * y
    return r

def D(u, ts, z, degree=1):
    '''Calculates the derivative of a state vector.'''
    if degree is 0:
        return u
    else:
        return 1j * u * (N.array(range(len(u))))**degree + \
                sum([ z**(2*(i+1)) * N.array(S.diff(u, t, degree)) for (i, t) in enumerate(ts[1:])])

def delay(u, tau=S.pi/2):
    '''Calculates the state vector at a time $t - \tau$.'''
    return u * N.fromfunction(lambda i: S.exp(-1j*i*tau), u.shape)

def delay_taylor(u, ts, z, param, order=2, tau=S.pi/2):
    '''Calculates the taylor expansion around time $t - \tau$ in terms of the small parameter param.'''
    return N.sum([ delay(D(u, ts, z, i), tau) * (-1)**i * (param)**i for i in range((order+1)//2 + 1) ], axis=0)

def subs(E, *args, **kwargs):
    '''Performs substitutions on an array of expressions. Keeps the same syntax as sympy's .subs() function.'''
    return N.array([d.subs(*args, **kwargs) for d in E.flat]).reshape(E.shape)

def as_trig(u, ts):
    '''Express state vector as a (real) sum of sines and cosines.'''
    return N.sum([ 2 * c.as_real_imag()[0] * S.cos(i*ts[0]) - 2 * c.as_real_imag()[1] * S.sin(i*ts[0]) for i, c in enumerate(u) ])

def get_deriv(z, s, g, Q=None, n=4):
    '''Takes the rhs of the secular equation, and returns a function ready for the integration routine, with the parameters s, g and z already substituted in.'''
    if not Q:
        r = go(n)
        if not r['success']:
            return None
    Q[0] = S.simplify(Q[0].subs({'z': z, 's': s, 'g': g}))
    Q[1] = S.simplify(Q[1].subs({'z': z, 's': s, 'g': g}))
    return lambda y, t: N.array([float(Q[0].subs({'x': y[0], 'y': y[1]})),
                                float(Q[1].subs({'x': y[0], 'y': y[1]}))])

def deriv_n(n=4, z=0.1, s=1., g=1.):
    import pickle
    f = open('/home/renato/mestrado/code/delay/anal/f%i.pik' % n, 'r')
    Q = pickle.load(f)
    f.close()
    return get_deriv(z, s, g, Q)

def anal_integrate(x0, t, n=4, z=0.1, s=1., g=1.):
    from scipy.integrate import odeint
    d = deriv_n(n, z, s, g)
    return odeint(d, x0, t)

def go(n):
    # order to calculate up to
    #n = 4

    # dimension of the states base
    d = n + 2

    # number of time scales
    T = 1 + n//2
    
    z = S.Symbol('z', real=True) # perturbation \epsilon << 1
    s = S.Symbol('s', real=True) # parameter \sigma ~ ord(1)
    g = S.Symbol('g', real=True) # forcing oscilattion parameter
    # time variables
    ts = [ S.Symbol('t%i' % i, real=True) for i in range(T) ]
    
    # coefficients array
    # not including the ones we know are zero
    c = N.array([S.Symbol("c_%i%i" % (i, j), complex=True) if j <= i+1 and j % 2 == (i+1) % 2 else S.sympify(0) for (i, j) in iproduct(range(n+1), range(d))]).reshape(n+1, d)

    # the amplitude at order zero is a "free" parameter, depending on t1, t2 etc. (but *not* t0)
    A = S.Function('A')(*ts[1:])
    c[0][1] = A

    # the solution ansatz
    u = N.sum([ z**i * c[i,:] for i in range(n+1) ], axis = 0)

    one = N.zeros_like(u)
    one[0] = 1/2

    cosine = N.zeros_like(u)
    cosine[1] = 1/2

    # finally the equation
    E = N.vectorize(S.simplify)( D(u, ts, z, 1) + prod(delay_taylor(u, ts, z, param=s*z**2, order=n, tau=S.pi/2), one + z * u + g * z**2 * prod(u, cosine)) )
    E = N.vectorize(lambda x: S.Poly(x, z))(E)

    # cross your fingers
    sols = {}
    diffs = {}
    M = S.sympify(0)
    for o in range(1, n+1):
        eq1 = N.vectorize(lambda x: x.coeff(o))(E)
        eq = N.vectorize(S.simplify) (subs(subs(eq1, sols), diffs))
        # keep firt position out
        coeffs = [ c[o][i] for i in range(d) if c[o][i] ]
        # as well as the equation for it
        solution = apply(S.solvers.solve, [[eq[0]] + eq[2:].tolist()] + coeffs)
        if solution:
            sols.update(solution)
            # zero frequency coefficients can be taken to be real
            sols[c[o][0]] = S.simplify(sols[c[o][0]].as_real_imag()[0])
            if o is not 0:
                # homogeneous solution appears only in order zero
                sols[c[o][1]] = S.sympify(0)
            if o % 2 == 0:
                ss = S.solve(E[1].subs(sols).coeff(o).subs(diffs), A.diff(ts[o//2]))
                if ss:
                    diffs[A.diff(ts[o//2])] = ss[0]
                    M += z ** (o) * ss[0]
        else:
            print 'Solution not found at order %i.' % o
            return { 'success': False, 'eq': eq }
    
    x, y = S.symbols('xy', real=True)
    rmsubs = {S.re(A): x, S.im(A): y}
    Q = list((M.subs(diffs)/z**2).expand().as_real_imag())
    Q[0] = S.collect(S.simplify(Q[0].subs(rmsubs)), z)
    Q[1] = S.collect(S.simplify(Q[1].subs(rmsubs)), z)
    return { 'success': True,
            'M': M,
            'Q': Q,
            'E': E,
            'diffs': diffs,
            'sols': sols,
            'ts': ts, 'c': c, 'A': A, 'z': z, 'g': g, 's': s, 'x': x, 'y': y
           }

