# -*- coding: utf-8 -*-
from sympy.mpmath import *
#from numpy import *
from numpy import arange, array, vectorize, zeros, where, abs
from pylab import plot, xlabel, ylabel, legend, scatter, show
from scipy.stats._support import unique
#from scipy import optimize as op

# equation parameters
p = {
        'a': 0.1,
        'b': 0.,
        't1': 2.,
        't2': 2.,
        'mJ': 0.1,
        'mA': 0.001,
        'DJ': 1.,
        'DA': 1.,
        'd': 100,
        'L1': 1.
    }

# grid parameters
scale = 0.05
offsetx = 10.
offsety = 0
sizex = 20.
sizey = 10

gridx = arange(-offsetx, -offsetx + sizex, scale)
gridy = arange(-offsety, -offsety + sizey, scale)

# work around for lack of numpy coth
try:
    coth
except NameError:
    coth = lambda x: 1./tanh(x)

def f(x, p=p):
    return (1 - p['b'] * exp(-x * p['t2'])) * exp(x * p['t1']) \
            * (cosh(sqrt((p['mJ'] + x)/p['DJ']) * p['L1']) \
            - sinh(sqrt((p['mJ'] + x)/p['DJ']) * p['L1']) \
            * sqrt((p['mJ'] + x)/p['DJ']) / sqrt((p['mA'] + x)/p['DA']) \
            * coth(sqrt((p['mA'] + x)/p['DA']) * p['d'])) - p['a']

def M(x, l, p=p):
    return exp(sqrt((p['mJ'] + l)/p['DJ']) * x) + \
            ((1 - p['b'] * exp(-x * p['t2'])) * exp(x * p['t1']) - p['a'] * exp(sqrt((p['mJ'] + l)/p['DJ']) * p['L1'])) / \
            ((1 - p['b'] * exp(-x * p['t2'])) * exp(x * p['t1']) - p['a'] * exp(- sqrt((p['mJ'] + l)/p['DJ']) * p['L1'])) \
            * exp(- sqrt((p['mJ'] + l)/p['DJ']) * x)

def N(x, l, p=p):
    '''Incomplete!!'''
    return 2. * sinh(sqrt((p['mJ'] + l)/p['DJ']) * p['L1']) \
            * (exp(sqrt((p['mA'] + l)/p['DA']) * x) + \
            exp(sqrt((p['mA'] + l)/p['DA']) * (2*p['L2'] - x)))

def mp_solve(f, p, x0=0.01, limits=[1e-12, 5.], method='muller'):
    return findroot(lambda x: f(x, p=p), x0, solver=method)

def solve(f, p, x0=0.01, limits=[1e-12, 5.], method='fsolve'):
    if method == 'brentq':
        return op.brentq(f, limits[0], limits[1], args=p)
    elif method == 'fsolve':
        return op.fsolve(f, x0=x0, args=p)

def make_grid(gridx, gridy):
    r = array([ [ f(i + j*1j) for j in gridy ] for i in gridx ])
    return r

def roots_plot(r, part='real', scale=0.05, offset=10.):
    if part == 'real':
        p = vectorize(lambda x: x.real)(r) > zeros(r.shape)
    elif part == 'imag':
        p = vectorize(lambda x: x.imag)(r) > zeros(r.shape)
    pT = where(p == True)
    pF = where(p == False)
    scatter(scale*pT[0] - 10, scale*pT[1] - offset, c='b')
    scatter(scale*pF[0] - offset, scale*pF[1] - offset, c='r')
    xlabel('Re')
    ylabel('Im')
    show()

def trace_roots(r, scale=0.05, offsetx=10., offsety=0):
    rr = vectorize(lambda x: x.real)(r) > zeros(r.shape)
    ri = vectorize(lambda x: x.imag)(r) > zeros(r.shape)

    rroots = []
    iroots = []
    for i in range(r.shape[0] - 1):
        for j in range(r.shape[1] - 1):
            if rr[i][j] ^ rr[i+1][j]:
                rroots.append([i, j])
                rroots.append([i+1, j])
                #rroots.append([i+0.5, j])
            if rr[i][j] ^ rr[i][j+1]:
                rroots.append([i, j])
                rroots.append([i, j+1])
                #rroots.append([i, j+0.5])
            if ri[i][j] ^ ri[i+1][j]:
                iroots.append([i, j])
                iroots.append([i+1, j])
                #iroots.append([i+0.5, j])
            if ri[i][j] ^ ri[i][j+1]:
                iroots.append([i, j])
                iroots.append([i, j+1])
                #iroots.append([i, j+0.5])
    rroots = array(rroots)
    iroots = array(iroots)
    return (intersect(rroots, iroots), rroots, iroots)

def get_roots(f, p, guesses):
    roots = []
    for x in guesses:
        try:
            r = mp_solve(f, p, x0=x, method='muller')
        except:
            pass
        else:
            roots.append(r)
        try:
            r = mp_solve(f, p, x0=x, method='secant')
        except:
            pass
        else:
            roots.append(r)

    return mp_uniq(roots)

def intersect(a, b, axis=2):
    from numpy import logical_and, logical_or
    return unique(a[logical_or.reduce(logical_and.reduce(a == b[:,None], axis=axis))])

def mp_uniq(seq, tol=1e-20):
    '''Uniqfies a list subject to a certain tolerance. Order preserving.
    Adapted from http://www.peterbe.com/plog/uniqifiers-benchmark
    '''
    seq = array(seq)
    noDupes = [seq[0]]
    [ noDupes.append(i) for i in seq if min(abs(array(noDupes) - i)) > tol ]
    return noDupes

def go():
    r = make_grid(gridx, gridy)
    tr = trace_roots(r)
    guesses = tr[0][:,0]*scale - offsetx. + 1j*tr[0][:,1]*scale - offsety
    roots = get_roots(f, p, guesses)
    return (r, tr, guesses, roots)
