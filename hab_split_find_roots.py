# -*- coding: utf-8 -*-
#from sympy.mpmath import *
import sympy.mpmath as mp
#from numpy import *
from numpy import arange, array, vectorize, zeros, where, abs, concatenate, transpose
from pylab import plot, xlabel, ylabel, legend, scatter, show
from scipy.stats._support import unique
from scipy import optimize as op
from itertools import product as iproduct
import multiprocessing

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
scale = 0.01
minx = -10.
maxx = 10.
miny = -0.02
maxy = 10.

def F(x, p=p):
    return (1 - p['b'] * mp.exp(-x * p['t2'])) * mp.exp(x * p['t1']) \
            * (mp.cosh(mp.sqrt((p['mJ'] + x)/p['DJ']) * p['L1']) \
            + mp.sinh(mp.sqrt((p['mJ'] + x)/p['DJ']) * p['L1']) \
            * mp.sqrt((p['mJ'] + x)/p['DJ']) / mp.sqrt((p['mA'] + x)/p['DA']) \
            * mp.coth(mp.sqrt((p['mA'] + x)/p['DA']) * p['d'])) - p['a']

# identical to above, to be used in multprocessing. The function must be pickle-friendly.
def fmult(x, p=p):
    return ( x[0], x[1], F(x[0] + 1j*x[1], p=p) )

def M(x, l, p=p):
    return mp.exp(mp.sqrt((p['mJ'] + l)/p['DJ']) * x) + \
            ((1 - p['b'] * mp.exp(-x * p['t2'])) * mp.exp(x * p['t1']) - p['a'] * mp.exp(mp.sqrt((p['mJ'] + l)/p['DJ']) * p['L1'])) / \
            ((1 - p['b'] * mp.exp(-x * p['t2'])) * mp.exp(x * p['t1']) - p['a'] * mp.exp(- mp.sqrt((p['mJ'] + l)/p['DJ']) * p['L1'])) \
            * mp.exp(- mp.sqrt((p['mJ'] + l)/p['DJ']) * x)

def N(x, l, p=p):
    '''Incomplete!!'''
    return 2. * mp.sinh(sqrt((p['mJ'] + l)/p['DJ']) * p['L1']) \
            * (mp.exp(mp.sqrt((p['mA'] + l)/p['DA']) * x) + \
            mp.exp(mp.sqrt((p['mA'] + l)/p['DA']) * (2*p['L2'] - x)))

def mp_real(x):
    return vectorize(lambda y: y.real)(x)

def mp_imag(x):
    return vectorize(lambda y: y.imag)(x)

def mp_to_float(x):
    def get_real(y):
        if type(y) == mpc:
            if y.imag == mpf(0):
                return y.real
            else:
                raise ValueError
        else:
            return mp.mpf(y)

    return vectorize(get_real)(x)

def mp_solve(f, p, x0=0.01, limits=None, method='muller'):
    if not limits:
        return mp.findroot(lambda x: f(x, p=p), x0, solver=method)
    else:
        if method == 'muller':
            method = 'anderson'
        return mp.findroot(lambda x: f(x, p=p), limits, solver=method)

def solve(f, p, x0=0.01, limits=[1e-12, 5.], method='fsolve'):
    if method == 'brentq':
        return op.brentq(f, limits[0], limits[1], args=p)
    elif method == 'fsolve':
        return op.fsolve(f, x0=x0, args=p)

def make_grid(scale, minx, maxx, miny, maxy, mult=True):
    gridx = arange(minx, maxx, scale)
    gridy = arange(miny, maxy, scale)

    if mult:
        pool = multiprocessing.Pool(None)
        tasks = list(iproduct(gridx, gridy))
        results = []
    
        r = pool.map_async(fmult, tasks, callback=results.append)
        r.wait() # Wait on the results
        
        rr = zeros((len(gridx), len(gridy)), dtype='object')
        def insert(x):
            rr[int((x[0] - minx)/scale)][int((x[1] - miny)/scale)] = x[2]
    
        for x in results[0]:
            insert(x)

        return rr
    else:
        return array([ [ F(i + j*1j) for j in gridy ] for i in gridx ])

def roots_plot(r, part='real', scale=0.05, offset=10.):
    if part == 'real':
        p = mp_real(r) > zeros(r.shape)
    elif part == 'imag':
        p = mp_imag(r) > zeros(r.shape)
    pT = where(p == True)
    pF = where(p == False)
    scatter(scale*pT[0] - 10, scale*pT[1] - offset, c='b')
    scatter(scale*pF[0] - offset, scale*pF[1] - offset, c='r')
    xlabel('Re')
    ylabel('Im')
    show()

def trace_roots(r, scale=0.05, offsetx=10., offsety=0):
    rroots = []
    iroots = []
    for i in range(r.shape[0] - 1):
        for j in range(r.shape[1] - 1):
            if r[i][j].real * r[i+1][j].real < 0:
                rroots.append([i, j])
                rroots.append([i+1, j])
                #rroots.append([i+0.5, j])
            if r[i][j].real * r[i][j+1].real < 0:
                rroots.append([i, j])
                rroots.append([i, j+1])
                #rroots.append([i, j+0.5])
            if r[i][j].imag * r[i+1][j].imag < 0:
                iroots.append([i, j])
                iroots.append([i+1, j])
                #iroots.append([i+0.5, j])
            if r[i][j].imag * r[i][j+1].imag < 0:
                iroots.append([i, j])
                iroots.append([i, j+1])
                #iroots.append([i, j+0.5])
    rroots = array(rroots)
    iroots = array(iroots)
    return (intersect(rroots, iroots), rroots, iroots)

def get_roots(f, p, guesses, methods=['muller','secant']):
    roots = []
    for x in guesses:
        for m in methods:
            try:
                r = mp_solve(f, p, x0=x, method=m)
            except:
                pass
            else:
                roots.append(r)

    return mp_uniq(roots)

# definitely not working!
def intersect(a, b, axis=2):
    from numpy import logical_and, logical_or
    return unique(a[logical_or.reduce(logical_and.reduce(a == b[:,None], axis=axis))])

def mp_uniq(seq, tol=1e-16):
    '''Uniqfies a list subject to a certain tolerance. Order preserving.
    Adapted from http://www.peterbe.com/plog/uniqifiers-benchmark
    '''
    seq = array(seq)
    noDupes = [seq[0]]
    [ noDupes.append(i) for i in seq if min(abs(array(noDupes) - i)) > tol ]
    return noDupes

def c_plot(x, *args, **kwargs):
    as_ri = lambda y: [ y.real, y.imag if type(y) == mpc else 0 ]
    z = array([ as_ri(y) for y in x ])
    return plot(z[:,0], z[:,1], *args, **kwargs)

def get_root_seq(f, p, x0, params, methods=['secant']):
    param = params[0]
    ps = params[1]
    roots = [x0]
    for x in ps[1:]:
        p[param] = x
        roots.append(get_roots(f, p, [roots[-1]], methods)[0])
    # reset params dict
    p[param] = ps[0]
    return [ ps, roots ]

def get_real_roots(f, p, limits=[-0., 20.], scale=1e-2, method='anderson'):
    x = arange(limits[0], limits[1], scale)
    v = mp_to_float(vectorize(f)(x))
    r = []
    for i in where(v[1:] * v[:-1] < 0)[0]:
        r.append(mp_solve(f, p, limits=[x[i], x[i+1]], method=method))
    return array(r)

def get_all_roots_seq(f, p, x0, params, limits=[0., 20.], scale=5*1e-3, methods=['secant']):
    resultc = []
    for x in x0:
        resultc.append(get_root_seq(f, p, x, params, methods=['secant'])[1])

    resultr = []
    param = params[0]
    ps = params[1]
    for x in ps:
        p[param] = x
        resultr.append(get_real_roots(f, p, limits, scale))

    # reset params dict
    p[param] = ps[0]
    return [ array([ps]), transpose(array(resultr)), array(resultc) ]

def zero_stable(f, p, extra_par={}, limits=[-0., 30.], res=1e-2):
    old_p = p.copy()
    p.update(extra_par)
    x = arange(limits[0], limits[1], res)
    v = vectorize(lambda y: f(y, p=p))(x)
    p.update(old_p)
    return v.max() * v.min() > 0

def find_bif(f=F, p=p, par='L1', plimits=[0.06, 0.07], res=1e-3, limits=[0., 1.]):
    return op.bisect(lambda y: -0.5 + int(zero_stable(F, p, extra_par={par: y}, limits=limits, res=res)), plimits[0], plimits[1], xtol=res)

def go():
    r = make_grid(scale, minx, maxx, miny, maxy)
    tr = trace_roots(r)
    guesses = tr[0][:,0]*scale + minx + 1j*tr[0][:,1]*scale + miny
    roots = get_roots(F, p, guesses)
    return (r, tr, guesses, roots)

