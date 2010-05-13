# -*- coding: utf-8 -*-
from numpy import array, arange, vectorize, where
#from numpy import exp, cosh, sinh, tanh, sqrt
from sympy.mpmath import exp, cosh, sinh, tanh, sqrt, mpf, mpc
from pylab import plot, show, legend, xlabel, ylabel
import sympy.mpmath as mp
from scipy import integrate

p = {
        'r': 2.,        # r
        't1': 1.,       # t_1
        'mJ': 1.,       # \mu_J
        'mA': 0.001,    # \mu_A
        'DJ': 1.,       # D_J
        'DA': 1.,       # D_A
        'd': 1.,        # d := L_2 - L_1
        'L1': 1.,       # L_1
        'K': 1.,        # K
        'a': 0          # \alpha := 1/h
    }

def G(l, p=p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    wJ = sqrt((p['mJ'] + l)/p['DJ'])
    wA = sqrt((p['mA'] + l)/p['DA'])
    result = exp(l*p['t1']) * (cosh(wJ*p['L1']) + wA/wJ * tanh(wA*p['d']) * sinh(wJ*p['L1']))
    p.update(old_p)
    return result

def J(x, p=p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    assert x >= 0
    assert x <= p['L1']
    wJ = sqrt((p['mJ'])/p['DJ'])
    wA = sqrt((p['mA'])/p['DA'])
    result = p['K'] * exp(wJ*(x-p['L1'])) / 2. * (1 - wA/wJ*tanh(wA*p['d'])) *\
            (1/(cosh(wJ*p['L1']) + wA/wJ * tanh(wA*p['d']) *\
            sinh(wJ*p['L1'])) - 1./(p['r']*p['d']*exp(-p['a']*p['L1'])))\
            + p['K'] * exp(-wJ*(x-p['L1'])) / 2. * (1 + wA/wJ*tanh(wA*p['d'])) *\
            (1/(cosh(wJ*p['L1']) + wA/wJ * tanh(wA*p['d']) *\
            sinh(wJ*p['L1'])) -1./(p['r']*p['d']*exp(-p['a']*p['L1'])))
    p.update(old_p)
    return result

def A(x, p=p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    assert x >= p['L1']
    assert x <= p['L1'] + p['d']
    wJ = sqrt((p['mJ'])/p['DJ'])
    wA = sqrt((p['mA'])/p['DA'])
    result = cosh(wA*(p['L1'] + p['d'] - x)) / cosh(wA*p['d']) *\
            (1/(cosh(wJ*p['L1']) + wA/wJ * tanh(wA*p['d']) *\
            sinh(wJ*p['L1'])) - 1./(p['r']*p['d']*exp(-p['a']*p['L1'])))
    p.update(old_p)
    return result

def intA(p=p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    result = integrate.quad(lambda y: A(y, p=p), p["L1"], p["L1"]+p["d"])
    p.update(old_p)
    return result

def sol(x, p=p, extra_par={}):
    assert x >= 0
    assert x <= p['L1'] + p['d']
    if x <= p['L1']:
        return J(x, p, extra_par)
    else:
        return A(x, p, extra_par)

def Lcrit(p=p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    f = lambda y: G(0, p=p, extra_par={'L1': y}) - p['r']
    result = mp_solve(f, limits=[0., 10.])
    p.update(old_p)
    return result

def check_consistency(p=p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    wJ = sqrt(p['mJ']/p['DJ'])
    wA = sqrt(p['mA']/p['DA'])
    assert p['r'] >= 1.
    assert 1./p['r'] * (cosh(wJ*p['L1']) + wA/wJ * tanh(wA*p['d']) * sinh(wJ*p['L1'])) <= 1.
    p.update(old_p)

def consistency_condition(par='L1', p=p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
#    check_consistency(p=p)
    g = lambda x: G(0, p=p, extra_par={par: x}) - p['r']
    result = mp.findroot(g, p[par], solver='secant')
    p.update(old_p)
    return result

def mp_real(x):
    return vectorize(lambda y: y.real)(x)

def mp_imag(x):
    return vectorize(lambda y: y.imag)(x)

def mp_real(x):
    if type(x) == mpc:
        if x.imag == mpf(0):
            return mpf(x.real)
        else:
            raise ValueError
    else:
        return mpf(x)

def mp_to_float(x):
    return vectorize(mp_real)(x)

def mp_solve(f, x0=0.01, limits=None, method='muller'):
    if not limits:
        return mp.findroot(f, x0, solver=method)
    else:
        if method == 'muller':
            method = 'anderson'
        return mp.findroot(f, limits, solver=method)

def get_real_roots(f, limits=[-4., 4.], scale=1e-2, method='anderson'):
    x = arange(limits[0], limits[1], scale)
    g = lambda x: f(x)
    v = vectorize(g)(x)
    r = []
    for i in where(v[1:] * v[:-1] < 0)[0]:
        try:
            s = mp_solve(g, limits=[x[i], x[i+1]], method=method)
        except ValueError:
            s = None
        else:
            r.append(s)
    return array(r)

def ffp(x, p=p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    result = mp_real(G(x, p=p) - p['K'] / p['r'] * G(0, p=p)**2)
    p.update(old_p)
    return result

def f0(x, p=p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    result = mp_real(G(x, p=p) - p['r'] / p['K'])
    p.update(old_p)
    return result

if __name__ == '__main__':
    check_consistency()
