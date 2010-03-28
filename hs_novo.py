# -*- coding: utf-8 -*-
from numpy import array, arange, vectorize
#from numpy import exp, cosh, sinh, tanh, sqrt
from sympy import exp, cosh, sinh, tanh, sqrt
from pylab import plot, show, legend, xlabel, ylabel
import sympy.mpmath as mp

p = {
        'r': 2.,        # r
        't1': 2.,       # t_1
        'mJ': 1.,       # \mu_J
        'mA': 0.001,    # \mu_A
        'DJ': 1.,       # D_J
        'DA': 1.,       # D_A
        'd': 1,         # d := L_2 - L_1
        'L1': 1.,       # L_1
        'K': 1.         # K
    }

def G(l, p=p, extra_par={}):
    old_p = p
    p.update(extra_par)
    wJ = sqrt((p['mJ'] + l)/p['DJ'])
    wA = sqrt((p['mA'] + l)/p['DA'])
    result = exp(l*p['t1']) * (cosh(wJ*p['L1']) + wA/wJ * tanh(wA*p['d']) * sinh(wJ*p['L1']))
    p.update(old_p)
    return result

def J(x, p=p, extra_par={}):
    old_p = p
    p.update(extra_par)
    assert x >= 0
    assert x <= p['L1']
    wJ = sqrt((p['mJ'])/p['DJ'])
    wA = sqrt((p['mA'])/p['DA'])
    result = exp(wJ*(x-p['L1'])) / 2. * (1 - wA/wJ*tanh(wA*p['d'])) *\
            (1/(cosh(wJ*p['L1']) + wA/wJ * tanh(wA*p['d']) * sinh(wJ*p['L1'])) -p['K']/p['r'])\
            + exp(-wJ*(x-p['L1'])) / 2. * (1 + wA/wJ*tanh(wA*p['d'])) *\
            (1/(cosh(wJ*p['L1']) + wA/wJ * tanh(wA*p['d']) * sinh(wJ*p['L1'])) -p['K']/p['r'])
    p.update(old_p)
    return result

def A(x, p=p, extra_par={}):
    old_p = p
    p.update(extra_par)
    assert x >= p['L1']
    assert x <= p['L1'] + p['d']
    wJ = sqrt((p['mJ'])/p['DJ'])
    wA = sqrt((p['mA'])/p['DA'])
    result = cosh(wA*(p['L1'] + p['d'] - x)) / cosh(wA*p['d']) *\
            (1/(cosh(wJ*p['L1']) + wA/wJ * tanh(wA*p['d']) * sinh(wJ*p['L1'])) -p['K']/p['r'])
    p.update(old_p)
    return result

def sol(x, p=p, extra_par={}):
    assert x >= 0
    assert x <= p['L1'] + p['d']
    if x <= p['L1']:
        return J(x, p, extra_par)
    else:
        return A(x, p, extra_par)

def check_consistency(p=p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    wJ = sqrt(p['mJ']/p['DJ'])
    wA = sqrt(p['mA']/p['DA'])
    assert p['r']/p['K'] >= 1.
    assert p['K']/p['r'] * (cosh(wJ*p['L1']) + wA/wJ * tanh(wA*p['d']) * sinh(wJ*p['L1'])) <= 1.
    p.update(old_p)

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

def get_real_roots(f, p, limits=[-0., 20.], scale=1e-2, method='anderson'):
    x = arange(limits[0], limits[1], scale)
    v = mp_to_float(vectorize(f)(x))
    r = []
    for i in where(v[1:] * v[:-1] < 0)[0]:
        r.append(mp_solve(f, p, limits=[x[i], x[i+1]], method=method))
    return array(r)

if __name__ == '__main__':
    check_consistency()

