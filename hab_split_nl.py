# -*- coding: utf-8 -*-
from __future__ import division
from numpy import array, arange, sqrt, cosh, sinh, tanh, exp, poly1d, isnan
from scipy.integrate import quad
from scipy.optimize import fsolve

p = {
        'a': 1.1,
        'b': 0.,
        't1': 2.,
        't2': 2.,
        'mJ': 0.1,
        'mA': 0.001,
        'DJ': 1.,
        'DA': 1.,
        'd': 100,
        'L1': 1.,
        'g': 0.1
    }


cubic_root = lambda x: x**(1/3) if x > 0 else -(-x)**(1/3)

def f(k, p=p):
    return p['d'] - quad(g, A2(k), r(p)*k, args=(A2(k),p))

def g(x, A2, p=p):
    return 1/sqrt( (p['m']*(-A2**2 + x**2) + 2*p['g']/3*(-A2**3 + x**3)) / p['DA'] )

def A2(k, p=p, all=False):
    a = 2*p['g']/3/p['DA']
    b = p['mA']/p['DA']
    rr = r(p)
    d = (1 - b*rr**2)*k**2 - a*rr**3*k**3

    x1 = -b/(3*a) - (2**(1/3)*(-b**2))/(3*a*cubic_root(-2*b**3 - 27*a**2*d + sqrt(4*(-b**2)**3 + (-2*b**3 - 27*a**2*d)**2))) + cubic_root(-2*b**3 - 27*a**2*d + sqrt(4*(-b**2)**3 + (-2*b**3 - 27*a**2*d)**2))/(3*2**(1/3)*a)

    x2 = -b/(3*a) + ((1 + 1j*sqrt(3))*(-b**2))/(3*2**(2/3)*a*cubic_root(-2*b**3 - 27*a**2*d + sqrt(4*(-b**2)**3 + (-2*b**3 - 27*a**2*d)**2))) - (1 - 1j*sqrt(3))*cubic_root(-2*b**3 - 27*a**2*d + sqrt(4*(-b**2)**3 + (-2*b**3 - 27*a**2*d)**2))/(6*2**(1/3)*a)

    x3 = -b/(3*a) + ((1 - 1j*sqrt(3))*(-b**2))/(3*2**(2/3)*a*cubic_root(-2*b**3 - 27*a**2*d + sqrt(4*(-b**2)**3 + (-2*b**3 - 27*a**2*d)**2))) - (1 - 1j*sqrt(3))*cubic_root(-2*b**3 - 27*a**2*d + sqrt(4*(-b**2)**3 + (-2*b**3 - 27*a**2*d)**2))/(6*2**(1/3)*a)

    if all:
        return [x1, x2, x3]
    else:
        assert not isnan(x1)
        return x1

def r(p):
    return ((1-p['b'])*cosh(sqrt(p['mJ']/p['DJ'])*p['L1']) - p['a'])/ sinh(sqrt(p['mJ']/p['DJ'])*p['L1']) / sqrt(p['mJ']/p['DJ']) / (1-p['b'])

def check_consistency(k, p=p):
    assert p['a'] / (1. - p['b']) > cosh(sqrt(p['mJ']/p['DJ'])*p['L1']), "diff is %f" % (p['a'] / (1. - p['b']) - cosh(sqrt(p['mJ']/p['DJ'])*p['L1']))
    assert r(p) * k > 0
    assert A2(k) > 0
    assert k*r(p) > A2(k)

def find_k(k0=-1., p=p):
    return fsolve(f, k0, args=p)

