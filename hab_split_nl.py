# -*- coding: utf-8 -*-
from __future__ import division
from numpy import array, arange, vectorize, sqrt, cosh, sinh, tanh, exp, poly1d, isnan, nan, linspace, logspace, log, log10
from scipy.integrate import quad
from scipy.optimize import fsolve, brentq

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
    return p['d'] - quad(g, A2(k), r(p)*k, args=(A2(k),p))[0]

def g(x, A2, p=p):
    return 1/sqrt( (p['mA']*(-A2**2 + x**2) + 2*p['g']/3*(-A2**3 + x**3)) / p['DA'] )

def A2(k, p=p, extra_par={}, all=False):
    old_p = p.copy()
    p.update(extra_par)
    a = 2*p['g']/3/p['DA']
    b = p['mA']/p['DA']
    rr = r(p)
    d = (1 - b*rr**2)*k**2 - a*rr**3*k**3
    p.update(old_p)

#    x1 = -b/(3*a) - (2**(1/3)*(-b**2))/(3*a*cubic_root(-2*b**3 - 27*a**2*d + sqrt(4*(-b**2)**3 + (-2*b**3 - 27*a**2*d)**2))) + cubic_root(-2*b**3 - 27*a**2*d + sqrt(4*(-b**2)**3 + (-2*b**3 - 27*a**2*d)**2))/(3*2**(1/3)*a)
#    x2 = -b/(3*a) + ((1 + 1j*sqrt(3))*(-b**2))/(3*2**(2/3)*a*cubic_root(-2*b**3 - 27*a**2*d + sqrt(4*(-b**2)**3 + (-2*b**3 - 27*a**2*d)**2))) - (1 - 1j*sqrt(3))*cubic_root(-2*b**3 - 27*a**2*d + sqrt(4*(-b**2)**3 + (-2*b**3 - 27*a**2*d)**2))/(6*2**(1/3)*a)
#    x3 = -b/(3*a) + ((1 - 1j*sqrt(3))*(-b**2))/(3*2**(2/3)*a*cubic_root(-2*b**3 - 27*a**2*d + sqrt(4*(-b**2)**3 + (-2*b**3 - 27*a**2*d)**2))) - (1 - 1j*sqrt(3))*cubic_root(-2*b**3 - 27*a**2*d + sqrt(4*(-b**2)**3 + (-2*b**3 - 27*a**2*d)**2))/(6*2**(1/3)*a)

    pol = poly1d([a, b, 0, d])

    if all:
        return pol.r
    else:
        roots = []
        for x in pol.r:
            if x.imag == 0:
                roots.append(x.real)
        return max(roots)

def r(p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    result = ((1-p['b'])*cosh(sqrt(p['mJ']/p['DJ'])*p['L1']) - p['a'])/ sinh(sqrt(p['mJ']/p['DJ'])*p['L1']) / sqrt(p['mJ']/p['DJ']) / (1-p['b'])
    p.update(old_p)
    return result

def A(x, k, p=p, check=False):
    assert x >= 0
    assert x <= p['L1'] + p['d']
    if check:
        check_consistency(k, p)

    if x <= p['L1']:
        wJ = sqrt(p['mJ']/p['DJ'])
        wA = sqrt(p['mA']/p['DA'])
        c1 = k*(1-p['b'] - p['a']*exp(-wJ*p['L1']))/2/wJ/sinh(wJ*p['L1'])
        c2 = k*(1-p['b'] - p['a']*exp(+wJ*p['L1']))/2/wJ/sinh(wJ*p['L1'])
        return c1 * exp(wJ*x) + c2 * exp(-wJ*x)
    else:
        return brentq(lambda y: p['L1'] + p['d'] - x - quad(g, A2(k), y, args=(A2(k),p))[0],
                A2(k)+1e-9, r(p)*k)

def solution(k, p=p, npoints=200, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    check_consistency(k, p)
    x1 = linspace(0, p['L1'], npoints/2)
    J = vectorize(lambda x: A(x, k))(x1)
    
    AA2 = linspace(A2(k, p), r(p)*k, npoints/2)
    x2 = [p['L1'] + p['d']]
    for i in range(len(AA2)-1):
        x2.append(x2[i] - quad(g, AA2[i], AA2[i+1], args=(A2(k),p))[0])

    sol = zip(x1, J) + zip(x2[::-1], AA2[::-1])
    p.update(old_p)
    return array(sol)

def check_consistency(k=None, p=p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    assert p['a'] / (1. - p['b']) > cosh(sqrt(p['mJ']/p['DJ'])*p['L1'])
    if k is not None:
        assert r(p) * k > 0
        # k < (1-b*r**2)/r**3
        assert A2(k) > 0
        assert k*r(p) > A2(k)
    p.update(old_p)

def find_k(k0=None, p=p, extra_par={}):
    old_p = p.copy()
    p.update(extra_par)
    #return fsolve(f, k0, args=p)
    if not k0:
        a = 2*p['g']/3/p['DA']
        b = p['mA']/p['DA']
        rr = r(p)
        k0 = (1-b*rr**2)/a/rr**3
    try:
        k = brentq(f, k0, 2*k0, args=p)
    except ValueError:
        k = fsolve(f, k0 - 1e-6, args=p)
    p.update(old_p)
    return k

def varia_p(param, values, p=p):
    orig = p[param]
    result = []
    for x in values:
        p[param] = x
        k = find_k(p=p)
        try:
            check_consistency(k, p)
        except AssertionError:
            result.append(nan)
        else:
            result.append(k)
    p[param] = orig
    return array(result)

