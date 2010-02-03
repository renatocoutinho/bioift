'''Model for a species that has delayed growth response and is subject to a oscillating environmenting through a oscillating carrying capacity.'''
# -*- coding: utf-8 -*-
from PyDDE import pydde
from numpy import array, arange, pi, transpose, concatenate, cos

def ddegrad(s, c, t):
    """Retorna o gradiente de x, o lado direito da equação a ser integrada"""
    # condição inicial constante 
    alag = c[-1]
    
    if t > c[0]:
        # aqui entra o delay
        alag = pydde.pastvalue(0, t-c[0], 0)

    return array([ - alag * ( 1.0 + (1.0 + c[1] * cos(t)) * s[0] ) ])
 
def dde_integrate(x0, c=[1.7, 0.1], t=arange(0.0, 1000.0, 0.1)):
    '''Solves a DDE for a species with delayed growth response and oscillating carrying capacity.
            c[0]    retard of the 1st species
            c[1]    oscillation amplitude
    '''
    dde_eg = pydde.dde()

    # last constants in array are initial conditions
    x0 = array([x0])
    c = concatenate((array(c), x0))
    
    dde_eg.dde(y=x0, times=t,
            func=ddegrad, parms=c,
            tol=1e-8, dt=t[1]-t[0], hbsize=1e6, nlag=1, ssc=array([0]))
    
    return dde_eg.data

def batch_sigma(eps=0.1, sigmas=arange(0, 2.51, 0.05)):
    results = []
    for sigma in sigmas:
        # finalmente, o cálculo propriamente dito
        par = pi/2.0 + sigma * eps ** 2
        results.append(dde_integrate(x0=0.1, c=[par, eps], t=arange(0, 5000, 0.1)))

    return results

