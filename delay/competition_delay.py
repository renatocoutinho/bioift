'''Model for 2 competing species that have delayed growth response.'''
# -*- coding: utf-8 -*-
from PyDDE import pydde
from numpy import array, arange, pi, transpose, concatenate

def ddegrad(s, c, t):
    """Retorna o gradiente de x, o lado direito da equação a ser integrada"""

    # condição inicial constante 
    alag = [c[-2], c[-1]]
    
    if t > c[0]:
        # aqui entra o delay
        alag[0] = pydde.pastvalue(0, t-c[0], 0)
    if t > c[2]:
        alag[1] = pydde.pastvalue(1, t-c[2], 1)

    return array([ s[0] * (1.0 - alag[0] - c[1] * alag[1]), c[4] * s[1] * (1.0 - alag[1] - c[3] * alag[0]) ])
 
def solve(x0, c=[1.56, 0.9, 2.8, 0.9, 1.0], t=arange(0.0, 1000.0, 0.1)):
    '''Solves a system of DDEs for 2 species competing and showing delayed growth response.
            c[0]    retard of the 1st species
            c[1]    competing coefficient for 1st sp.
            c[2]    retard of the 2nd sp.
            c[3]    competing coefficient for 2nd sp.
            c[4]    ratio of intrinsic growth r2/r1
    '''
    dde_eg = pydde.dde()

    x0 = array(x0)

    # last constants in array are initial conditions
    c = concatenate((array(c), x0))
    
    dde_eg.dde(y=x0, times=t,
            func=ddegrad, parms=c,
            tol=1e-8, dt=0.1, hbsize=1e6, nlag=2, ssc=array([0, 0]))
    
    return dde_eg.data
    
