# -*- coding: utf-8 -*-
from PyDDE import pydde as p
from numpy import array, arange, pi, transpose

# to save results
from pickle import dump

dde_eg = p.dde()

def ddegrad(s, c, t):
    """Retorna o gradiente de x, o lado direito da equação a ser integrada"""

    # condição inicial constante 
    alag = 0.2
    
    if t > c[0]:
        # aqui entra o delay
        alag = p.pastvalue(0,t-c[0],0)

    r = 1.0
    if t > 6000:
        r -= r * 2e-3
    elif t > 2000:
        r -= r * 5e-7 * (t-2000)

    return array([ r * s[0] * (1.0 - alag) ])
    
def ddesthist(g, s, c, t):
    """Função para guardar variáveis "históricas". Neste caso, não faz nada."""
    return (s, g)

def rr(t):
    r = 1.0
    if t > 6000:
        r -= r * 2e-3
    elif t > 2000:
        r -= r * 5e-7 * (t-2000)
    return r

# condição inicial
ddeist = array([0.2])

# "state-scaling array for use in error control when values are very close to
# 0." ??
ddestsc = array([1e-10])

#pars = arange(3.7, 12.0, 1.0)
pars = [1.572]
result = []

for par in pars:

    dde_eg = p.dde()
    # finalmente, o cálculo propriamente dito
    dde_eg.dde(y=ddeist, times=arange(0.0, 8000.0, 0.2), 
        func=ddegrad, parms=array([ par ]),
        tol=1e-10, dt=0.1, hbsize=1e7, nlag=1, ssc=ddestsc)
    r = dde_eg.data
    result.append(r)
   
