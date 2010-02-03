# -*- coding: utf-8 -*-
"""Integra as equações para uma rede de "caranguejos" acoplados.

É sorteada uma matriz de caranguejos, em que cada um está voltado pra uma certa
direção, podendo ver 3 de seus vizinhos 8 vizinhos imediatos. Cada caranguejo
oscila com frequênca \omega, e tem sua fase acoplada à oscilação daqueles
vizinhos que são visíveis.
"""
import numpy as np
import random
from scipy.integrate import odeint

def ij2n(i, j, L):
    '''Converte índices (i,j) para posição num vetor linear de tamanho L^2.'''
    return L * (i%L) + (j%L)

def n2ij(n, L):
    '''Converte índice linear n numa posição (i,j) no plano.'''
    return (n // L, n % L)

def topol_quadrante(q, i, j, L):
    '''Calcula os índices dos 3 vizinhos acoplados em um quadrante escolhido.'''
    if q == 0:
        return [ ij2n(i, j+1, L), ij2n(i-1, j, L), ij2n(i-1, j+1, L) ]
    elif q == 1:
        return [ ij2n(i-1, j, L), ij2n(i, j-1, L), ij2n(i-1, j-1, L) ]
    elif q == 2:
        return [ ij2n(i+1, j, L), ij2n(i, j-1, L), ij2n(i+1, j-1, L) ]
    elif q == 3:
        return [ ij2n(i+1, j, L), ij2n(i, j+1, L), ij2n(i+1, j+1, L) ]

def topol_vizinho(q, i, j, L):
    '''Calcula o índice do vizinho acoplado dada uma direção.'''
    if q == 0:
        if j+1 < L:
            return [ ij2n(i, j+1, L) ]
        else:
            return [ ij2n(i, j, L) ]
    elif q == 1:
        if i-1 >= 0:
            return [ ij2n(i-1, j, L) ]
        else:
            return [ ij2n(i, j, L) ]
    elif q == 2:
        if j-1 >= 0:
            return [ ij2n(i, j-1, L) ]
        else:
            return [ ij2n(i, j, L) ]
    elif q == 3:
        if i+1 < L:
            return [ ij2n(i+1, j, L) ]
        else:
            return [ ij2n(i, j, L) ]

def plot_plane(m, L):
    '''Plota diagrama de flechas com a direção de visão dos caranguejos.'''
    import itertools as it
    from pylab import scatter, Arrow, gca
    ax = gca()
    ax.scatter(*(np.array(list(it.product(range(L), range(L)))).transpose()), s=1)

    for i in range(L):
        for j in range(L):
            pi, pj = n2ij(m[i][j][0], L)
            ax.add_patch(Arrow(i, j, pi - i, pj - j, 0.4, edgecolor='white'))

def plot_phases(times, r, N, total=False):
    '''Plota a soma das diferenças de fase entre cada caranguejo e o primeiro.'''
    from pylab import plot
    if total:
        plot(times, np.sum( np.abs(np.repeat(r[:,0], N-1).reshape((len(times), N-1)) - r[:,1:]), axis=1))
    else:
        plot(times, np.sum( np.abs(np.repeat(np.mod(r[:,0], 2*np.pi), N-1).reshape((len(times), N-1)) - np.mod(r[:,1:], 2*np.pi)), axis=1), '.-')

if __name__ == '__main__':
    L = 10 # tamanho da rede de caranguejos
    N = L**2
    w = 1.0 # frequência de oscilação
    c = 0.1 # grau de acoplamento

    # distribuição NÂO-ALEATÓRIA (manter resultado consistente)
    random.seed(0)

    ## distribuição de acoplamentos

    # olha pra um quadrante (3 vizinhos)
    #m = np.array([ [ topol_quadrante(random.choice(range(4)), i, j, L) for j in range(L) ] for i in range(L) ])
    
    # olha pra apenas um vizinho
    m = np.array([ [ topol_vizinho(random.choice(range(4)), i, j, L) for j in range(L) ] for i in range(L) ])

    # condição inicial (fase aleatória)
    X0 = np.array([ random.uniform(0, 2.0*np.pi) for x in range(N) ])
    
    def flux(r, t):
        viz = r[m]
        return w + c * np.sum(np.sin(viz - np.repeat(r, viz.shape[2]).reshape(L,L,viz.shape[2])), axis=2).reshape(N)

    times = np.arange(0, 500, 0.01)
    r = odeint(flux, X0, times)

    ## plotting
    from pylab import show
    #plot_plane(m, L)
    #plot_phases(times, r, N)
    #show()
