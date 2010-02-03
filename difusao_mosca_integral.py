# -*- coding: utf-8 -*-
"""Evolui uma equação a diferenças com difusão espacial dada por um kernel que modela uma população de moscas."""
import numpy as np
from scipy.integrate import romb

def diff_off_diag(p, u, v):
    """Equação a diferenças tipo logística com difusão espacial:
    u[t+1] (x) = a1 * v[t] * (1 - g1 * v[t])
    v[t+1] (x) = a2 * \int K(y-x) * u[t](y) dy

    Com K(y-x) := exp(-(y-x)^2 / s^2)

    u representa uma população de moscas e v a quantidade de larvas.
    """ 
    n = len(u) - 1
    uu = p['a1'] * v * (1 - p['g1'] * v)
    vv = np.zeros(v.size)
    for i in range(1, v.size):
        vv[i] = p['a2'] * romb(u * p['K'][n-i:-i], p['dx'])
    vv[0] =  p['a2'] * romb(u * p['K'][n:], p['dx'])
    return [uu, vv]

def ricker_int(p, u, v):
    """Equação a diferenças do tipo Ricker com difusão espacial:
    u[t+1] (x) = a1 * v[t] * exp(- g1 * v[t])
    v[t+1] (x) = a2 * \int K(y-x) * u[t](y) dy

    Com K(y-x) := exp(-(y-x)^2 / s^2)

    u representa uma população de moscas e v a quantidade de larvas.
    """ 
    n = len(u) - 1
    uu = p['a1'] * v * np.exp(- p['g1'] * v)
    vv = np.zeros(v.size)
    for i in range(1, v.size):
        vv[i] = p['a2'] * romb(u * p['K'][n-i:-i], p['dx'])
    vv[0] =  p['a2'] * romb(u * p['K'][n:], p['dx'])
    return [uu, vv]

def ricker_int_1d(p, u):
    """Equação a diferenças do tipo Ricker com difusão espacial:
    v[t+1] (x) = a * \int K(y-x) * v[t](y) * exp(-g * v[t](y)) dy

    Com K(y-x) := exp(-(y-x)^2 / s^2)

    v representa uma população de larvas.
    """ 
    n = len(u) - 1
    uu = np.zeros(u.size)
    for i in range(1, u.size):
        uu[i] = p['a'] * romb(u * np.exp(- p['g'] * u) * p['K'][n-i:-i], p['dx'])
    uu[0] =  p['a'] * romb(u * np.exp(- p['g'] * u) * p['K'][n:], p['dx'])
    return [uu]


def ricker_homogeneous(p, u, v):
    """Equação a diferenças do tipo Ricker *sem* difusão espacial:
    u[t+1] = a1 * v[t] * exp(- g1 * v[t])
    v[t+1] = a2 * u[t]

    Com K(y-x) := exp(-(y-x)^2 / s^2)

    u representa uma população de moscas e v a quantidade de larvas.
    """
    uu = p['a1'] * v * np.exp(- p['g1'] * v)
    vv = p['a2'] * u
    return [uu, vv]

def evolve(func, I, T, params):
    """Evolui uma equação a diferenças.

    Argumentos:
    - func: função de evolução do mapa.
    - I: condição inicial
    - T: número de iterações
    - params: dicionário com os parâmetros do mapa
    """
    T = int(T)
    r = np.zeros((T, len(I), len(I[0])))
    r[0] = I
    for t in range(T-1):
        r[t+1] = func(params, *r[t])
 
    return r

def runit():
    """Gera condição inicial e roda o programa com parâmetros comuns."""
    # domínio espacial
    L=8.0
    # n tem que ser potência de 2
    n = 2 ** 11
    xr = np.linspace(-L/2, L/2, n+1)
    
    # número de iterações
    T = 30

    # parâmetros
    p = {
            #'a1': 1.0,
            #'a2': 200.0,
            #'g1': 0.2,
            'a': 57.0,
            'g': 0.5,
            'dx': L/n
            }

    # largura da condição inicial
    l = 0.1

    # condição inicial gaussiana
    #X0 = [np.zeros(xrange.size), p['a2'] * np.exp(- xrange ** 2 / 20 / 1e-3) ] 

    # condição inicial quadrada
    quadrado = 2.0 * np.ones(xr.size)
    quadrado[:np.ceil(n*(1-l/L)/2)] = 0
    quadrado[-np.ceil(n*(1-l/L)/2):] = 0
    #X0 = [np.zeros(xrange.size), quadrado]
    X0 = [ quadrado ]

    ## kernel da evolução
    # largura da distribuição
    # é relacionada à taxa de dispersão D que tínhamos antes
    s = 0.1 
    p['K'] = np.exp( - (np.linspace(-L, L, 2*n+1)) ** 2 / s ** 2 ) / s / np.sqrt(np.pi)

    # resolvendo o sistema
    r = evolve(ricker_int_1d, X0, T, p)

    return p, r


def divplot(r, L, n, t=[0, 1, 2, 3], filename='teste1.png', ymax=25.0):
    """Plota para arquivo uma solução em vários instantes de tempo especificados."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

    fig = Figure((8,8), dpi=200)
    fig.patch.set_facecolor('w')
    ax = []
    xr = np.linspace(-L/2, L/2, n+1)

    xcut = 5/20
    # plota apenas larvas, a cada 2 iterações
    for i, ti in enumerate(t):
        ax.append(fig.add_subplot(len(t) * 100 + 10 + i + 1))
        ax[i].plot(xr[n*xcut:-n*xcut], r[ti][0][n*xcut:-n*xcut], '-')
        ax[i].set_ylabel("t = %d" % ti)
        ax[i].set_ylim(ymax=ymax)
        if i + 1 < len(t):
            ax[i].set_xticklabels([])
    ax[-1].set_xlabel(u"position")
    canvas = FigureCanvas(fig)
    canvas.print_png(filename)

# é executado quando rodamos o arquivo diretamente
if __name__ == '__main__':
    s = 0.1
    T = 10
    (p, r) = runit()

    ### plotting ###
    
    # string contendo os parâmetros
    #s_params = ''
    #for k, v in p.items():
    #    if type(v) != np.ndarray:
    #        s_params += "%s: %s, " % (k, v)
    #s_params += "s: %s, L: %s, n: %s" % (s, L, n)


    ## plotando na tela

    from pylab import plot, show, legend, xlabel, ylabel, figtext

    # plota população de larvas
    for i in range(T):
        plot(xrange[5*n/20:-5*n/20], r[i][0][5*n/20:-5*n/20], '-', label=str(i))
    legend(loc='best')
    xlabel(u"posição")
    ylabel(u"larvas")
    #figtext(0.3, 0.03, s_params)

    show()

#    ## plotando pra arquivo
#
#    from matplotlib.figure import Figure
#    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
#
#    fig = Figure((8,8), dpi=200)
#    ax = fig.add_subplot(111)
#    
#    filename = "quadrado.png" % (D)
#    
#    # plota apenas larvas, a cada 2 iterações
#    for i in range(0, 13, 2):
#        ax.plot(xrange[9*n/20:-9*n/20], r[i][1][9*n/20:-9*n/20], '-', label=str(i))
#    ax.set_xlabel("posicao")
#    ax.set_ylabel("larvas")
#    ax.text(0.3, 0, s_params)
#    canvas = FigureCanvas(fig)
#    canvas.print_png(filename)

