# -*- coding: utf-8 -*-
"""Integra a equação de Fermi-Pasta-Ulam.

Equação FPU tipo alfa:

    d^2 u[n] / dt^2 = u[n+1] + u[n-1] - 2 * u[n] + a * ( (u[n+1] - u[n])^2 - (u[n] - u[n-1])^2 )

Com condições de contorno u[0] = u[N] = 0

"""
import sys
import numpy as np
from scipy.integrate import odeint

def fpu_flux(r, t):

    # parameters
    a = 0
    # array size
    N = 32

    u = np.zeros(2*N)
    u[:N] = r[N:]
    u[N] = r[1] - 2 * r[0] + a * ((r[1] - r[0]) ** 2 - r[0] ** 2)
    u[N+1:-1] = r[2:N] + r[:N - 2] - 2 * r[1:N - 1] + a * ((r[2:N] - r[1:N - 1]) ** 2 - (r[1:N - 1] - r[:N - 2]) ** 2)
    u[-1] = r[N - 2] - 2 * r[N - 1] + a * (r[N - 1] ** 2 - (r[N - 1] - r[N - 2]) ** 2)
    return u


def plot_solution(r, filename='fpu_tmp.png'):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    # make a square figure and axes
    fig = Figure(figsize=(7,7), dpi=200)
    ax = fig.add_axes([0.15, 0.1, 0.65, 0.8])
    # set background color to white
    fig.patch.set_facecolor('w')

    for i in range(0, 500, 50):
        ax.plot(np.arange(0, 32, 1), r[i][:32], label=str(i*20))
    ax.legend(loc=(1.03,0.2))
    ax.set_xlabel("posicao")
    ax.set_ylabel("amplitude")
    canvas = FigureCanvas(fig)
    canvas.print_png(filename)


def fig():
    from matplotlib.figure import Figure
    # make a square figure and axes
    fig = Figure(figsize=(7,7), dpi=200)
    ax = fig.add_axes([0.15, 0.1, 0.65, 0.8])
    # set background color to white
    fig.patch.set_facecolor('w')
    return (fig, ax)

def save_fig(fig, filename):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    canvas = FigureCanvas(fig)
    canvas.print_png(filename)

if __name__ == '__main__':
    N = 32
    Nmodes = 5

    X0 = np.concatenate(( np.sin(np.pi * np.arange(1.0, N + 0.5, 1.0)/(N+1)), np.zeros(N) ))  # FPU initial condition

    # condições iniciais retiradas de 
    # http://www.scholarpedia.org/article/Fermi-Pasta-Ulam_nonlinear_lattice_oscillations

    # a=1; b(I)=a*sin(pi*N*I/(N+1)); b(I+N)=0; # Zabusky-Deem init. cond.
    # k=0.8; sk=(sinh(k))\verb!^!2; ek=exp(-k); i1=I-N/4; i2=i1-N/2; # Solitons
    # b(I)=-0.5/alpha*log((1+exp(2*k*(i1-1)))/(1+exp(2*k*i1)));  # Kink
    # b(I)=b(I)+0.5/alpha*log((1+exp(2*k*(i2-1)))/(1+exp(2*k*i2)));  # Anti-kink
    # b(I+N)= sk*ek/alpha/cosh(k*i1)/(exp(-k*i1)+exp(k*i1)*exp(-2*k));
    # b(I+N)=b(I+N)-sk*ek/alpha/cosh(k*i2)/(exp(-k*i2)+exp(k*i2)*exp(-2*k));
    # omegak2 = 4 * (np.sin(np.pi * arange(1.0, N + 0.5, 1.0) / 2 / N)) ** 2;   # Mode Frequencies

    # X0 = np.concatenate((np.exp(- 100 * np.arange(-0.16, 0.16, 0.1) ** 2), np.zeros(32))) # condição inicial (gaussiana)

    times = np.arange(0, 1e4, 20)
    r = odeint(fpu_flux, X0, times, rtol=1e-4)

    omegak2 = 4 * (np.sin(np.pi * np.arange(1.0, Nmodes + 0.5, 1.0) / 2.0 / N)) ** 2   # Mode Frequencies

    modes = np.zeros((Nmodes, N))
    for i in range(1, Nmodes+1):
        modes[i-1] = np.sqrt(2.0/N) * np.sin(np.pi * float(i) * np.arange(N) / N)

    r_modes = np.zeros((len(times), Nmodes))
    for i in range(len(times)):
        for j in range(1, Nmodes):
            r_modes[i][j-1] = (omegak2[j-1] ** 2 \
                    * np.dot(r[i][:32], modes[j-1]) ** 2 + np.dot(r[i][32:], modes[j-1]) ** 2) / 2.0 


    ## plotting

    #figure, ax = fig()
    #for i in range(1, Nmodes):
    #    ax.plot(times, r_modes[:,i-1], label='modo %s' % str(i))
    #save_fig(figure, 'modes1.png')

    #plot(r, 'fpu3.png')

