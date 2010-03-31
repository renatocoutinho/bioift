# -*- coding: utf-8 -*-
"""integra a equação de fisher-kolmogorov (fkpp) em uma dimensão

u_t = Du_{xx} + ru(1-u/K) em um dománio finito com condições do tipo Dirichlet
( no caso, a funcao incógnita é nula na borda) adaptado de
http://www.maths.uq.edu.au/courses/MATH3203/assignment1.pdf (Dion Weatherley) 
a condição inicial é um meio período e um seno.
"""
from numpy import *
from scipy.interpolate import KroghInterpolator
from pylab import plot, show, legend, xlabel, ylabel, ion, draw, figure, ylim

p = {
        'r': 2.,        # r
        't1': 0.01,       # t_1
        'mJ': 1.,       # \mu_J
        'mA': 0.001,    # \mu_A
        'DJ': 1.,       # D_J
        'DA': 1.,       # D_A
        'd': 1,         # d := L_2 - L_1
        'L1': 1.,       # L_1
        'K': 1.         # K
        }

class PDE_sapos():
    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        self.L2 = self.L1 + self.d

    def _D(self, x):
        if x > self.L1:
            return self.DA
        else:
            return self.DJ
    
    def _m(self, x):
        if x > self.L1:
            return self.mA
        else:
            return self.mJ

    def _past_value(self, t, dt, xi):
        if t <= 0:
            return self.u0[xi]
        else:
            # interpolate using Hemite polynomials

            # extreme points
            t0 = int(floor(t/dt))
            t1 = int(ceil(t/dt))

            # derivatives at those points
            deriv2_0 = (self.data[t0][xi+1] - 2*self.data[t0][xi] + self.data[t0][xi-1]) / self.dx**2
            dt0 = self.D[xi] * deriv2_0 - self.m[xi] * self.data[t0][xi]
            deriv2_1 = (self.data[t1][xi+1] - 2*self.data[t1][xi] + self.data[t1][xi-1]) / self.dx**2
            dt1 = self.D[xi] * deriv2_1 - self.m[xi] * self.data[t1][xi]
            
            # find interpolator
            kip = KroghInterpolator([t0, t0, t1, t1], [self.data[t0][xi], dt0, self.data[t1][xi], dt1])
            return kip(t)

    def set_grid(self, M):
        self.M = M
        self.dx = self.L2 / (M+1)
        self.grid = arange(0, self.L2, self.dx)
        self.D = vectorize(self._D)(self.grid)
        self.m = vectorize(self._m)(self.grid)
        # index of the discontinuity
        self.jump = int(self.L1 / self.dx)

    def initialize(self, func):
        self.u0 = fromfunction(lambda j: func(j*self.dx), (self.M+1,))

    def _evolve(self, t, dt):
        U = self.data[-1]
        U_new = zeros(len(U))
        
        # central difference space derivative discretization
        deriv2 = (U[2:] - 2*U[1:-1] + U[:-2]) / self.dx**2

        # first-order euler time discretization
        U_new[1:-1] = U[1:-1] + dt * (self.D[1:-1] * deriv2 - self.m[1:-1] * U[1:-1])

        # boundary conditions
        #U_new[0] = 1. # constant
        #U_new[0] = self.r * U_new[self.jump] / (self.K + self.r*U_new[self.jump]) # no delay
        A1 = self._past_value(t - self.t1, dt, self.jump)
        U_new[0] = self.r * A1 / (self.K + A1) # full
        U_new[-1] = U_new[-2]

        return U_new

    def integrate(self, T, dt=0, mu=0.4, view=False):
        self.data = [ self.u0 ]
        if not dt:
            dt = mu * self.dx**2
        # ensure stability
        assert dt/self.dx**2 <= 0.5
        steps = int(T/dt)

        if view:
            ion()
            #fig = figure(figsize=(12,8), dpi=100)
            #ax = fig.add_axes()
            line, = plot(self.grid, self.data[-1])
            ylim(0, 2.)

        for t in arange(dt, T, dt):
            self.data.append(self._evolve(t, dt))
            if view:
                line.set_ydata(self.data[-1])
                draw()


def PDE_sapos_integrate(p, T, mesh_size=100, view=False):
    s = PDE_sapos(**p)
    s.set_grid(100)
    s.initialize(lambda x: 0.01*exp(-x))
    dt = 0.4 * s.dx**2
    s.integrate(T, dt, view=view)
    return s

if __name__ == '__main__':
    s = PDE_sapos_integrate(p, 0.5, 100, view=True)
