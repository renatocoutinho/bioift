# -*- coding: utf-8 -*-
"""Integrates a parabolic PDE with non-homogeneous coefficients and delayed boundary conditions.

The equation 
    u_t = D(x) u_{xx} - \mu(x) u
with boundary conditions
    -> J_x|_{x=L_2} = 0
"""
from numpy import *
from pylab import plot, show, legend, xlabel, ylabel, ion, draw, figure, ylim
from scipy.integrate import odeint

class PDE():
    '''Diffusion equation.'''
    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)

    def set_grid(self, grid_size):
        self.grid_size = grid_size
        self.dx = self.L / (grid_size+1)
        self.grid = arange(0, self.L, self.dx)
        self.grid = self.grid[1:-1]

    def initialize(self, func):
        self.y0 = fromfunction(lambda j: func((j+1)*self.dx), (self.grid_size-1,))

    def flux(self, y0, t):
        y = zeros(len(y0))
        y[1:-1] = y0[2:] - 2*y0[1:-1] + y0[:-2]
        y[0] = self.left -2*y0[0] + y0[1]
        y[-1] = self.right -2*y0[-1] + y0[-2]
        return self.D * y

    def integrate(self, t):
        self.data = odeint(self.flux, self.y0, t)

class PDE_fkpp(PDE):
    '''Fisher-Kolmogorov equation.'''
    def flux(self, y0, t):
        y = zeros(len(y0))
        y[1:-1] = self.r * y0[1:-1] * (1. - y0[1:-1]/self.K) + self.D * (y0[2:] - 2*y0[1:-1] + y0[:-2])
        y[0] = self.r * y0[0] * (1. - y0[0]/self.K) + self.D * (self.left -2*y0[0] + y0[1])
        y[-1] = self.r * y0[-1] * (1. - y0[-1]/self.K) + self.D * (self.right -2*y0[-1] + y0[-2])
        return y


def PDE_integrate(p, times, equation=PDE_fkpp, grid_size=100):
    s = equation(**p)
    s.set_grid(grid_size)
    s.initialize(lambda x: 0.01*exp(-50*(x-s.L/2.)**2))
    s.integrate(times)
    return s

if __name__ == '__main__':
    p = {
        'r': 2.,        # r
        'L': 1.,        # L
        'K': 1.,        # K
        'D': 10.,
        'left': 0.,
        'right': 0.
        }
    times = arange(0, 200, 1.)
    s = PDE_integrate(p, times, equation=PDE_fkpp, grid_size=100)
