# -*- coding: utf-8 -*-
"""Integrates parabolic PDEs with fixed boundary conditions.

The equation 
    u_t = f(u) + D u_{xx}
with boundary conditions
    -> J_x|_{x=0} = left
    -> J_x|_{x=L} = right
"""
from numpy import *
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
        y = self.r * y0 * (1. - y0/self.K)

        y[1:-1] += self.D * (y0[2:] - 2*y0[1:-1] + y0[:-2])
        y[0] += self.D * (self.left -2*y0[0] + y0[1])
        y[-1] += self.D * (self.right -2*y0[-1] + y0[-2])
        return y


class PDE_fkpp_competitive(PDE):
    '''Fisher-Kolmogorov equations for 2 competing populations.'''
    def flux(self, y0, t):
        u = zeros(len(y0)/2)
        v = zeros(len(y0)/2)
        u0 = y0[:len(y0)/2]
        v0 = y0[len(y0)/2:]
        
        u = self.r * u0 * (1. - u0/self.K1 - self.c21 * v0)
        v = self.r * v0 * (1. - v0/self.K2 - self.c12 * u0)
        
        u[1:-1] += self.D1 * (u0[2:] - 2*u0[1:-1] + u0[:-2])
        u[0] += self.D1 * (self.left -2*u0[0] + u0[1])
        u[-1] += self.D1 * (self.right -2*u0[-1] + u0[-2])

        v[1:-1] += self.D2 * (v0[2:] - 2*v0[1:-1] + v0[:-2])
        v[0] += self.D2 * (self.left -2*v0[0] + v0[1])
        v[-1] += self.D2 * (self.right -2*v0[-1] + v0[-2])
        
        return concatenate((u, v))

    def initialize(self, func):
        u0 = fromfunction(lambda j: func((j+1)*self.dx), (self.grid_size-1,))
        v0 = fromfunction(lambda j: func((j+1)*self.dx), (self.grid_size-1,))
        self.y0 = concatenate((u0, v0))


def PDE_integrate(p, times, equation, grid_size):
    s = equation(**p)
    s.set_grid(grid_size)
    s.initialize(lambda x: 0.01*exp(-50*(x-s.L/2.)**2))
    s.integrate(times)
    return s


def animate(grid, data, labelx='x', labely='', labels=[]):
    from pylab import plot, show, legend, xlabel, ylabel, ion, draw, figure, ylim
    import time
    
    ion()
    
    tstart = time.time()               # for profiling
    nvars = shape(data)[1]//len(grid)
    ldata = [ data[:,i*len(grid):(i+1)*len(grid)] for i in range(nvars) ]
    lines = []
    for d in ldata:
        lines.append(plot(grid, d[0])[0])
    xlabel(labelx)
    ylabel(labely)
    if len(labels) == nvars:
        legend(labels)
    ymin = 0 if data.min() > 0 else floor(data.min())
    ylim((ymin, ceil(data.max())))
    for i in range(shape(data)[0]):
        for l in range(len(lines)):
            lines[l].set_ydata(ldata[l][i])  # update the data
        draw()                         # redraw the canvas
    
    print 'FPS:' , shape(data)[0]/(time.time()-tstart)



if __name__ == '__main__':
    p = {
        'r': 2.,        # r
        'L': 1.,        # L
        'K1': 1.,        # K
        'K2': 1.,
        'D1': 1.,
        'D2': 1.,
        'c12': 0.1,
        'c21': 1.3,
        'left': 0.,
        'right': 0.
        }
    times = arange(0, 50, 0.1)
    s = PDE_integrate(p, times, equation=PDE_fkpp_competitive, grid_size=200)

