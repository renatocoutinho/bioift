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
    def __init__(self, dim=1, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)
        self.dim = dim

    def set_grid(self, grid_size):
        self.grid_size = grid_size
        self.dx = self.L / (grid_size+1)
        if self.dim == 1:
            self.grid = arange(0, self.L, self.dx)
            self.grid = self.grid[1:-1]

    def initialize(self, func):
        self.y0 = fromfunction(lambda j: func((j+1)*self.dx), (self.grid_size-1,))

    def DiffusiveTerm_1D(self, u):
        y = zeros(len(u))
        y[1:-1] = u[2:] - 2*u[1:-1] + u[:-2]
        y[0] = self.left - 2*u[0] + u[1]
        y[-1] = self.right - 2*u[-1] + u[-2]
        return y/self.dx/self.dx
   
    def DiffusiveTerm_2D(self, u):
        Nx, Ny = self.grid.shape
        u2 = reshape(u, (Nx, Ny))
        y = -4*u2
        y[1:-1,:] += u2[0:-2,:] + u2[2:,:]
        y[:,1:-1] += u2[:,0:-2] + u2[:,2:]
        y[0,:] += u2[1,:] + self.top
        y[-1,:] += u2[-2,:] + self.bottom
        y[:,0] += u2[:,1] + self.left
        y[:,-1] += u2[:,-2] + self.right
        return reshape(y, len(u))

    def flux(self, y0, t):
        y = self.D * self.DiffusiveTerm_1D(y0)
        return y

    def integrate(self, t):
        self.data = odeint(self.flux, self.y0, t)


class PDE_fkpp(PDE):
    '''Fisher-Kolmogorov equation.'''
    def flux(self, y0, t):
        y = self.r * y0 * (1. - y0/self.K) + self.D * self.DiffusiveTerm_1D(y0)
        return y


class PDE_fkpp_competitive(PDE):
    '''Fisher-Kolmogorov equations for 2 competing populations.'''
    def flux(self, y0, t):
        u0 = y0[:len(y0)/2]
        v0 = y0[len(y0)/2:]
        u = self.r * u0 * (1. - u0/self.K1 - self.c21 * v0) + self.D1 * self.DiffusiveTerm_1D(u0)
        v = self.r * v0 * (1. - v0/self.K2 - self.c12 * u0) + self.D2 * self.DiffusiveTerm_1D(v0)
        return concatenate((u, v))

    def initialize(self, func):
        u0 = fromfunction(lambda j: func((j+1)*self.dx), (self.grid_size-1,))
        v0 = fromfunction(lambda j: func((j+1)*self.dx), (self.grid_size-1,))
        self.y0 = concatenate((u0, v0))


class PDE_polyphenic(PDE):
    '''Fisher-Kolmogorov equations for 2 competing populations.'''
    def flux(self, y0, t):
        N0 = y0[:len(y0)/3]
        B0 = y0[len(y0)/3:-len(y0)/3]
        M0 = y0[-len(y0)/3:]
        
        N = self.r * (B0 + self.g*M0) * (1. - exp(-(B0 + self.g*M0)/self.NC)) \
            - self.s * N0 + self.DN * self.DiffusiveTerm_1D(N0)
        B = -self.mB * B0 + self.s / (1. + N0/self.K) * N0 \
            * (1. - self.b0 * exp(N0-self.NL)/(1.+exp(N0-self.NL))) \
            + self.DB * self.DiffusiveTerm_1D(B0)
        M = -self.mM * M0 + self.s / (1. + N0/self.K) * N0 * self.b0 \
            * exp(N0-self.NL)/(1.+exp(N0-self.NL)) + self.DM * self.DiffusiveTerm_1D(M0)
        
        return concatenate((N, B, M))

    def initialize(self, func):
        N0 = fromfunction(lambda j: func((j+1)*self.dx), (self.grid_size-1,))
        B0 = fromfunction(lambda j: func((j+1)*self.dx), (self.grid_size-1,))
        M0 = fromfunction(lambda j: func((j+1)*self.dx), (self.grid_size-1,))
        self.y0 = concatenate((N0, B0, M0))


def animate(grid, data, skip_frames=1, labelx='x', labely='', labels=[]):
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
    for i in range(shape(data)[0]//skip_frames):
        for l in range(len(lines)):
            lines[l].set_ydata(ldata[l][i*skip_frames])  # update the data
        draw()                         # redraw the canvas
    
    print 'FPS:' , shape(data)[0]/(time.time()-tstart)


def PDE_integrate(p, times, equation, grid_size):
    s = equation(**p)
    s.set_grid(grid_size)
    gaussiano = lambda x: 0.1*exp(-500*(x-s.L/2.)**2)
    quadrado = vectorize(lambda x: 0.1 if abs(x-s.L/2) < 0.2 else 0.)
    s.initialize(quadrado)
    s.integrate(times)
    return s


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

    p_pp = {
        'r': 2.,
        'L': 10.,
        'K': 1.,
        'DN': 0,
        'DB': 0,
        'DM': 0.1,
        's': 0.7,
        'g': 0.5,
        'b0': 0.4,
        'NL': 10,
        'NC': 0.2,
        'mB': 0.1,
        'mM': 0.1,
        'left': 0.,
        'right': 0.
        }
    times = arange(0, 100, 1)
    #s = PDE_integrate(p, times, equation=PDE_fkpp_competitive, grid_size=200)
    s = PDE_integrate(p_pp, times, equation=PDE_polyphenic, grid_size=100)
    animate(s.grid, s.data, labels=['N', 'B', 'M'])

