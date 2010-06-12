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

class PDE(object):
    '''Diffusion equation.'''
    def __init__(self, **kwargs):
        for (key, value) in kwargs.items():
            setattr(self, key, value)

    def set_grid(self, grid_size):
        self.grid_size = grid_size
        self.dx = self.L / (grid_size+1)
        self.grid = arange(0, self.L, self.dx)
        self.grid = self.grid[1:-1]

    def DiffusiveTerm_1D(self, u):
        y = -2 * u
        y[1:-1] = u[2:] + u[:-2]
        y[0] = self.left + u[1]
        y[-1] = self.right + u[-2]
        return y/self.dx/self.dx
   
    def DiffusiveTerm_2D(self, u):
        u2 = reshape(u, (self.grid_size-1, self.grid_size-1))
        y = -4*u2
        y[1:-1,:] += u2[0:-2,:] + u2[2:,:]
        y[:,1:-1] += u2[:,0:-2] + u2[:,2:]
        y[0,:] += u2[1,:] + self.top
        y[-1,:] += u2[-2,:] + self.bottom
        y[:,0] += u2[:,1] + self.left
        y[:,-1] += u2[:,-2] + self.right
        return reshape(y, len(u))/self.dx/self.dx

    def integrate(self, t):
        self.data = odeint(self.flux, self.y0, t)

    ## these methods/variables are specific to each problem
    # number of spatial dimensions
    dim = 1

    def flux(self, y0, t):
        y = self.D * self.DiffusiveTerm_1D(y0)
        return y

    def initialize(self, func):
        self.y0 = fromfunction(func, self.dim*(self.grid_size-1,)).flatten()


class PDE_fkpp(PDE):
    '''Fisher-Kolmogorov equation.'''
    def flux(self, y0, t):
        y = self.r * y0 * (1. - y0/self.K) + self.D * self.DiffusiveTerm_1D(y0)
        return y


class PDE_fkpp_2D(PDE):
    '''Fisher-Kolmogorov equation.'''
    dim = 2
    def flux(self, y0, t):
        y = self.r * y0 * (1. - y0/self.K) + self.D * self.DiffusiveTerm_2D(y0)
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
    from pylab import plot, legend, xlabel, ylabel, ion, draw, ylim
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


def animate_2D(grid, data, skip_frames=1, autoscale=False, labelx='x', labely='y', labels=[]):
    from pylab import pcolor, colorbar, ion, xlabel, ylabel, draw
    import time

    ion()

    tstart = time.time()               # for profiling
    X,Y = meshgrid(grid, grid)
    data = reshape(data, (data.shape[0], len(grid), len(grid)))
    p = pcolor(X, Y, data[0], vmin=data.min(), vmax=data.max())
    xlabel(labelx)
    ylabel(labely)
    colorbar()
    for i in range(1, data.shape[0]):
        p.set_array(data[i,0:-1,0:-1].ravel())
        # rescale colors on each iteration. Useful if scale changes too much
        if autoscale:
            p.autoscale()
        draw()

    print 'FPS:' , shape(data)[0]/(time.time()-tstart)

def PDE_integrate(p, times, equation, grid_size):
    s = equation(**p)
    s.set_grid(grid_size)
    gaussiano = lambda i: 0.1*exp(-500*(i*s.dx-s.L/2.)**2)
    quadrado = vectorize(lambda i: 0.1 if abs(i*s.dx-s.L/2) < 0.2 else 0.)
    quadrado2d = vectorize(lambda i, j: 0.1 if abs(i*s.dx-s.L/2) < 0.2 and abs(j*s.dx-s.L/2) < 0.2 else 0.)
    s.initialize(quadrado2d)
    s.integrate(times)
    return s


if __name__ == '__main__':
    p = {
        'r': 50.,
        'L': 1.,
        'K': 1.,
        'D': 1.,
        'K1': 1.,
        'K2': 1.,
        'D1': 1.,
        'D2': 1.,
        'c12': 0.1,
        'c21': 1.3,
        'left': 0.,
        'right': 0.,
        'top': 0.,
        'bottom': 0.
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
    times = arange(0, 0.2, 0.005)
    #s = PDE_integrate(p, times, equation=PDE_fkpp_competitive, grid_size=200)
    #s = PDE_integrate(p_pp, times, equation=PDE_polyphenic, grid_size=100)
    #animate(s.grid, s.data, labels=['N', 'B', 'M'])

    s = PDE_integrate(p , times, equation=PDE_fkpp_2D, grid_size=50)
    #animate_2D(s.grid, s.data)

