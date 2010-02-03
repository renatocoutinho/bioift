# -*- coding: utf-8 -*-
'''Some useful functions to calculate the period of a data series.'''
from numpy import array, pi, arange, loadtxt, fft, where, int_, mean

def get_peaks(r, tol=1e-5):
    # constant solution
    if r.max() - r.mean() < tol:
        return array([])
    
    # achar periodo olhando a distancia media entre picos
    picos = array([ [i,p] for i, p in enumerate(r) if i>0 and r[i-1] < p and i+1<r.size and r[i+1] < p ])
    return picos

def get_period_peaks(r, dt, tol=1e-5):
    '''Calculates the period of a data series using the poor method of averaging distance between peaks.'''
    picos = get_peaks(r, tol=tol)
    if picos.size == 0 or len(picos[:,0]) <= 1:
        # not enough points
        return -1
    else:
        ds = picos[:,0][1:] - picos[:,0][:-1]
        Tpicos = mean(ds) * dt
        return Tpicos

def get_period_fft(r, dt, tol=1e-5):
    '''Calculates period using fft method.'''
    # constant solution
    if r.max() - r.mean() < tol:
        return 0

    modf = abs(fft.rfft(r))

    # exclude constant part
    modf = modf[2:]
    imax = 2 + modf.argmax()

    T = r.size*dt / imax
    return T

def get_max_frequencies(r, dt, factor=1e-5):
    '''Calculates the maxima in the spectrum of frequencies of a data series.'''
    modf = abs(fft.rfft(r))
    rlen = r.size
    freqs = 2*pi*arange(0, modf.size, 1.0) / (rlen*dt)

    # calculate maxima
    maxima = array([ [i,x] for i, x in enumerate(modf) if i>0 and modf[i-1] < x and i+1<modf.size and modf[i+1] < x ])
    # filter out maxima of lower amplitude
    pmax = modf[1:].max()
    nmax = maxima[where(maxima[1] > pmax*factor)]

    return array(zip(freqs[int_(nmax[:,0])], nmax[:,1]))


