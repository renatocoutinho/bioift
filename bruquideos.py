# -*- coding: utf-8 -*-
"""Modela uma população de bruquídeos em tempo contínuo.

O modelo consiste de um sistema de equações diferenciais acopladas, não-autônomas:

    d_t y[0] = (a - m) * y[1] - b(t) * y[0] / n,
    d_t y[1] = K(t) * b(t) * y[0] / (y[0] + K(t)) - a * y[1]
"""
from scipy import *
from pylab import *
from scipy import integrate

def plot4(y, time, K, b):
    fig = figure(figsize=(6,9), dpi=100)
    ax = []
    for i in range(4):
        ax.append(fig.add_subplot(411 + i))
        ax[i].set_xticks(arange(start, end, 4))
        if i < 3:
            ax[i].set_xticklabels([])
    
    ax[0].plot(time, K(time))
    ax[0].set_ylabel(r'$K(t)$')
    ax[1].plot(time, b(time)/n)
    ax[1].set_ylabel(r'$\beta(t)/\nu$')
    ax[2].plot(time, y[:,0]/K(time))
    ax[2].set_ylabel(r'$b(t)/K$ (adultos)')
    ax[3].plot(time, y[:,1]/K(time))
    ax[3].set_ylabel(r'$v(t)/K$ (ovos)')
    #ax[4].plot(time, a * y[:,1] * exp(-y[:,1]/K(time)) / K(time))
    #ax[4].set_ylabel(r'$\alpha v e^{-v/K} / K$')
    
    ax[-1].set_xlabel('tempo (meses)')
    
    show()

#R1 = 5.0 # mínimo de p(t)
#R2 = 40.0 # escala de p(t)

p = lambda l, t, P, R1, R2: R1 + R2 * exp(-l) * l ** (2*(t % P))  / factorial(2*(t % P))
h = 0.45 # ajuste para continuidade em LN
LN = lambda t, l, s, P, R1, R2: R1 + R2 * exp(-log((t%P+h)/s)**2 / 2 / l**2) / (t%P+h) / s / l / sqrt(2*pi)

l = 3.0 # largura do período mais produtivo
P = 12.0 # duração do ano
tau = 3.0 # atraso da capacidade de suporte com relação ao início do pico de reprodução

#b = lambda t: p(l, t + tau, P, R1=10.0, R2=40)
#b = lambda t: LN(t+tau, l=0.5, s=3.0, P=12.0, R1=10.0, R2=100.0)
#b = vectorize(lambda t: 5.0)
b = vectorize(lambda t: 30 if K(t+tau) > 10 else 0)
#K = lambda t: p(l, t, P, R1=1.0, R2=100.0)
K = lambda t: LN(t, l=0.5, s=3.0, P=12.0, R1=1.0, R2=300.0)
#K = vectorize(lambda t: 5.0)

a = 0.7
n = 10.0
m = 0.01
deriv = lambda y, t: array([
        (a - m) * y[1] - b(t) * y[0] / n,
#        b(t) * y[0] * exp(- y[0] / K(t)) - a * y[1]
        K(t) * b(t) * y[0] / (y[0] + K(t)) - a * y[1]
        ])

# y[0] : adultos
# y[1] : ovos

start = 0.0
end = 100.0
numsteps = 1e4
time = linspace(start,end,numsteps)

y0 = array([10, 10])

y = integrate.odeint(deriv,y0,time)

plot(time, y[:,0]/K(time), label='adultos/K')
plot(time, y[:,1]/K(time), label='ovos/K')
plot(time, K(time), label='K')
plot(time, b(time), label='b')
xlabel('t (meses)')
legend()
show()

