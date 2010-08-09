# -*- coding: utf-8 -*-
from sympy import *

def fixed_points(eqs, x, assumps=[], positive=True):
    # doing it the smart way
    results = []
    def backsubs(y, ynew):
        return [ (v[0], simplify(v[1].subs([ynew]))) for v in y ] + [ ynew ]

    def solve_step(eqs, step, y=[]):
        if step == len(eqs):
            results.append(y)
            return
        eq = eqs[step].subs(y)
        ystep = solve(eq, x[step])
        for ynew in ystep:
            solve_step(eqs, step+1, backsubs(y, (x[step], ynew)))
    
    # do it
    solve_step(eqs, 0)

    # try to verify positivity (or better, non-negativity)
    if positive:
        assumps = reduce(And, assumps) if assumps else True
        for r in results:
            for X in [ y[1] for y in r ]:
                if ask(X, Q.negative, assumps):
                    results.remove(r)
    return results

def jacob(eqs, x):
    return Matrix([ [ eq.diff(y) for y in x] for eq in eqs ])

def coeff_to_Newton_sum(a):
    n = len(a)
    m = n*(n-1)//2
    a = a + (m-n)*[0]
    S = [n, -a[0]]
    for k in range(2, n+1):
        S.append(-k * a[k-1] - reduce(Add, [ S[k-i] * a[i-1] for i in range(1, k) ]))
    for k in range(n+1, m+1):
        S.append(-reduce(Add, [ S[k-i] * a[i-1] for i in range(1, k) ]))
    return S

def change_Newton_sum(S):
    m = len(S)
    return [ (reduce(Add, [ S[i] * S[l-i] * Binomial(l, i) for i in range(0, l+1) ]) - 2**l*S[l])/2 for l in range(1, m) ]

def Newton_sum_to_coeff(T):
    a = [ -T[0] ]
    for l in range(1, len(T)):
        a.append( - simplify((T[l] + reduce(Add, [ T[l-1-i]*a[i] for i in range(0, l) ])) / (l+1)) )
    return a

def Strelitz_stability(poly, s):
    '''Returns assumptions that need to be satisfied in order for the polynomial to be stable.

    Source: Sh. Strelitz, The American Mathematical Monthly, Vol. 84, No. 7 
    (Aug. - Sep., 1977), pp. 542-544

    See Also: Raymond Séroul, Programming for mathematicians, Springer (2000), sec. 10.13
    '''
    p = list(Poly(poly, s).monic().as_dict().iteritems())
    p.sort(key=lambda x: x[0], reverse=True)
    a = [ x[1] for x in p[1:] ]
    
    # substituting for efficiency
    C = numbered_symbols('c')
    clist = [ C.next() for x in range(len(a)) ]

    b = Newton_sum_to_coeff(change_Newton_sum(coeff_to_Newton_sum(clist)))
    b = [ simplify(x.subs(zip(clist, a))) for x in b ]
    #return [ Assume(x, Q.positive) for x in a + b ]
    return a+b

def full_stability(eqs, x, assumps=[]):
    fp = fixed_points(eqs, x, assumps)
    J = jacob(eqs, x)
    result = []
    for p in fp:
        try:
            e = J.subs(p).eigenvals()
        except:
            s = Symbol('s')
            det = (J.subs(p) - s*eye(J.shape[0])).det()
            result.append([p, Strelitz_stability(det, s)])
        else:
            result.append([p, e.keys()])
    return result

if __name__ == '__main__':
    a,b,p,q = symbols('abpq', real=True)
    x,y,z = symbols('xyz', real=True)
    
#    assumps = [ Assume(a, Q.positive),
#                Assume(b, Q.positive),
#                Assume(p, Q.positive),
#                Assume(q, Q.positive)
#                ]
    
    eqs = [ x * (1 - x - y - z),
            y * (-p + a*x),
            z * (-q + b*x) ]
    
    f = full_stability(eqs, [x,y,z])

