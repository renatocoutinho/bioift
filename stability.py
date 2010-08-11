# -*- coding: utf-8 -*-
from itertools import permutations
from sympy import *

def fixed_points(eqs, x, assumps=[], positive=True):
    # doing it the smart way
    results = []
    def backsubs(y, ynew):
        return [ (v[0], simplify(v[1].subs([ynew]))) for v in y ] + [ ynew ]

    def solve_step(eqs, xx, step, y=[]):
        if step == len(eqs):
            y.sort(key=lambda x: x[0])
            results.append(y)
            return
        eq = eqs[step].subs(y)
        try:
            ystep = solve(eq, xx[step])
        except:
            return
        for ynew in ystep:
            solve_step(eqs, xx, step+1, backsubs(y, (xx[step], ynew)))
    
    # do it
    for xx in permutations(x):
        solve_step(eqs, xx, 0)

    results = unique(results, idfun=lambda x: tuple(x))

    # try to verify solution and positivity (or better, non-negativity)
    assumps = reduce(And, assumps) if assumps else True
    for r in results:
        if any([ simplify(eq.subs(r)) for eq in eqs ]):
            results.remove(r)
        if positive:
            for X in [ y[1] for y in r ]:
                if ask(X, Q.negative, assumps):
                    results.remove(r)
    return results

def unique(seq, idfun=None):
    '''Returns list without repeats.

    Source: http://www.peterbe.com/plog/uniqifiers-benchmark
    '''
    # order preserving
    if idfun is None:
        def idfun(x): return x
    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker in seen: continue
        seen[marker] = 1
        result.append(item)
    return result

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
    a,b,c,p,q = symbols('abcpq', real=True)
    x,y,z = symbols('xyz', real=True)
    
#    assumps = [ Assume(a, Q.positive),
#                Assume(b, Q.positive),
#                Assume(p, Q.positive),
#                Assume(q, Q.positive)
#                ]
    
    eqs = [ x * (1 - x - y - z/(c+x**2)),
            y * (-p + a*x),
            z * (-q + b*x/(c+x**2)) ]
    
    #f = full_stability(eqs, [x,y,z])

