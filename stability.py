# -*- coding: utf-8 -*-
import multiprocessing
from itertools import permutations
from sympy import *

pprint_use_unicode(False)

def backsubs(y, ynew):
    return [ (v[0], simplify(v[1].subs([ynew]))) for v in y ] + [ ynew ]

def solve_step(eqs, xx, step, y=[]):
    if step == len(eqs):
        y.sort(key=lambda x: x[0])
        if any([ v[1] < 0 for v in y ]):
            return []
        print y
        return [ y ]
    eq = eqs[step].subs(y).as_numer_denom()[0]
    try:
        ystep = solve(eq, xx[step])
        print step, ystep
    except:
        return []
    
    result = []
    for ynew in ystep:
        y_next = backsubs(y, (xx[step], ynew))
        # discard NaN and Infinity
        if any([ numbers.Infinity in map(type, yi[1].atoms()) or numbers.NaN in map(type, yi[1].atoms()) for yi in y_next ]):
            continue
        # discard negative solutions assuming all parameters positive
        if any([ ask(yi[1], Q.negative, reduce(And, map(lambda q: Assume(q, Q.positive), list(yi[1].atoms(Symbol))))) for yi in y_next if yi[1].atoms(Symbol) ]):
            continue
        r = solve_step(eqs, xx, step+1, y_next)
        if r:
            result += r
    return result

def fixed_points(eqs, x, assumps=[], positive=True, nroots=None, multi=None):
    # doing it the smart way
    eqs2 = [ eq.as_numer_denom()[0] for eq in eqs ]
    #eqs_denom = [ eq.as_numer_denom()[1] for eq in eqs ]
    #exclude = unique([ (v, solve(eq, v)) for v in x if v in eq.atoms(Symbol) for eq in eqs_denom ], idfun=tuple)
    
    results = []
    # do it
    if not multi:
        for xx in permutations(x):
            results += solve_step(eqs2, xx, 0)
            results = unique(results, idfun=lambda x: tuple(x))
            if nroots and len(results) >= nroots:
                break
    else:
        # multiprocessing!
        pool = multiprocessing.Pool(None)
        r = pool.map_async(multi, permutations(x), callback=results.append)
        r.wait() # Wait on the results
        results = unique(results, idfun=lambda x: tuple(x))

    # try to verify solution and positivity (or better, non-negativity)
    #assumps = reduce(And, assumps) if assumps else True
    #for r in results:
    #    if any([ simplify(eq.subs(r)) for eq in eqs ]):
    #        results.remove(r)
    #    if positive:
    #        for X in [ y[1] for y in r ]:
    #            if ask(X, Q.negative, assumps):
    #                results.remove(r)
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

#if __name__ == '__main__':
if True:
    a,b,c,p,q,K,ay,az,dy,dz,by,bz,d = symbols(['a','b','c','p','q','K','ay','az','dy','dz','by','bz','d'], real=True)
    x,y,z = symbols('xyz', real=True)
    
#    assumps = [ Assume(a, Q.positive),
#                Assume(b, Q.positive),
#                Assume(p, Q.positive),
#                Assume(q, Q.positive)
#                ]
    
    eqs = [ x * (1 - x - y - z/(c+x**2)),
            y * (-p + a*x),
            z * (-q + b*x/(c+x**2)) ]
    
    eqs_IGP1 = [ x * (p*K - p*x - ay*y - az*z),
                 y * (-dy + by*ay*x - c*z),
                 z * (-dz + bz*az*x + d*c*y) ]
    
    eqs_IGP2 = [ x * (p*K - p*x - ay*y/(1+x) - az*z/(1+x)),
                 y * (-dy + by*ay*x/(1+x) - c*z/(1+y)),
                 z * (-dz + bz*az*x/(1+x) + d*c*y/(1+y)) ]

    w, m, pb, pp, pc, pd, zb, lb, lp, eb, ep = symbols(['w', 'm', 'pb', 'pp', 'pc', 'pd', 'zb', 'lb', 'lp', 'eb', 'ep'], real=True)
    y = [ Symbol('y%s' % i, real=True) for i in range(6)]
    eqs_coinf = [ w - m*y[0] - y[0]*y[4] - y[0]*y[5],
                  lb*y[0]*y[4] - y[1] - eb*y[1]*y[5],
                  y[0]*y[5] - pb*y[2] - ep*y[2]*y[4],
                  eb*y[1]*y[5] + ep*y[2]*y[4] - pp*y[2]/lp - pd*y[5]/lp,
                  y[1] - zb*y[2]/lp - pc*y[4] + pd*zb*y[5]/lp
                  ]
#                  eb*y[1]*y[5] + ep*y[2]*y[4] - pp*y[3],
#                  y[1] + zb*y[3] - pc*y[4],
#                  y[2] + lp*y[3] - pd*y[5]  ]


    #f = full_stability(eqs, [x,y,z])
    
    eqs2 = [ eq.as_numer_denom()[0] for eq in eqs_IGP1 ]
    def f(variables):
        solve_step(eqs2, variables, 0)
    #print fixed_points(eqs_IGP1, [x,y,z], multi=f)

