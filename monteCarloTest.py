import networkx as nx
import scoring as sc
from collections import Counter

def erdos_renyi_ensemble(G):
    assert(type(G)==nx.DiGraph)
    p=nx.density(G)
    n = len(G)
    while True:
        Gnew = nx.erdos_renyi_graph(n,p,directed=True)
        yield Gnew


def directed_config_ensemble(G):
    assert(type(G)==nx.DiGraph)
    indeg  = [G.in_degree(x)  for x in sorted(G)]
    outdeg = [G.out_degree(x) for x in sorted(G)]
    while True:
        Gnew = nx.directed_configuration_model(indeg,outdeg,create_using=nx.DiGraph)
        # remove multiedges
        Gnew = nx.DiGraph(Gnew)
        yield Gnew


def test_statistic(G,groups):
    cs = [[0,1,0,0],[0,1,0,0],[0,1,1,1],[0,0,0,0]]
    inEdges = 0
    outEdges = 0
    for x in G:
        g1 = groups[x]
        for y in G[x]:
            g2 = groups[y]
            if cs[g1][g2]==1:
                inEdges  += 1
            else:
                outEdges += 1
    counts = Counter(groups)
    outDenom = 0
    inDenom = 0
    for i in range(4):
        c1 = counts[i]
        for j in range(4):
            c2 = counts[j]
            if cs[i][j]==1:
               inDenom  += c1*c2
            else:
               outDenom += c1*c2
    pin  = inEdges/inDenom
    pout = outEdges/outDenom
    return pin - pout



import sys
def L_test(Greal,ensembleMeth,meths,reps=500):
    graphSet = ensembleMeth(Greal)
    results = {}
    nullStore = {}
    for meth in meths:
        nulls = []
        print(meth + ' rep:',end='')
        m1 = meths[meth]['meth']
        for rep in range(reps):
            if rep%10==0:
                print(rep,end =',')
                sys.stdout.flush()
            G=next(graphSet)
            m2 = m1(G)
            m3 = test_statistic(G,m2)
            nulls.append(m3)
        part = meths[meth]['part']
        realStat = test_statistic(Greal,part)
        numer = 1 + sum(x>=realStat for x in nulls)
        denom = 1 + reps
        pval  = numer/denom
        results[meth] = pval
        nullStore[meth] = nulls
    return results,nullStore
