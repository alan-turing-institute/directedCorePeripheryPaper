import pytest
import numpy as np
import networkx as nx
import random as rd
## this will need to be fixed before this is a package
import sys
sys.path.append('../')

import graphGeneration as gg


#### Test without the graph

def basicNoGraphTest():
    np.random.seed(0)
    n=15
    v1=np.random([n,1])
    v1/=(v1*v1).sum()
    u1=np.random([n,1])
    u1/=(u1*u1).sum()
    v2=np.random([n,1])
    v2/=(v1*v1).sum()
    u2=np.random([n,1])
    u2/=(u1*u1).sum()
    v3=np.random([n,1])
    v3/=(v1*v1).sum()
    u3=np.random([n,1])
    u3/=(u1*u1).sum()
    v4/=(v1*v1).sum()
    u4=np.random([n,1])
    u4/=(u1*u1).sum()
    for d1 in [1,5,10]:
        for d2 in [1,5,10]:
            for d3 in [1,5,10]:
                for d4 in [1,5,10]:
                    A=0
                    ordering=[]
                    ordering.append([d1,np.outer(u1,v1)])
                    ordering.append([d2,np.outer(u2,v2)])
                    ordering.append([d3,np.outer(u3,v3)])
                    ordering.append([d4,np.outer(u4,v4)])
                    ordering.sort(lambda x:-x)
                    A+=d1*np.outer(u1,v1)
                    A+=d2*np.outer(u2,v2)
                    A+=d3*np.outer(u3,v3)
                    A+=d4*np.outer(u4,v4)
                    for i in range(1,4):
                        A1=__lowRankHelper__(A,i)
                        A2=0
                        for j in range(i):
                            o1=ordering[j]
                            A2+=o1[0]*np.outer(o1[1][0],o1[1][1])
                        assert(np.abs((A1-A2).max()<10**(-10)))



#### Test with the graph
def getGroup(G,i):
    return G.name[i]

def test_model1P0():
    rd.seed(2001)
    correctStructure=[]
    correctStructure.append([0,1,0,0])
    correctStructure.append([0,1,0,0])
    correctStructure.append([0,1,1,1])
    correctStructure.append([0,0,0,0])
    for shuf in [0,1]:
        for n in range(4,100,4):
            G=gg.syntheticModel1(n,0,shuf)
            for x in range(n):
                g1=getGroup(G,x)
                for y in range(x):
                    g2=getGroup(G,y)
                    if correctStructure[g1][g2]==1:
                        assert(G.has_edge(x,y))
                    else:
                        assert(not G.has_edge(x,y))

def test_model1P1():
    rd.seed(2002)
    for shuf in [0,1]:
        for n in range(4,100,4):
            with pytest.raises(gg.IncorrectParameters):
                G=gg.syntheticModel1(n,1,shuf)

def test_model2P1_1_P2_0():
    rd.seed(2003)
    correctStructure=[]
    correctStructure.append([0,1,0,0])
    correctStructure.append([0,1,0,0])
    correctStructure.append([0,1,1,1])
    correctStructure.append([0,0,0,0])
    for shuf in [0,1]:
        for n in range(4,100,4):
            G=gg.syntheticModel2(n,1,0,shuf)
            for x in range(n):
                g1=getGroup(G,x)
                for y in range(x):
                    g2=getGroup(G,y)
                    if correctStructure[g1][g2]==1:
                        assert(G.has_edge(x,y))
                    else:
                        assert(not G.has_edge(x,y))


def test_graphReturnsCorrectSize():
    rd.seed(2004)
    rd.seed(100)
    for shuf in [0,1]:
        for i in range(1,51):
            G=gg.syntheticModel2(i*4,1,0,shuf)
            assert(len(G)==i*4)
            G=gg.syntheticModel2(i*4,0.0000001,0)
            assert(len(G)==i*4)

def test_model2P1_0_P2_1():
    rd.seed(2004)
    for shuf in [0,1]:
        for n in range(4,100,4):
            with pytest.raises(gg.IncorrectParameters):
                G=gg.syntheticModel2(n,0,1,shuf)


def test_model2P1_0_P2_1Helper():
    rd.seed(2005)
    correctStructure=[]
    correctStructure.append([0,1,0,0])
    correctStructure.append([0,1,0,0])
    correctStructure.append([0,1,1,1])
    correctStructure.append([0,0,0,0])
    for shuf in [0,1]:
        for n in range(4,100,4):
            G=gg.__syntheticModel2Helper__([int(n/4) for i in range(4)],0,1,shuf)
            for x in range(n):
                g1=getGroup(G,x)
                for y in range(x):
                    g2=getGroup(G,y)
                    if correctStructure[g1][g2]==0:
                        assert(G.has_edge(x,y))
                    else:
                        assert(not G.has_edge(x,y))


def test_model1BadSizes():
    rd.seed(2006)
    for n in range(5,100):
        if n%4==0:
            continue
        with pytest.raises(gg.IncorrectParameters):
            G=gg.syntheticModel1(n,1)

def test_model2BadSizes():
    rd.seed(2007)
    for n in range(5,100):
        if n%4==0:
            continue
        with pytest.raises(gg.IncorrectParameters):
            G=gg.syntheticModel2(n,1,0)
