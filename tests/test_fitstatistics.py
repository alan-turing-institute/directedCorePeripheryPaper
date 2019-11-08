import networkx as nx
import random as rd
## this will need to be fixed before this is a package
import sys
sys.path.append('../')

import fitStatistics as fs
import graphGeneration as gg


def test__getCounts1():
    G=nx.DiGraph()
    G.add_nodes_from(range(30))
    for i in range(15):
        for j in range(15,30):
            G.add_edge(i,j)
    res = fs.__getCounts__(G,[0,]*15+[1,]*15,2)
    assert(len(res)==2)
    assert(len(res[0])==2)
    assert(len(res[1])==2)
    assert(res[0][0]==0)
    assert(res[0][1]==225)
    assert(res[1][0]==0)
    assert(res[1][1]==0)
    res = fs.__getCounts__(G,[0,]*15+[1,]*15,4)
    assert(len(res)==4)
    for i in range(4):
        assert(len(res[i])==4)
    for i in range(4):
        for j in range(4):
            if (i,j)!=(0,1):
                assert(res[i][j]==0)
            else:
                assert(res[i][j]==225)
    for i in range(15):
        for j in range(15,30):
            G.add_edge(j,i)
    res = fs.__getCounts__(G,[0,]*15+[1,]*15,2)
    assert(len(res)==2)
    assert(len(res[0])==2)
    assert(len(res[1])==2)
    assert(res[0][0]==0)
    assert(res[0][1]==225)
    assert(res[1][0]==225)
    assert(res[1][1]==0)


def test__getCounts2():
    rd.seed(568)
    G=nx.DiGraph()
    G.add_nodes_from(range(50))
    groups=[i%4 for i in range(50)]
    current = [[0 for i in range(4)] for j in range(4)]
    for run in range(1000):
        i=rd.randint(0,49)
        j=rd.randint(0,49)
        while j in G[i]:
            i=rd.randint(0,49)
            j=rd.randint(0,49)
        G.add_edge(i,j)
        current[groups[i]][groups[j]]+=1
        res = fs.__getCounts__(G,groups,4)
        for i in range(4):
            for j in range(4):
                assert(current[i][j]==res[i][j])



def test_getConnectionProb():
    rd.seed(569)
    for run in range(20):
        G=nx.DiGraph()
        G.add_nodes_from(range(50))
        groups=[rd.randint(0,3) for i in range(50)]
        numerator = [[0 for i in range(4)] for j in range(4)]
        denom     = [[0 for i in range(4)] for j in range(4)]
        probs=[[rd.random() for i in range(4)] for j in range(4)]
        for i in range(50):
            for j in range(50):
                thres = probs[groups[i]][groups[j]]
                if rd.random()<thres:
                    numerator[groups[i]][groups[j]]+=1
                    G.add_edge(i,j)
                denom[groups[i]][groups[j]]+=1
        res = fs.getConnectionProb(G,groups)
        for i in range(4):
            for j in range(4):
                assert(abs(numerator[i][j]/denom[i][j]-res[i][j])<10**(-8))

def test_getUpsets1():
    rd.seed(570)
    for i in range(10):
        G=gg.syntheticModel1(4*(i+1),0)
        groups=sorted([i%4 for i in range(4*i+4)])
        result,offEdges=fs.getUpsets(G,groups)
        assert(offEdges==0)
        for i in range(4):
            for j in range(4):
                assert(result[i][j]==0)


def test_getUpsets2():
    rd.seed(571)
    for i in range(10):
        G=gg.syntheticModel1(4*(i+2),0)
        groups=sorted([i%4 for i in range(4*i+8)])
        G.add_edge(0,1)
        result,offEdges=fs.getUpsets(G,groups)
        assert(offEdges==1)
        assert(result[0][0]==1)
        for j in range(1,4):
            assert(result[0][j]==0)
        for i in range(1,4):
            for j in range(4):
                assert(result[i][j]==0)


def test_getUpsets3():
    rd.seed(572)
    for i in range(10):
        G=gg.syntheticModel1(4*(i+2),0)
        groups=sorted([i%4 for i in range(4*i+8)])
        G.add_edge(0,2*(i+2))
        result,offEdges=fs.getUpsets(G,groups)
        assert(offEdges==1)
        assert(result[0][2]==1)
        for j in [0,1,3]:
            assert(result[0][j]==0)
        for i in range(1,4):
            for j in range(4):
                assert(result[i][j]==0)

def test_getUpsets4():
    rd.seed(573)
    cs = [[0,1,0,0],[0,1,0,0],[0,1,1,1],[0,0,0,0]]
    for run in range(20):
        G=nx.DiGraph()
        G.add_nodes_from(range(50))
        groups=[rd.randint(0,3) for i in range(50)]
        connect = [[0 for i in range(4)] for j in range(4)]
        noconnect     = [[0 for i in range(4)] for j in range(4)]
        probs=[[rd.random() for i in range(4)] for j in range(4)]
        for i in range(50):
            for j in range(50):
                thres = probs[groups[i]][groups[j]]
                if rd.random()<thres:
                    connect[groups[i]][groups[j]]+=1
                    G.add_edge(i,j)
                else:
                    noconnect[groups[i]][groups[j]]+=1
        res,offEdges = fs.getUpsets(G,groups)
        for i in range(4):
            for j in range(4):
                if cs[i][j]:
                    assert(noconnect[i][j]==res[i][j])
                else:
                    assert(connect[i][j]==res[i][j])
