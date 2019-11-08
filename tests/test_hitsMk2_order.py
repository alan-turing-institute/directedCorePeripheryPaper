import random as rd
import numpy as np
import networkx as nx
import sys
from copy import deepcopy
sys.path.append('../')
import hitsMk2 as hm
import hitsMk2NonVect as hmV
import pytest

### silly initial Tests

def test_getNorm_memory():
    coms = np.random.random([1000,4])
    c1 = hm.__getNorm__(coms)
    assert(id(c1)!=id(coms))




def slowMethod(G, p_in, p_out, c_in, c_out, order, combined=True):
    A, mA, At, mAt = hmV.__makeMatrices__(G)
    q1 = A.sum()
    PoutNew = hmV.updatePout_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
    CoutNew = hmV.updateCout_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
    PinNew = hmV.updatePin_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
    CinNew = hmV.updateCin_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
    if order==0:
        p_out = PoutNew
    if order==1:
        c_in = CinNew
    if order==2:
        c_out = CoutNew
    if order==3:
        p_in = PinNew
#    p_out, c_in, c_out, p_in = hmV.__getNormNonVect__(p_out, c_in, c_out, p_in)
    if combined:
        combined = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
        return combined
    else:
        return p_in, p_out, c_in, c_out


def __compareVectToNonVector__(G, p_in1, p_out1, c_in1, c_out1):
    __compareVectToNonVectorOneChange__(G, p_in1, p_out1, c_in1, c_out1)
    __compareVectToNonVectorAllChanges__(G, p_in1, p_out1, c_in1, c_out1)
    r1 = np.array([rd.random()*10**(-15) for x in range(len(p_in1))])
    r2 = np.array([rd.random()*10**(-15) for x in range(len(p_in1))])
    r3 = np.array([rd.random()*10**(-15) for x in range(len(p_in1))])
    r4 = np.array([rd.random()*10**(-15) for x in range(len(p_in1))])
    __compareVectToNonVectorOneChange__(G, p_in1+r1, p_out1+r2, c_in1+r3, c_out1+r4)

    r1 = np.array([rd.random()*10**(-15) for x in range(len(p_in1))])
    r2 = np.array([rd.random()*10**(-15) for x in range(len(p_in1))])
    r3 = np.array([rd.random()*10**(-15) for x in range(len(p_in1))])
    r4 = np.array([rd.random()*10**(-15) for x in range(len(p_in1))])
    __compareVectToNonVectorAllChanges__(G, p_in1+r1, p_out1+r2, c_in1+r3, c_out1+r4)


def __compareVectToNonVectorOneChange__(G, p_in1, p_out1, c_in1, c_out1):
    func1 = hmV.__getNormNonVect__
    func2 = hm.__hitsmk2Order__
# func2 = hm.__hitsmk2Order__(A, At, C, Ct, oldVer, s1, C2, order):
    for order in range(4):
        p_in = np.array(deepcopy(p_in1)).reshape([len(G), 1])
        p_out = np.array(deepcopy(p_out1)).reshape([len(G), 1])
        c_in = np.array(deepcopy(c_in1)).reshape([len(G), 1])
        c_out = np.array(deepcopy(c_out1)).reshape([len(G), 1])
        p_out, c_in, c_out, p_in = func1(p_out, c_in, c_out, p_in)
        # make copy here just in case of modification
        oldVer = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
        slowRes = slowMethod(G, p_in, p_out, c_in, c_out, order)
        A, At, C, Ct, n, s1, s2 = hm.__setup__(G)
        C2 = C + Ct
        newVer = func2(A, At, C, Ct, oldVer, s1, C2, order)
        assert(abs(newVer[:, order]-slowRes[:, order]).max() < 10**(-8))
#        assert(abs(oldVer.sum(axis=0)-totals).max() < 10**(-10))

def __compareVectToNonVectorAllChanges__(G, p_in1, p_out1, c_in1, c_out1):
    p_in = np.array(deepcopy(p_in1)).reshape([len(G), 1])
    p_out = np.array(deepcopy(p_out1)).reshape([len(G), 1])
    c_in = np.array(deepcopy(c_in1)).reshape([len(G), 1])
    c_out = np.array(deepcopy(c_out1)).reshape([len(G), 1])
    p_out, c_in, c_out, p_in = hmV.__getNormNonVect__(p_out, c_in, c_out, p_in)
    oldVer = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
    order = list(range(4))
    rd.shuffle(order)
    A, At, C, Ct, n, s1, s2 = hm.__setup__(G)
    As, Ats, totals = hm.__setupOneAtATime__(A, At, oldVer)
    slwM = slowMethod
    func1 = hm.__hitsmk2Order__
    C2 = C + Ct
    for t1 in order:
        # make copy here just in case of modification
        oldVer = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
        p_in, p_out, c_in, c_out = slwM(G, p_in, p_out, c_in, c_out, t1, False)
        oldVer = func1(A, At, C, Ct, oldVer, s1, C2, t1)
        slowRes = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
        assert(abs(oldVer[:,t1]-slowRes[:,t1]).max() < 10**(-8))
        takeNorm = hmV.__getNormNonVect__
        p_out, c_in, c_out, p_in = takeNorm(p_out, c_in, c_out, p_in)
    ## This test is a little redundant but there you go
    oldVer = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
    p_in2 = np.array(deepcopy(p_in1)).reshape([len(G), 1])
    p_out2 = np.array(deepcopy(p_out1)).reshape([len(G), 1])
    c_in2 = np.array(deepcopy(c_in1)).reshape([len(G), 1])
    c_out2 = np.array(deepcopy(c_out1)).reshape([len(G), 1])
    gtNormNoVec = hmV.__getNormNonVect__
    p_out2, c_in2, c_out2, p_in2 = gtNormNoVec(p_out2, c_in2, c_out2, p_in2)

    original = np.concatenate([p_out2, c_in2, c_out2, p_in2], axis=1)
    actualMaxDiff = abs(original-oldVer).max()

    p_outM = abs(p_out2-p_out).max()
    c_outM = abs(c_out2-c_out).max()
    p_inM = abs(p_in2-p_in).max()
    c_inM = abs(c_in2-c_in).max()
    slowMax = max([p_outM, c_outM, p_inM, c_inM])

    assert(abs(actualMaxDiff-slowMax) < 10**(-10))


def test_TwoNodeOneEdgeOnlyOneCoutVectSelfLoop():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([1, ]+[0 for x in range(1)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyOnePinVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1, ]+[0 for x in range(19)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyOneCinVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([1, ]+[0 for x in range(19)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyOnePoutVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([1, ]+[0 for x in range(19)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyOneCoutVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([1, ]+[0 for x in range(19)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOneCoutVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([1, ]+[0 for x in range(1)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOnePinVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        p_in = np.array([1, ]+[0 for x in range(1)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOneCinVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([1, ]+[0 for x in range(1)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOnePoutVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([1, ]+[0 for x in range(1)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyPinVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1, ]+[1 for x in range(19)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyCinVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([1, ]+[1 for x in range(19)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyPoutVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([1, ]+[1 for x in range(19)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyCoutVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([1, ]+[1 for x in range(19)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeGraphConstVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([1 for x in range(20)])
        p_out = np.array([1 for x in range(20)])
        c_out = np.array([1 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeGraphVariable():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([x for x in range(20)])
        p_out = np.array([x**2 for x in range(20)])
        c_out = np.array([(x-10)**3 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeGraphVariable1():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([abs(x-10) for x in range(20)])
        p_out = np.array([x**2-x for x in range(20)])
        c_out = np.array([(x-10)**3 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoEdgeGraphConstVect():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([1 for x in range(20)])
        p_out = np.array([1 for x in range(20)])
        c_out = np.array([1 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoEdgeGraphVariable():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([x for x in range(20)])
        p_out = np.array([x**2 for x in range(20)])
        c_out = np.array([(x-10)**3 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoEdgeGraphVariable1():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([abs(x-10) for x in range(20)])
        p_out = np.array([x**2-x for x in range(20)])
        c_out = np.array([(x-10)**3 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_RandomGraphConstVect():
        rd.seed(2000)
        np.random.seed(2001)
        G = nx.erdos_renyi_graph(20, 0.1, directed=True)
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([1 for x in range(20)])
        p_out = np.array([1 for x in range(20)])
        c_out = np.array([1 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_RandomGraphVariable():
        rd.seed(2002)
        np.random.seed(2003)
        G = nx.erdos_renyi_graph(20, 0.1, directed=True)
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([x for x in range(20)])
        p_out = np.array([x**2 for x in range(20)])
        c_out = np.array([(x-10)**3 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)

###############


def test_BadCase0():
        G = nx.DiGraph()
        G.add_nodes_from(range(5))
        G.add_edge(3, 2)
        p_in = np.array([1 for x in range(5)])
        c_in = np.array([1 for x in range(5)])
        p_out = np.array([1 for x in range(5)])
        c_out = np.array([1 for x in range(5)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_BadCase():
        G = nx.DiGraph()
        G.add_nodes_from(range(5))
        G.add_edge(0, 2)
        G.add_edge(1, 2)
        G.add_edge(2, 2)
        G.add_edge(3, 2)
        G.add_edge(4, 3)
        p_in = np.array([1 for x in range(5)])
        c_in = np.array([1 for x in range(5)])
        p_out = np.array([1 for x in range(5)])
        c_out = np.array([1 for x in range(5)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)

######################


def test_RandomGraphSmallVariable1():
        rd.seed(2004)
        np.random.seed(2005)
        G = nx.erdos_renyi_graph(5, 0.1, directed=True)
        for i in range(len(G)):
            if rd.random() < 0.1:
                G.add_edge(i, i)
        p_in = np.array([1 for x in range(5)])
        c_in = np.array([abs(x-10) for x in range(5)])
        p_out = np.array([x**2-x for x in range(5)])
        c_out = np.array([(x-10)**3 for x in range(5)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_RandomGraphSmallConstVectSelfLoop():
        rd.seed(2000)
        np.random.seed(2001)
        for rep in range(100):
            G = nx.erdos_renyi_graph(5, 0.1, directed=True)
            for i in range(len(G)):
                if rd.random() < 0.1:
                    G.add_edge(i, i)
            p_in = np.array([1 for x in range(5)])
            c_in = np.array([1 for x in range(5)])
            p_out = np.array([1 for x in range(5)])
            c_out = np.array([1 for x in range(5)])
            __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_RandomGraphSmallVariableSelfLoop():
        rd.seed(2002)
        np.random.seed(2003)
        for rep in range(10):
            G = nx.erdos_renyi_graph(5, 0.1, directed=True)
            for i in range(len(G)):
                if rd.random() < 0.1:
                    G.add_edge(i, i)
            p_in = np.array([1 for x in range(5)])
            c_in = np.array([x for x in range(5)])
            p_out = np.array([x**2 for x in range(5)])
            c_out = np.array([(x-10)**3 for x in range(5)])
            __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_RandomGraphSmallVariable1SelfLoop():
        rd.seed(2004)
        np.random.seed(2005)
        for rep in range(10):
            G = nx.erdos_renyi_graph(5, 0.1, directed=True)
            for i in range(len(G)):
                if rd.random() < 0.1:
                    G.add_edge(i, i)
            p_in = np.array([1 for x in range(5)])
            c_in = np.array([abs(x-10) for x in range(5)])
            p_out = np.array([x**2-x for x in range(5)])
            c_out = np.array([(x-10)**3 for x in range(5)])
            __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


################

def test_RandomGraphVariable1():
        rd.seed(2004)
        np.random.seed(2005)
        G = nx.erdos_renyi_graph(20, 0.1, directed=True)
        for i in range(len(G)):
            if rd.random() < 0.1:
                G.add_edge(i, i)
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([abs(x-10) for x in range(20)])
        p_out = np.array([x**2-x for x in range(20)])
        c_out = np.array([(x-10)**3 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_RandomGraphConstVectSelfLoop():
        rd.seed(2000)
        np.random.seed(2001)
        for rep in range(10):
            G = nx.erdos_renyi_graph(20, 0.1, directed=True)
            for i in range(len(G)):
                if rd.random() < 0.1:
                    G.add_edge(i, i)
            p_in = np.array([1 for x in range(20)])
            c_in = np.array([1 for x in range(20)])
            p_out = np.array([1 for x in range(20)])
            c_out = np.array([1 for x in range(20)])
            __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_RandomGraphVariableSelfLoop():
        rd.seed(2002)
        np.random.seed(2003)
        for rep in range(10):
            G = nx.erdos_renyi_graph(20, 0.1, directed=True)
            for i in range(len(G)):
                if rd.random() < 0.1:
                    G.add_edge(i, i)
            p_in = np.array([1 for x in range(20)])
            c_in = np.array([x for x in range(20)])
            p_out = np.array([x**2 for x in range(20)])
            c_out = np.array([(x-10)**3 for x in range(20)])
            __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_RandomGraphVariable1SelfLoop():
        rd.seed(2004)
        np.random.seed(2005)
        for rep in range(10):
            G = nx.erdos_renyi_graph(20, 0.1, directed=True)
            for i in range(len(G)):
                if rd.random() < 0.1:
                    G.add_edge(i, i)
            p_in = np.array([1 for x in range(20)])
            c_in = np.array([abs(x-10) for x in range(20)])
            p_out = np.array([x**2-x for x in range(20)])
            c_out = np.array([(x-10)**3 for x in range(20)])
            __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


@pytest.mark.slowtest
def test_BigRandomGraphConstVect():
        rd.seed(2006)
        np.random.seed(2007)
        G = nx.erdos_renyi_graph(200, 0.1, directed=True)
        p_in = np.array([1 for x in range(200)])
        c_in = np.array([1 for x in range(200)])
        p_out = np.array([1 for x in range(200)])
        c_out = np.array([1 for x in range(200)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


@pytest.mark.slowtest
def test_BigRandomGraphVariable():
        rd.seed(2008)
        np.random.seed(2009)
        G = nx.erdos_renyi_graph(200, 0.1, directed=True)
        p_in = np.array([1 for x in range(200)])
        c_in = np.array([x for x in range(200)])
        p_out = np.array([x**2 for x in range(200)])
        c_out = np.array([(x-10)**3 for x in range(200)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


@pytest.mark.slowtest
def test_BigRandomGraphVariable1():
        rd.seed(2010)
        np.random.seed(2011)
        G = nx.erdos_renyi_graph(200, 0.1, directed=True)
        p_in = np.array([1 for x in range(200)])
        c_in = np.array([abs(x-10) for x in range(200)])
        p_out = np.array([x**2-x for x in range(200)])
        c_out = np.array([(x-10)**3 for x in range(200)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


@pytest.mark.slowtest
def test_BigRandomGraphVariable1SelfLoop():
        rd.seed(2010)
        np.random.seed(2011)
        G = nx.erdos_renyi_graph(200, 0.1, directed=True)
        for i in range(200):
            if rd.random() < 0.1:
                G.add_edge(i, i)
        p_in = np.array([1 for x in range(200)])
        c_in = np.array([abs(x-10) for x in range(200)])
        p_out = np.array([x**2-x for x in range(200)])
        c_out = np.array([(x-10)**3 for x in range(200)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)
