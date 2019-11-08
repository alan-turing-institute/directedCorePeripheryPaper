import random as rd
import numpy as np
import networkx as nx
import sys
from copy import deepcopy
sys.path.append('../')
import hitsMk2 as hm
import hitsMk2NonVect as hmN
import pytest


def hitsReallySimple(G, pin, pot, cin, cot):
    pinN = [0 for x in range(len(G))]
    cinN = [0 for x in range(len(G))]
    potN = [0 for x in range(len(G))]
    cotN = [0 for x in range(len(G))]
    hasE = G.has_edge
    q1 = G.number_of_edges()
    s1 = q1/float(len(G)*len(G))
    s2 = 1 - s1
    tPin = sum(pin)
    if tPin < 10**(-14):
        tPin = 1
    tPot = sum(pot)
    if tPot < 10**(-14):
        tPot = 1
    tCin = sum(cin)
    if tCin < 10**(-14):
        tCin = 1
    tCot = sum(cot)
    if tCot < 10**(-14):
        tCot = 1
    pin = [x/tPin for x in pin]
    cin = [x/tCin for x in cin]
    pot = [x/tPot for x in pot]
    cot = [x/tCot for x in cot]
    for i in range(len(G)):

        # first update pinN
        for j in range(len(G)):
            pinN[i] += -s2*hasE(i, j)*pin[j]
            pinN[i] += -s2*hasE(i, j)*pot[j]
            pinN[i] += -s2*hasE(i, j)*cin[j]
            pinN[i] += -s2*hasE(i, j)*cot[j]

            pinN[i] += +s1*(1-hasE(i, j))*pin[j]
            pinN[i] += +s1*(1-hasE(i, j))*pot[j]
            pinN[i] += +s1*(1-hasE(i, j))*cin[j]
            pinN[i] += +s1*(1-hasE(i, j))*cot[j]

            pinN[i] += -s2*hasE(j, i)*pin[j]
            pinN[i] += -s2*hasE(j, i)*pot[j]
            pinN[i] += -s2*hasE(j, i)*cin[j]
            pinN[i] += +s2*hasE(j, i)*cot[j]

            pinN[i] += +s1*(1-hasE(j, i))*pin[j]
            pinN[i] += +s1*(1-hasE(j, i))*pot[j]
            pinN[i] += +s1*(1-hasE(j, i))*cin[j]
            pinN[i] += -s1*(1-hasE(j, i))*cot[j]

        # first update cinN
        for j in range(len(G)):
            cinN[i] += -s2*hasE(i, j)*pin[j]
            cinN[i] += -s2*hasE(i, j)*pot[j]
            cinN[i] += +s2*hasE(i, j)*cin[j]
            cinN[i] += -s2*hasE(i, j)*cot[j]

            cinN[i] += +s1*(1-hasE(i, j))*pin[j]
            cinN[i] += +s1*(1-hasE(i, j))*pot[j]
            cinN[i] += -s1*(1-hasE(i, j))*cin[j]
            cinN[i] += +s1*(1-hasE(i, j))*cot[j]

            cinN[i] += -s2*hasE(j, i)*pin[j]
            cinN[i] += +s2*hasE(j, i)*pot[j]
            cinN[i] += +s2*hasE(j, i)*cin[j]
            cinN[i] += +s2*hasE(j, i)*cot[j]

            cinN[i] += +s1*(1-hasE(j, i))*pin[j]
            cinN[i] += -s1*(1-hasE(j, i))*pot[j]
            cinN[i] += -s1*(1-hasE(j, i))*cin[j]
            cinN[i] += -s1*(1-hasE(j, i))*cot[j]

        # first update cotN
        for j in range(len(G)):
            cotN[i] += +s2*hasE(i, j)*pin[j]
            cotN[i] += -s2*hasE(i, j)*pot[j]
            cotN[i] += +s2*hasE(i, j)*cin[j]
            cotN[i] += +s2*hasE(i, j)*cot[j]

            cotN[i] += -s1*(1-hasE(i, j))*pin[j]
            cotN[i] += +s1*(1-hasE(i, j))*pot[j]
            cotN[i] += -s1*(1-hasE(i, j))*cin[j]
            cotN[i] += -s1*(1-hasE(i, j))*cot[j]

            cotN[i] += -s2*hasE(j, i)*pin[j]
            cotN[i] += -s2*hasE(j, i)*pot[j]
            cotN[i] += -s2*hasE(j, i)*cin[j]
            cotN[i] += +s2*hasE(j, i)*cot[j]

            cotN[i] += +s1*(1-hasE(j, i))*pin[j]
            cotN[i] += +s1*(1-hasE(j, i))*pot[j]
            cotN[i] += +s1*(1-hasE(j, i))*cin[j]
            cotN[i] += -s1*(1-hasE(j, i))*cot[j]

        # first update potN
        for j in range(len(G)):
            potN[i] += -s2*hasE(i, j)*pin[j]
            potN[i] += -s2*hasE(i, j)*pot[j]
            potN[i] += +s2*hasE(i, j)*cin[j]
            potN[i] += -s2*hasE(i, j)*cot[j]

            potN[i] += +s1*(1-hasE(i, j))*pin[j]
            potN[i] += +s1*(1-hasE(i, j))*pot[j]
            potN[i] += -s1*(1-hasE(i, j))*cin[j]
            potN[i] += +s1*(1-hasE(i, j))*cot[j]

            potN[i] += -s2*hasE(j, i)*pin[j]
            potN[i] += -s2*hasE(j, i)*pot[j]
            potN[i] += -s2*hasE(j, i)*cin[j]
            potN[i] += -s2*hasE(j, i)*cot[j]

            potN[i] += +s1*(1-hasE(j, i))*pin[j]
            potN[i] += +s1*(1-hasE(j, i))*pot[j]
            potN[i] += +s1*(1-hasE(j, i))*cin[j]
            potN[i] += +s1*(1-hasE(j, i))*cot[j]
    # print([s1, s2])
    # print(np.array([potN, cinN, cotN, pinN]).T)
    pinN1, potN1, cinN1, cotN1 = hmN.__getNormNonVect__(pinN, potN, cinN, cotN)
    return [potN1, cinN1, cotN1, pinN1]


def slowMethod(G, p_in, p_out, c_in, c_out, combined=True):
    A, mA, At, mAt = hmN.__makeMatrices__(G)
    q1 = A.sum()
    ptot1 = p_in.sum()
    if ptot1 < 10**(-14):
        ptot1 = 1
    p_in1 = p_in/ptot1
    ctot1 = c_in.sum()
    if ctot1 < 10**(-14):
        ctot1 = 1
    c_in1 = c_in/ctot1
    ptot1 = p_out.sum()
    if ptot1 < 10**(-14):
        ptot1 = 1
    p_out1 = p_out/ptot1
    ctot1 = c_out.sum()
    if ctot1 < 10**(-14):
        ctot1 = 1
    c_out1 = c_out/ctot1
    h = hmN
    PoNew = h.updatePout_mk2(p_in1, p_out1, c_in1, c_out1, A, mA, At, mAt, q1)
    CoNew = h.updateCout_mk2(p_in1, p_out1, c_in1, c_out1, A, mA, At, mAt, q1)
    PiNew = h.updatePin_mk2(p_in1, p_out1, c_in1, c_out1, A, mA, At, mAt, q1)
    CiNew = h.updateCin_mk2(p_in1, p_out1, c_in1, c_out1, A, mA, At, mAt, q1)
    p_out, c_in, c_out, p_in = h.__getNormNonVect__(PoNew, CiNew, CoNew, PiNew)
    if combined:
        combined = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
        return combined
    else:
        return p_in, p_out, c_in, c_out


def __compareVectToNonVector__(G, p_in1, p_out1, c_in1, c_out1):
    # test one change
    __compareVectToNonVectorOneChange__(G, p_in1, p_out1, c_in1, c_out1)
    # test with small perturbation (tests if results are sensitive to small
    # values
    r1 = np.array([rd.random()*10**(-15) for x in range(len(p_in1))])
    r2 = np.array([rd.random()*10**(-15) for x in range(len(p_in1))])
    r3 = np.array([rd.random()*10**(-15) for x in range(len(p_in1))])
    r4 = np.array([rd.random()*10**(-15) for x in range(len(p_in1))])

    p_in2 = p_in1 + r1
    p_out2 = p_out1 + r2
    c_in2 = c_in1 + r3
    c_out2 = c_out1 + r4
    __compareVectToNonVectorOneChange__(G, p_in2, p_out2, c_in2, c_out2)


def __compareVectToNonVectorOneChange__(G, p_in1, p_out1, c_in1, c_out1):
    p_in = np.array(deepcopy(p_in1)).reshape([len(G), 1])
    p_out = np.array(deepcopy(p_out1)).reshape([len(G), 1])
    c_in = np.array(deepcopy(c_in1)).reshape([len(G), 1])
    c_out = np.array(deepcopy(c_out1)).reshape([len(G), 1])
    p_out, c_in, c_out, p_in = hmN.__getNormNonVect__(p_out, c_in, c_out, p_in)
    # make copy here just in case of modification
    oldVer = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
    slowRes = slowMethod(G, p_in, p_out, c_in, c_out)
    p_in = np.array(deepcopy(p_in1)).reshape([len(G), 1])
    p_out = np.array(deepcopy(p_out1)).reshape([len(G), 1])
    c_in = np.array(deepcopy(c_in1)).reshape([len(G), 1])
    c_out = np.array(deepcopy(c_out1)).reshape([len(G), 1])
    p_out, c_in, c_out, p_in = hmN.__getNormNonVect__(p_out, c_in, c_out, p_in)
    potN1, cinN1, cotN1, pinN1 = hitsReallySimple(G, p_in, p_out, c_in, c_out)
    reallySlowRes = np.concatenate([potN1, cinN1, cotN1, pinN1], axis=1)
    A, At, C, Ct, n, s1, s2 = hm.__setup__(G)
    C2 = C +Ct
    oldVer = hm.__hitsmk2Group__(A, At, C, Ct, oldVer, s1, C2)
    oldVer = hm.__getNorm__(oldVer)
    assert(abs(oldVer-slowRes).max() < 10**(-8))
    assert(abs(oldVer-reallySlowRes).max() < 10**(-8))


def test_TwoNodeOneEdgeOnlyOneCoutVectSelfLoop():
    rd.seed(100000)
    G = nx.DiGraph()
    G.add_edge(0, 1)
    G.add_edge(0, 0)
    p_in = np.array([0 for x in range(2)])
    c_in = np.array([0 for x in range(2)])
    p_out = np.array([0 for x in range(2)])
    c_out = np.array([1, ]+[0 for x in range(1)])
    __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyOnePinVect():
        rd.seed(100001)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1, ]+[0 for x in range(19)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyOneCinVect():
        rd.seed(100002)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([1, ]+[0 for x in range(19)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyOnePoutVect():
        rd.seed(100003)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([1, ]+[0 for x in range(19)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyOneCoutVect():
        rd.seed(100004)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([1, ]+[0 for x in range(19)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOneCoutVect():
        rd.seed(100005)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([1, ]+[0 for x in range(1)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOnePinVect():
        rd.seed(100006)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        p_in = np.array([1, ]+[0 for x in range(1)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOneCinVect():
        rd.seed(100007)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([1, ]+[0 for x in range(1)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOnePoutVect():
        rd.seed(100008)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([1, ]+[0 for x in range(1)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyPinVect():
        rd.seed(100009)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1, ]+[1 for x in range(19)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyCinVect():
        rd.seed(100010)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([1, ]+[1 for x in range(19)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyPoutVect():
        rd.seed(100011)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([1, ]+[1 for x in range(19)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyCoutVect():
        rd.seed(100012)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([1, ]+[1 for x in range(19)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeGraphConstVect():
        rd.seed(100013)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([1 for x in range(20)])
        p_out = np.array([1 for x in range(20)])
        c_out = np.array([1 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeGraphVariable():
        rd.seed(100014)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([x for x in range(20)])
        p_out = np.array([x**2 for x in range(20)])
        c_out = np.array([(x-10)**3 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeGraphVariable1():
        rd.seed(100015)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([abs(x-10) for x in range(20)])
        p_out = np.array([x**2-x for x in range(20)])
        c_out = np.array([(x-10)**3 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoEdgeGraphConstVect():
        rd.seed(100016)
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
        rd.seed(100017)
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
        rd.seed(100018)
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
        rd.seed(100019)
        rd.seed(2000)
        np.random.seed(2001)
        G = nx.erdos_renyi_graph(20, 0.1, directed=True)
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([1 for x in range(20)])
        p_out = np.array([1 for x in range(20)])
        c_out = np.array([1 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_RandomGraphVariable():
        rd.seed(100020)
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
        rd.seed(100021)
        G = nx.DiGraph()
        G.add_nodes_from(range(5))
        G.add_edge(3, 2)
        p_in = np.array([1 for x in range(5)])
        c_in = np.array([1 for x in range(5)])
        p_out = np.array([1 for x in range(5)])
        c_out = np.array([1 for x in range(5)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_BadCase():
        rd.seed(100022)
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
