import random as rd
import numpy as np
import networkx as nx
import sys
from copy import deepcopy
sys.path.append('../')
import hitsMk2 as hm
import hitsMk2NonVect as hmNV
import pytest


def hitsReallySimple(G, pin0, pot0, cin0, cot0,t1,flag=False):
    pinN = [0 for x in range(len(G))]
    cinN = [0 for x in range(len(G))]
    potN = [0 for x in range(len(G))]
    cotN = [0 for x in range(len(G))]
    hasE = G.has_edge
    q1 = G.number_of_edges()
    s1 = q1/float(len(G)*len(G))
    s2 = 1 - s1
    tPin  = float(sum(pin0))
    if tPin<10**(-14):
       tPin=1
    tPot = float(sum(pot0))
    if tPot<10**(-14):
       tPot=1
    tCin  = float(sum(cin0))
    if tCin<10**(-14):
       tCin=1
    tCot = float(sum(cot0))
    if tCot<10**(-14):
       tCot=1
    pin = [float(x)/tPin if x>10**(-14) else 0 for x in pin0 ]
    cin = [float(x)/tCin if x>10**(-14) else 0  for x in cin0]
    pot = [float(x)/tPot if x>10**(-14) else 0  for x in pot0]
    cot = [float(x)/tCot if x>10**(-14) else 0 for x in cot0]
    #for i in range(len(G)):
    for i in [t1,]:

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
#    print(np.array([potN, cinN, cotN, pinN]).T)
#    if flag:
#        print(np.array([pinN, potN, cinN, cotN]).T)
#        import pdb
#        pdb.set_trace()
    pinN1, potN1, cinN1, cotN1 = hmNV.__getNormNonVect__(pinN, potN, cinN, cotN)
    pot = np.array([float(x) for x in pot0])
    pin = np.array([float(x) for x in pin0])
    cot = np.array([float(x) for x in cot0])
    cin = np.array([float(x) for x in cin0])
    pot[t1]=float(potN1[t1])
    pin[t1]=float(pinN1[t1])
    cot[t1]=float(cotN1[t1])
    cin[t1]=float(cinN1[t1])
    pot = pot.reshape([1,len(pot0)])
    pin = pin.reshape([1,len(pot0)])
    cot = cot.reshape([1,len(pot0)])
    cin = cin.reshape([1,len(pot0)])
    return np.concatenate([pot, cin, cot, pin]).T


def slowMethod(G,p_in, p_out, c_in, c_out,t1,combined=True,flag=False):
    A, mA, At, mAt = hmNV.__makeMatrices__(G)
    q1= A.sum()
    ptot1 = p_in.sum()
    if ptot1<10**(-14):
       ptot1=1
    p_in1   = p_in/ptot1
    ctot1 = c_in.sum()
    if ctot1<10**(-14):
       ctot1=1
    c_in1   = c_in/ctot1
    ptot1 = p_out.sum()
    if ptot1<10**(-14):
       ptot1=1
    p_out1  = p_out/ptot1
    ctot1 = c_out.sum()
    if ctot1<10**(-14):
       ctot1=1
    c_out1  = c_out/ctot1
    #if flag:
    #    import pdb
    #    pdb.set_trace()
    PoutNew = hmNV.updatePout_mk2(p_in1, p_out1, c_in1, c_out1, A, mA, At, mAt, q1)
    CoutNew = hmNV.updateCout_mk2(p_in1, p_out1, c_in1, c_out1, A, mA, At, mAt, q1)
    PinNew  = hmNV.updatePin_mk2(p_in1, p_out1, c_in1, c_out1, A, mA, At, mAt, q1)
    CinNew  = hmNV.updateCin_mk2(p_in1, p_out1, c_in1, c_out1, A, mA, At, mAt, q1)
    p_out[t1] = PoutNew[t1]
    c_out[t1] = CoutNew[t1]
    p_in[t1] = PinNew[t1]
    c_in[t1] = CinNew[t1]
    p_out, c_in, c_out, p_in = hmNV.__getNormNonVect__(p_out, c_in, c_out, p_in)
    #if flag:
    #    import pdb
    #    pdb.set_trace()
    if combined:
        combined = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
        return combined
    else:
       return p_in, p_out, c_in, c_out

def __compareVectToNonVector__(G, p_in1, p_out1, c_in1, c_out1):
    ## test one change
    __compareVectToNonVectorOneChange__(G, p_in1, p_out1, c_in1, c_out1)
    ## test all changes
    __compareVectToNonVectorAllChanges__(G, p_in1, p_out1, c_in1, c_out1)
    ## test with small perturbation (tests if results are sensitive to small
    ## values
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
    A,At,C,Ct,n,s1,s2 = hm.__setup__(G)
    p_in = np.array(deepcopy(p_in1)).reshape([len(G), 1])
    p_out = np.array(deepcopy(p_out1)).reshape([len(G), 1])
    c_in = np.array(deepcopy(c_in1)).reshape([len(G), 1])
    c_out = np.array(deepcopy(c_out1)).reshape([len(G), 1])
    oldVer = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
    As,Ats,totals =  hm.__setupOneAtATime__(A,At,oldVer)
    for t1 in range(len(p_in1)):
        p_in = np.array(deepcopy(p_in1)).reshape([len(G), 1])
        p_out = np.array(deepcopy(p_out1)).reshape([len(G), 1])
        c_in = np.array(deepcopy(c_in1)).reshape([len(G), 1])
        c_out = np.array(deepcopy(c_out1)).reshape([len(G), 1])
        p_out, c_in, c_out, p_in = hmNV.__getNormNonVect__(p_out, c_in, c_out, p_in)
        ## make copy here just in case of modification
        oldVerOrig = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
        oldVer = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
        slowRes = slowMethod(G,p_in, p_out, c_in, c_out,t1)
        p_in = np.array(deepcopy(p_in1)).reshape([len(G), 1])
        p_out = np.array(deepcopy(p_out1)).reshape([len(G), 1])
        c_in = np.array(deepcopy(c_in1)).reshape([len(G), 1])
        c_out = np.array(deepcopy(c_out1)).reshape([len(G), 1])
        p_out, c_in, c_out, p_in = hmNV.__getNormNonVect__(p_out, c_in, c_out, p_in)
        reallySlow = hitsReallySimple(G, p_in, p_out, c_in, c_out,t1)


        x,max1 = hm.__hitsmk2OneAtATimeGroup__(As, Ats, t1, C, Ct, oldVer,s1,s2)
        assert(abs(oldVer-slowRes).max()<10**(-8))
        assert(abs(oldVer-reallySlow).max()<10**(-8))


def __compareVectToNonVectorAllChanges__(G, p_in1, p_out1, c_in1, c_out1):
    A,At,C,Ct,n,s1,s2 = hm.__setup__(G)
    p_in = np.array(deepcopy(p_in1)).reshape([len(G), 1])
    p_out = np.array(deepcopy(p_out1)).reshape([len(G), 1])
    c_in = np.array(deepcopy(c_in1)).reshape([len(G), 1])
    c_out = np.array(deepcopy(c_out1)).reshape([len(G), 1])
    oldVer = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
    As,Ats,totals =  hm.__setupOneAtATime__(A,At,oldVer)
    for rep in range(10):
        p_in = np.array(deepcopy(p_in1)).reshape([len(G), 1])
        p_out = np.array(deepcopy(p_out1)).reshape([len(G), 1])
        c_in = np.array(deepcopy(c_in1)).reshape([len(G), 1])
        c_out = np.array(deepcopy(c_out1)).reshape([len(G), 1])
        p_out, c_in, c_out, p_in = hmNV.__getNormNonVect__(p_out, c_in, c_out, p_in)
        oldVer = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
        order = list(range(len(p_in1)))
        rd.shuffle(order)
        maxDiff=-100
        for t1 in order:
            ## make copy here just in case of modification
#            oldVerOrig = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
#            oldVerOrig1 = oldVer.copy()
            reallySlow = hitsReallySimple(G, p_in.copy(), p_out.copy(), c_in.copy(), c_out.copy(),t1)
            p_in, p_out, c_in, c_out = slowMethod(G,p_in, p_out, c_in, c_out,t1,False)
            x,max1 = hm.__hitsmk2OneAtATimeGroup__(As, Ats, t1, C, Ct, oldVer,s1,s2)
            if max1>maxDiff:
               maxDiff=max1
            slowRes = np.concatenate([p_out, c_in, c_out, p_in], axis=1)
            assert(abs(oldVer-slowRes).max()<10**(-8))
            assert(abs(oldVer-reallySlow).max()<10**(-8))
        p_in2 = np.array(deepcopy(p_in1)).reshape([len(G), 1])
        p_out2 = np.array(deepcopy(p_out1)).reshape([len(G), 1])
        c_in2 = np.array(deepcopy(c_in1)).reshape([len(G), 1])
        c_out2 = np.array(deepcopy(c_out1)).reshape([len(G), 1])
        p_out2, c_in2, c_out2, p_in2 = hmNV.__getNormNonVect__(p_out2, c_in2, c_out2, p_in2)
        original = np.concatenate([p_out2, c_in2, c_out2, p_in2], axis=1)
        actualMaxDiff = abs(original-oldVer).max()
        p_outM = abs(p_out2-p_out).max()
        c_outM = abs(c_out2-c_out).max()
        p_inM  = abs(p_in2-p_in).max()
        c_inM  = abs(c_in2-c_in).max()
        slowMax = max([p_outM,c_outM,p_inM,c_inM])
        assert(abs(actualMaxDiff-maxDiff)<10**(-10))
        assert(abs(actualMaxDiff-slowMax)<10**(-10))
        assert(abs(maxDiff-slowMax)<10**(-10))





def test_TwoNodeOneEdgeOnlyOneCoutVectSelfLoop():
        rd.seed(3000)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([1, ]+[0 for x in range(1)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyOnePinVect():
        rd.seed(3001)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1, ]+[0 for x in range(19)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyOneCinVect():
        rd.seed(3002)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([1, ]+[0 for x in range(19)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyOnePoutVect():
        rd.seed(3002)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([1, ]+[0 for x in range(19)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyOneCoutVect():
        rd.seed(3003)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([1, ]+[0 for x in range(19)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOneCoutVect():
        rd.seed(3004)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([1, ]+[0 for x in range(1)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOnePinVect():
        rd.seed(3005)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        p_in = np.array([1, ]+[0 for x in range(1)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOneCinVect():
        rd.seed(3006)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([1, ]+[0 for x in range(1)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOnePoutVect():
        rd.seed(3007)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([1, ]+[0 for x in range(1)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyPinVect():
        rd.seed(3008)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1, ]+[1 for x in range(19)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyCinVect():
        rd.seed(3009)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([1, ]+[1 for x in range(19)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)



def test_OneEdgeOnlyPoutVect():
        rd.seed(3010)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([1, ]+[1 for x in range(19)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeOnlyCoutVect():
        rd.seed(3011)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([1, ]+[1 for x in range(19)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeGraphConstVect():
        rd.seed(3012)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([1 for x in range(20)])
        p_out = np.array([1 for x in range(20)])
        c_out = np.array([1 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeGraphVariable():
        rd.seed(3013)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([x for x in range(20)])
        p_out = np.array([x**2 for x in range(20)])
        c_out = np.array([(x-10)**3 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_OneEdgeGraphVariable1():
        rd.seed(3014)
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([abs(x-10) for x in range(20)])
        p_out = np.array([x**2-x for x in range(20)])
        c_out = np.array([(x-10)**3 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoEdgeGraphConstVect():
        rd.seed(3015)
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
        rd.seed(3016)
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
        rd.seed(3017)
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
        rd.seed(3020)
        G = nx.DiGraph()
        G.add_nodes_from(range(5))
        G.add_edge(3,2)
        p_in = np.array([1 for x in range(5)])
        c_in = np.array([1 for x in range(5)])
        p_out = np.array([1 for x in range(5)])
        c_out = np.array([1 for x in range(5)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)

def test_BadCase():
        rd.seed(3021)
        G = nx.DiGraph()
        G.add_nodes_from(range(5))
        G.add_edge(0,2)
        G.add_edge(1,2)
        G.add_edge(2,2)
        G.add_edge(3,2)
        G.add_edge(4,3)
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


@pytest.mark.slowtest
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
