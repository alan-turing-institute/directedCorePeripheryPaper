import random as rd
import numpy as np
import networkx as nx
import sys
from copy import deepcopy
from scipy import sparse
sys.path.append('../')
import hitsMk2 as hm
import hitsMk2NonVect as hmNV
import pytest


# test first norm function
def test_IntAllEqualNormTest():
    p_in =  np.array([0, 0, 0, 0, 1, 0, 0, 0, ]).reshape([8,1])
    p_out = np.array([0, 1, 0, 0, 0, 1, 0, 0, ]).reshape([8,1])
    c_in =  np.array([0, 0, 1, 0, 0, 0, 0, 0, ]).reshape([8,1])
    c_out = np.array([0, 0, 0, 1, 0, 0, 0, 1, ]).reshape([8,1])
    p_inI = p_in.astype(np.int)  + 10**(-15)
    p_outI =p_out.astype(np.int) - 10**(-15)
    c_inI = c_in.astype(np.int)  + 10**(-15)
    c_outI =c_out.astype(np.int) - 10**(-15)
    a,b,c,d = hmNV.__getNormNonVect__(p_inI, p_outI, c_inI, c_outI)
    for i in [0,6]:
        assert(abs(a[i]-0.25)<10**(-8))
        assert(abs(b[i]-0.25)<10**(-8))
        assert(abs(c[i]-0.25)<10**(-8))
        assert(abs(d[i]-0.25)<10**(-8))

    t1 = np.concatenate([p_outI, c_inI, c_outI, p_inI], axis=1)
    t2 = hm.__getNorm__(t1)
    for i in [0,6]:
        assert(abs(t2[i,0]-0.25)<10**(-8))
        assert(abs(t2[i,1]-0.25)<10**(-8))
        assert(abs(t2[i,2]-0.25)<10**(-8))
        assert(abs(t2[i,3]-0.25)<10**(-8))
    p_in =  np.array([2, 2, 1, 2, 1, 6, 5.5, 7, ]).reshape([8,1])
    p_out = np.array([2, 1, 3, 2, 5, 1, 5.5, 3, ]).reshape([8,1])
    c_in =  np.array([2, 3, 1, 2, 3, 2, 5.5, 9, ]).reshape([8,1])
    c_out = np.array([2, 4, 2, 1, 2, 5, 5.5, 1, ]).reshape([8,1])
    p_inF = p_in.astype(np.float) + 10**(-15)
    p_outF =p_out.astype(np.float)- 10**(-15)
    c_inF = c_in.astype(np.float) + 10**(-15)
    c_outF =c_out.astype(np.float)- 10**(-15)
    a,b,c,d = hmNV.__getNormNonVect__(p_inF, p_outF, c_inF, c_outF)
    for i in [0,6]:
        assert(abs(a[i]-0.25)<10**(-8))
        assert(abs(b[i]-0.25)<10**(-8))
        assert(abs(c[i]-0.25)<10**(-8))
        assert(abs(d[i]-0.25)<10**(-8))
    t1 = np.concatenate([p_outF, c_inF, c_outF, p_inF], axis=1)
    t2 = hm.__getNorm__(t1)
    for i in [0,6]:
        assert(abs(t2[i,0]-0.25)<10**(-8))
        assert(abs(t2[i,1]-0.25)<10**(-8))
        assert(abs(t2[i,2]-0.25)<10**(-8))
        assert(abs(t2[i,3]-0.25)<10**(-8))


def test_FloatAllEqualNormTest():
    p_in =  np.array([0, 0, 0, 0, 1, 0, 0, 0, ]).reshape([8,1])
    p_out = np.array([0, 1, 0, 0, 0, 1, 0, 0, ]).reshape([8,1])
    c_in =  np.array([0, 0, 1, 0, 0, 0, 0, 0, ]).reshape([8,1])
    c_out = np.array([0, 0, 0, 1, 0, 0, 0, 1, ]).reshape([8,1])
    p_inF = p_in.astype(np.float)  + 10**(-15)
    p_outF =p_out.astype(np.float) - 10**(-15)
    c_inF = c_in.astype(np.float)  + 10**(-15)
    c_outF =c_out.astype(np.float) - 10**(-15)
    a,b,c,d = hmNV.__getNormNonVect__(p_inF, p_outF, c_inF, c_outF)
    for i in [0,6]:
        assert(abs(a[i]-0.25)<10**(-8))
        assert(abs(b[i]-0.25)<10**(-8))
        assert(abs(c[i]-0.25)<10**(-8))
        assert(abs(d[i]-0.25)<10**(-8))

    t1 = np.concatenate([p_outF, c_inF, c_outF, p_inF], axis=1)
    t2 = hm.__getNorm__(t1)
    for i in [0,6]:
        assert(abs(t2[i,0]-0.25)<10**(-8))
        assert(abs(t2[i,1]-0.25)<10**(-8))
        assert(abs(t2[i,2]-0.25)<10**(-8))
        assert(abs(t2[i,3]-0.25)<10**(-8))
    p_in =  np.array([2, 2, 1, 2, 1, 6, 5.5, 7, ]).reshape([8,1])
    p_out = np.array([2, 1, 3, 2, 5, 1, 5.5, 3, ]).reshape([8,1])
    c_in =  np.array([2, 3, 1, 2, 3, 2, 5.5, 9, ]).reshape([8,1])
    c_out = np.array([2, 4, 2, 1, 2, 5, 5.5, 1, ]).reshape([8,1])
    p_inF = p_in.astype(np.float) + 10**(-15)
    p_outF =p_out.astype(np.float)- 10**(-15)
    c_inF = c_in.astype(np.float) + 10**(-15)
    c_outF =c_out.astype(np.float)- 10**(-15)
    a,b,c,d = hmNV.__getNormNonVect__(p_inF, p_outF, c_inF, c_outF)
    for i in [0,6]:
        assert(abs(a[i]-0.25)<10**(-8))
        assert(abs(b[i]-0.25)<10**(-8))
        assert(abs(c[i]-0.25)<10**(-8))
        assert(abs(d[i]-0.25)<10**(-8))
    t1 = np.concatenate([p_outF, c_inF, c_outF, p_inF], axis=1)
    t2 = hm.__getNorm__(t1)
    for i in [0,6]:
        assert(abs(t2[i,0]-0.25)<10**(-8))
        assert(abs(t2[i,1]-0.25)<10**(-8))
        assert(abs(t2[i,2]-0.25)<10**(-8))
        assert(abs(t2[i,3]-0.25)<10**(-8))



def test_getNormOneOnes():
    p_in = [1, 0, 0, 0, 1, 0, 0, 0, ]
    p_out = [0, 1, 0, 0, 0, 1, 0, 0, ]
    c_in = [0, 0, 1, 0, 0, 0, 1, 0, ]
    c_out = [0, 0, 0, 1, 0, 0, 0, 1, ]
    oldp_in = p_in[:]
    oldp_out = p_out[:]
    oldc_in = c_in[:]
    oldc_out = c_out[:]
    p_in, p_out, c_in, c_out = hmNV.__getNormNonVect__(p_in, p_out, c_in, c_out)
    assert((p_in == oldp_in).all())
    assert((c_in == oldc_in).all())
    assert((p_out == oldp_out).all())
    assert((c_out == oldc_out).all())


def test_getNormAllOnes():
    p_in = [1, 1, 1, 1, 1, 1, 1, 1, ]
    p_out = [1, 1, 1, 1, 1, 1, 1, 1, ]
    c_in = [1, 1, 1, 1, 1, 1, 1, 1, ]
    c_out = [1, 1, 1, 1, 1, 1, 1, 1, ]
    p_in, p_out, c_in, c_out = hmNV.__getNormNonVect__(p_in, p_out, c_in, c_out)
    for i in range(8):
            assert(abs(p_in[i]-0.25) < 10**(-8))
            assert(abs(c_in[i]-0.25) < 10**(-8))
            assert(abs(p_out[i]-0.25) < 10**(-8))
            assert(abs(c_out[i]-0.25) < 10**(-8))


def test_getNormOnesWithMinus():
    # testing the behaviour with negative values
    p_in = np.array([1, 0, 0, 0, 1, 0, 0, 0, ])
    p_out = np.array([0, 1, 0, 0, 0, 1, 0, 0, ])
    c_in = np.array([0, 0, 1, 0, 0, 0, 1, 0, ])
    c_out = np.array([0, 0, 0, 1, 0, 0, 0, 1, ])
    oldp_in = p_in[:]
    oldp_out = p_out[:]
    oldc_in = c_in[:]
    oldc_out = c_out[:]
    getNorm = hmNV.__getNormNonVect__
    p_in, p_out, c_in, c_out = getNorm(p_in-1, p_out-1, c_in-1, c_out-1)
    assert((p_in == oldp_in).all())
    assert((c_in == oldc_in).all())
    assert((p_out == oldp_out).all())
    assert((c_out == oldc_out).all())


def test_getNormFiveOnes():
    p_in = [5, 1, 1, 1, 5, 1, 1, 1, ]
    p_out = [1, 5, 1, 1, 1, 5, 1, 1, ]
    c_in = [1, 1, 5, 1, 1, 1, 5, 1, ]
    c_out = [1, 1, 1, 5, 1, 1, 1, 5, ]
    getNorm = hmNV.__getNormNonVect__
    p_in, p_out, c_in, c_out = getNorm(p_in, p_out, c_in, c_out)
    for i in range(8):
        if i % 4 == 0:
            assert(abs(p_in[i]-1) < 10**(-8))
            assert(abs(p_out[i]-0) < 10**(-8))
            assert(abs(c_in[i]-0) < 10**(-8))
            assert(abs(c_out[i]-0) < 10**(-8))
        if i % 4 == 1:
            assert(abs(p_in[i]-0) < 10**(-8))
            assert(abs(p_out[i]-1) < 10**(-8))
            assert(abs(c_in[i]-0) < 10**(-8))
            assert(abs(c_out[i]-0) < 10**(-8))
        if i % 4 == 2:
            assert(abs(p_in[i]-0) < 10**(-8))
            assert(abs(p_out[i]-0) < 10**(-8))
            assert(abs(c_in[i]-1) < 10**(-8))
            assert(abs(c_out[i]-0) < 10**(-8))
        if i % 4 == 3:
            assert(abs(p_in[i]-0) < 10**(-8))
            assert(abs(p_out[i]-0) < 10**(-8))
            assert(abs(c_in[i]-0) < 10**(-8))
            assert(abs(c_out[i]-1) < 10**(-8))


def test_getNormFiveTwoOnes():
    p_in = [5, 1, 1, 2, 5, 1, 1, 2, ]
    p_out = [2, 5, 1, 1, 2, 5, 1, 1, ]
    c_in = [1, 2, 5, 1, 1, 2, 5, 1, ]
    c_out = [1, 1, 2, 5, 1, 1, 2, 5, ]
    getNorm = hmNV.__getNormNonVect__
    p_in, p_out, c_in, c_out = getNorm(p_in, p_out, c_in, c_out)
    for i in range(8):
        if i % 4 == 0:
            assert(abs(p_in[i]-0.8) < 10**(-8))
            assert(abs(p_out[i]-0.2) < 10**(-8))
            assert(abs(c_in[i]-0) < 10**(-8))
            assert(abs(c_out[i]-0) < 10**(-8))
        if i % 4 == 1:
            assert(abs(p_in[i]-0) < 10**(-8))
            assert(abs(p_out[i]-0.8) < 10**(-8))
            assert(abs(c_in[i]-0.2) < 10**(-8))
            assert(abs(c_out[i]-0) < 10**(-8))
        if i % 4 == 2:
            assert(abs(p_in[i]-0) < 10**(-8))
            assert(abs(p_out[i]-0) < 10**(-8))
            assert(abs(c_in[i]-0.8) < 10**(-8))
            assert(abs(c_out[i]-0.2) < 10**(-8))
        if i % 4 == 3:
            assert(abs(p_in[i]-0.2) < 10**(-8))
            assert(abs(p_out[i]-0) < 10**(-8))
            assert(abs(c_in[i]-0) < 10**(-8))
            assert(abs(c_out[i]-0.8) < 10**(-8))


def test_getNormFiveTwoMinusOnes():
    p_in = [5, -1, -1, 2, 5, -1, -1, 2, ]
    p_out = [2, 5, -1, -1, 2, 5, -1, -1, ]
    c_in = [-1, 2, 5, -1, -1, 2, 5, -1, ]
    c_out = [-1, -1, 2, 5, -1, -1, 2, 5, ]
    getNorm = hmNV.__getNormNonVect__
    p_in, p_out, c_in, c_out = getNorm(p_in, p_out, c_in, c_out)
    for i in range(8):
        if i % 4 == 0:
            assert(abs(p_in[i]-6.0/9.0) < 10**(-8))
            assert(abs(p_out[i]-3.0/9.0) < 10**(-8))
            assert(abs(c_in[i]-0) < 10**(-8))
            assert(abs(c_out[i]-0) < 10**(-8))
        if i % 4 == 1:
            assert(abs(p_in[i]-0) < 10**(-8))
            assert(abs(p_out[i]-6.0/9.0) < 10**(-8))
            assert(abs(c_in[i]-3.0/9.0) < 10**(-8))
            assert(abs(c_out[i]-0) < 10**(-8))
        if i % 4 == 2:
            assert(abs(p_in[i]-0) < 10**(-8))
            assert(abs(p_out[i]-0) < 10**(-8))
            assert(abs(c_in[i]-6.0/9.0) < 10**(-8))
            assert(abs(c_out[i]-3.0/9.0) < 10**(-8))
        if i % 4 == 3:
            assert(abs(p_in[i]-3.0/9.0) < 10**(-8))
            assert(abs(p_out[i]-0) < 10**(-8))
            assert(abs(c_in[i]-0) < 10**(-8))
            assert(abs(c_out[i]-6.0/9.0) < 10**(-8))


# test vector norm function

def test_getNormVectOneOnes():
        scores = np.array(
            [[1, 0, 0, 0, 1, 0, 0, 0, ],
             [0, 1, 0, 0, 0, 1, 0, 0, ],
             [0, 0, 1, 0, 0, 0, 1, 0, ],
             [0, 0, 0, 1, 0, 0, 0, 1, ]]).T
        scores1 = np.array(
            [[1, 0, 0, 0, 1, 0, 0, 0, ],
             [0, 1, 0, 0, 0, 1, 0, 0, ],
             [0, 0, 1, 0, 0, 0, 1, 0, ],
             [0, 0, 0, 1, 0, 0, 0, 1, ]]).T
        scores = hm.__getNorm__(scores)
        assert(abs(scores-scores1).max() < 10**(-8))


def test_getNormVectAllOnes():
        scores = np.array(
            [[1, 1, 1, 1, 1, 1, 1, 1, ],
             [1, 1, 1, 1, 1, 1, 1, 1, ],
             [1, 1, 1, 1, 1, 1, 1, 1, ],
             [1, 1, 1, 1, 1, 1, 1, 1, ]]).T
        scores1 = 0.25 * np.array([[1, 1, 1, 1, 1, 1, 1, 1, ],
                                   [1, 1, 1, 1, 1, 1, 1, 1, ],
                                   [1, 1, 1, 1, 1, 1, 1, 1, ],
                                   [1, 1, 1, 1, 1, 1, 1, 1, ]]).T
        scores = hm.__getNorm__(scores)
        assert(abs(scores-scores1).max() < 10**(-8))


def test_getNormVectOnesWithMinus():
        # testing the behaviour with neggtive values
        scores = np.array(
            [[1, 0, 0, 0, 1, 0, 0, 0, ],
             [0, 1, 0, 0, 0, 1, 0, 0, ],
             [0, 0, 1, 0, 0, 0, 1, 0, ],
             [0, 0, 0, 1, 0, 0, 0, 1, ]]).T
        scores = scores-1
        scores1 = np.array(
            [[1, 0, 0, 0, 1, 0, 0, 0, ],
             [0, 1, 0, 0, 0, 1, 0, 0, ],
             [0, 0, 1, 0, 0, 0, 1, 0, ],
             [0, 0, 0, 1, 0, 0, 0, 1, ]]).T
        scores = hm.__getNorm__(scores)
        assert(abs(scores-scores1).max() < 10**(-8))


def test_getNormVectFiveOnes():
        scores = np.array(
            [[5, 1, 1, 1, 5, 1, 1, 1, ],
             [1, 5, 1, 1, 1, 5, 1, 1, ],
             [1, 1, 5, 1, 1, 1, 5, 1, ],
             [1, 1, 1, 5, 1, 1, 1, 5, ]]).T
        scores = hm.__getNorm__(scores)
        for i in range(8):
            if i % 4 == 0:
                assert(abs(scores[i, 0]-1) < 10**(-8))
                assert(abs(scores[i, 1]-0) < 10**(-8))
                assert(abs(scores[i, 2]-0) < 10**(-8))
                assert(abs(scores[i, 3]-0) < 10**(-8))
            if i % 4 == 1:
                assert(abs(scores[i, 0]-0) < 10**(-8))
                assert(abs(scores[i, 1]-1) < 10**(-8))
                assert(abs(scores[i, 2]-0) < 10**(-8))
                assert(abs(scores[i, 3]-0) < 10**(-8))
            if i % 4 == 2:
                assert(abs(scores[i, 0]-0) < 10**(-8))
                assert(abs(scores[i, 1]-0) < 10**(-8))
                assert(abs(scores[i, 2]-1) < 10**(-8))
                assert(abs(scores[i, 3]-0) < 10**(-8))
            if i % 4 == 3:
                assert(abs(scores[i, 0]-0) < 10**(-8))
                assert(abs(scores[i, 1]-0) < 10**(-8))
                assert(abs(scores[i, 2]-0) < 10**(-8))
                assert(abs(scores[i, 3]-1) < 10**(-8))


def test_getNormVectFiveTwoOnes():
        scores = np.array(
            [[5, 1, 1, 2, 5, 1, 1, 2, ],
             [2, 5, 1, 1, 2, 5, 1, 1, ],
             [1, 2, 5, 1, 1, 2, 5, 1, ],
             [1, 1, 2, 5, 1, 1, 2, 5, ]]).T
        scores = hm.__getNorm__(scores)
        for i in range(8):
                if i % 4 == 0:
                        assert(abs(scores[i, 0]-0.8) < 10**(-8))
                        assert(abs(scores[i, 1]-0.2) < 10**(-8))
                        assert(abs(scores[i, 2]-0) < 10**(-8))
                        assert(abs(scores[i, 3]-0) < 10**(-8))
                if i % 4 == 1:
                        assert(abs(scores[i, 0]-0) < 10**(-8))
                        assert(abs(scores[i, 1]-0.8) < 10**(-8))
                        assert(abs(scores[i, 2]-0.2) < 10**(-8))
                        assert(abs(scores[i, 3]-0) < 10**(-8))
                if i % 4 == 2:
                        assert(abs(scores[i, 0]-0) < 10**(-8))
                        assert(abs(scores[i, 1]-0) < 10**(-8))
                        assert(abs(scores[i, 2]-0.8) < 10**(-8))
                        assert(abs(scores[i, 3]-0.2) < 10**(-8))
                if i % 4 == 3:
                        assert(abs(scores[i, 0]-0.2) < 10**(-8))
                        assert(abs(scores[i, 1]-0) < 10**(-8))
                        assert(abs(scores[i, 2]-0) < 10**(-8))
                        assert(abs(scores[i, 3]-0.8) < 10**(-8))


def test_getNormVectFiveTwoMinusOnes():
        scores = np.array(
            [[5, -1, -1, 2, 5, -1, -1, 2, ],
             [2, 5, -1, -1, 2, 5, -1, -1, ],
             [-1, 2, 5, -1, -1, 2, 5, -1, ],
             [-1, -1, 2, 5, -1, -1, 2, 5, ]]).T
        scores = hm.__getNorm__(scores)
        for i in range(8):
                if i % 4 == 0:
                        assert(abs(scores[i, 0]-6.0/9.0) < 10**(-8))
                        assert(abs(scores[i, 1]-3.0/9.0) < 10**(-8))
                        assert(abs(scores[i, 2]-0) < 10**(-8))
                        assert(abs(scores[i, 3]-0) < 10**(-8))
                if i % 4 == 1:
                        assert(abs(scores[i, 0]-0) < 10**(-8))
                        assert(abs(scores[i, 1]-6.0/9.0) < 10**(-8))
                        assert(abs(scores[i, 2]-3.0/9.0) < 10**(-8))
                        assert(abs(scores[i, 3]-0) < 10**(-8))
                if i % 4 == 2:
                        assert(abs(scores[i, 0]-0) < 10**(-8))
                        assert(abs(scores[i, 1]-0) < 10**(-8))
                        assert(abs(scores[i, 2]-6.0/9.0) < 10**(-8))
                        assert(abs(scores[i, 3]-3.0/9.0) < 10**(-8))
                if i % 4 == 3:
                        assert(abs(scores[i, 0]-3.0/9.0) < 10**(-8))
                        assert(abs(scores[i, 1]-0) < 10**(-8))
                        assert(abs(scores[i, 2]-0) < 10**(-8))
                        assert(abs(scores[i, 3]-6.0/9.0) < 10**(-8))
# test making of COO matrices


def test_COOtest1():
        rd.seed(1000)
        np.random.seed(1001)
        for x in range(100):
                G = nx.erdos_renyi_graph(100, 0.1, directed=True)
                A1 = hm.constructCOOtest(G)
                A2 = nx.to_numpy_array(G)
                A3 = nx.to_scipy_sparse_matrix(G)
                assert(abs(A1-A2).max() < 10**(-8))
                assert(abs(A1-A3).max() < 10**(-8))
                assert(abs(A2-A3).max() < 10**(-8))


@pytest.mark.slowtest
def test_COOtest2():
        rd.seed(1002)
        np.random.seed(1003)
        for x in range(100):
                G = nx.erdos_renyi_graph(200, 0.1, directed=True)
                A1 = hm.constructCOOtest(G)
                A2 = nx.to_numpy_array(G)
                A3 = nx.to_scipy_sparse_matrix(G)
                assert(abs(A1-A2).max() < 10**(-8))
                assert(abs(A1-A3).max() < 10**(-8))
                assert(abs(A2-A3).max() < 10**(-8))


@pytest.mark.slowtest
def test_COOtest3SelfLoop():
        rd.seed(1004)
        np.random.seed(1005)
        for x in range(100):
                G = nx.erdos_renyi_graph(200, 0.1, directed=True)
                G.add_edge(0, 0)
                G.add_edge(100, 100)
                G.add_edge(150, 150)
                A1 = hm.constructCOOtest(G)
                A2 = nx.to_numpy_array(G)
                A3 = nx.to_scipy_sparse_matrix(G)
                assert(abs(A1-A2).max() < 10**(-8))
                assert(abs(A1-A3).max() < 10**(-8))
                assert(abs(A2-A3).max() < 10**(-8))


@pytest.mark.slowtest
def test_COOtest4SelfLoopRand():
        rd.seed(1004)
        np.random.seed(1005)
        for x in range(100):
                G = nx.erdos_renyi_graph(200, 0.1, directed=True)
                for i in range(200):
                        if rd.random() < 0.1:
                                G.add_edge(i, i)
                A1 = hm.constructCOOtest(G)
                A2 = nx.to_numpy_array(G)
                A3 = nx.to_scipy_sparse_matrix(G)
                assert(abs(A1-A2).max() < 10**(-8))
                assert(abs(A1-A3).max() < 10**(-8))
                assert(abs(A2-A3).max() < 10**(-8))


# Compare approaches

def __runOneVectCycle__(G, oldVer):
        A = nx.to_numpy_array(G)
        if nx.density(G) < 0.2:
                A = sparse.csr_matrix(A)
                At = (A.T).asformat('csr')
        else:
                At = A.T
        C = [[-1,  1, -1, -1],
             [-1,  1, -1, -1],
             [-1,  1,  1,  1],
             [-1, -1, -1, -1]]
        C = np.array(C)
        Ct = C.T
        C2 = C + Ct
        q1 = G.number_of_edges()
        n = A.shape[0]
        s1 = q1/float(n*n)
        s2 = 1-q1/float(n*n)
        print('vect Zero call')
        print(oldVer)
        scores = hm.__hitsmk2__(A, At, C, Ct, oldVer, s1, C2)
        print('vect First call')
        print(scores)
        scores = hm.__getNorm__(scores)
        print('vect second call')
        print(scores)
        return scores

float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})

def __runOneCycleNoNorm__(G, p_in, p_out, c_in, c_out):
    A, mA, At, mAt = hmNV.__makeMatrices__(G)
    q1 = A.sum()

    # update each of the values
    p_inN = hmNV.updatePin_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
    c_outN = hmNV.updateCout_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
    p_outN = hmNV.updatePout_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
    c_inN = hmNV.updateCin_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
    p_inN = np.array(p_inN).T
    c_outN =np.array(c_outN).T
    p_outN =np.array(p_outN).T
    c_inN = np.array(c_inN).T

    t1 = np.concatenate([p_outN, c_inN, c_outN, p_inN], axis=1)
    return t1


def __runOneCycle__(G, p_in, p_out, c_in, c_out):
    A, mA, At, mAt = hmNV.__makeMatrices__(G)
    q1 = A.sum()

    # update each of the values
    p_inN = hmNV.updatePin_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
    c_outN = hmNV.updateCout_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
    p_outN = hmNV.updatePout_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
    c_inN = hmNV.updateCin_mk2(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
#    print('first call')
    p_inN = np.array(p_inN).T
    c_outN = np.array(c_outN).T
    p_outN = np.array(p_outN).T
    c_inN = np.array(c_inN).T
#    print(np.array([p_outN, c_inN, c_outN, p_inN]).T)
    # Take norm
    getNorm = hmNV.__getNormNonVect__
    p_inN, p_outN, c_inN, c_outN = getNorm(p_inN, p_outN, c_inN, c_outN)
#    print('second call')
    p_inN = np.array(p_inN)
    c_inN = np.array(c_inN)
    c_outN = np.array(c_outN)
    p_outN = np.array(p_outN)
# print([float(p_inN[1]), float(p_outN[1]), float(c_inN[1]), float(c_outN[1])])
    return p_inN, p_outN, c_inN, c_outN


def hitsReallySimpleNoNorm(G, pin, pot, cin, cot):
    pinN = [0 for x in range(len(G))]
    cinN = [0 for x in range(len(G))]
    potN = [0 for x in range(len(G))]
    cotN = [0 for x in range(len(G))]
    hasE = G.has_edge
    q1 = G.number_of_edges()
    s1 = q1/float(len(G)*len(G))
    s2 = 1 - s1
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

    pinN = deepcopy(np.array(pinN)).reshape([len(G), 1])
    potN = deepcopy(np.array(potN)).reshape([len(G), 1])
    cinN = deepcopy(np.array(cinN)).reshape([len(G), 1])
    cotN = deepcopy(np.array(cotN)).reshape([len(G), 1])
    t1 = np.concatenate([potN, cinN, cotN, pinN], axis=1)
    return t1




def hitsReallySimple(G, pin, pot, cin, cot):
    pinN = [0 for x in range(len(G))]
    cinN = [0 for x in range(len(G))]
    potN = [0 for x in range(len(G))]
    cotN = [0 for x in range(len(G))]
    hasE = G.has_edge
    q1 = G.number_of_edges()
    s1 = q1/float(len(G)*len(G))
    s2 = 1 - s1
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
    pinN1, potN1, cinN1, cotN1 = hmNV.__getNormNonVect__(pinN, potN, cinN, cotN)
    return [potN1, cinN1, cotN1, pinN1]


def hitsMk2SimpleVectorised(G, oldVer):
    q1 = G.number_of_edges()
    n = len(G)
    s1 = q1/float(n*n)
    s2 = 1-q1/float(n*n)
    C = [[-1,  1, -1, -1],
         [-1,  1, -1, -1],
         [-1,  1,  1,  1],
         [-1, -1, -1, -1]]
#    C = list(zip(*C))
    result = np.zeros([n, 4])
    g1 = G.has_edge
    temp1 = 0
    temp2 = 0
    temp3 = 0
    temp4 = 0
    for i in range(len(G)):
        for c1 in range(4):
            for j in range(len(G)):
                for c2 in range(4):
                    if c1 == 2 and i == 1:
                        temp1 += s2*g1(i, j)*C[c2][c1]*oldVer[j, c2]
                        temp2 += s1*(1-g1(i, j))*(-C[c2][c1])*oldVer[j, c2]
                        temp3 += s2*g1(j, i)*C[c1][c2]*oldVer[j, c2]
                        temp4 += s1*(1-g1(j, i))*(-C[c1][c2])*oldVer[j, c2]
#                        if i == 1:
#                            print([temp1,temp2,temp3,temp4])
#                        import pdb
#                        pdb.set_trace()
                    result[i, c1] += s2*g1(i, j)*C[c1][c2]*oldVer[j, c2]
                    result[i, c1] += s1*(1-g1(i, j))*(-C[c1][c2])*oldVer[j, c2]
                    result[i, c1] += s2*g1(j, i)*C[c2][c1]*oldVer[j, c2]
                    result[i, c1] += s1*(1-g1(j, i))*(-C[c2][c1])*oldVer[j, c2]
#            if i==0 and c1==1:
#                print([temp1,temp2,temp3,temp4])
    # print([temp1, temp2, temp3, temp4])
    print(result)
    result = hm.__getNorm__(result)
    return result


##########################

def hitsMk2SimpleVectorisedNoNorm(G, p_in, p_out, c_in, c_out):
    p_in1 = deepcopy(p_in).reshape([len(G), 1])
    p_out1 = deepcopy(p_out).reshape([len(G), 1])
    c_in1 = deepcopy(c_in).reshape([len(G), 1])
    c_out1 = deepcopy(c_out).reshape([len(G), 1])
    oldVer = np.concatenate([p_out1, c_in1, c_out1, p_in1], axis=1)
    q1 = G.number_of_edges()
    n = len(G)
    s1 = q1/float(n*n)
    s2 = 1-q1/float(n*n)
    C = [[-1,  1, -1, -1],
         [-1,  1, -1, -1],
         [-1,  1,  1,  1],
         [-1, -1, -1, -1]]
    result = np.zeros([n, 4])
    g1 = G.has_edge
    for i in range(len(G)):
        for c1 in range(4):
            for j in range(len(G)):
                for c2 in range(4):
                    result[i, c1] += s2*g1(i, j)*C[c1][c2]*oldVer[j, c2]
                    result[i, c1] += s1*(1-g1(i, j))*(-C[c1][c2])*oldVer[j, c2]
                    result[i, c1] += s2*g1(j, i)*C[c2][c1]*oldVer[j, c2]
                    result[i, c1] += s1*(1-g1(j, i))*(-C[c2][c1])*oldVer[j, c2]
    return result



#########################



def __getSimOldApp__(G,p_in,p_out,c_in,c_out):
    p_in1 = deepcopy(p_in).reshape([len(G), 1])
    p_out1 = deepcopy(p_out).reshape([len(G), 1])
    c_in1 = deepcopy(c_in).reshape([len(G), 1])
    c_out1 = deepcopy(c_out).reshape([len(G), 1])
    p_inN, p_outN, c_inN, c_outN = __runOneCycle__(G, p_in1, p_out1, c_in1, c_out1)
    p_inN = deepcopy(p_inN).reshape([len(G), 1])
    p_outN = deepcopy(p_outN).reshape([len(G), 1])
    c_inN = deepcopy(c_inN).reshape([len(G), 1])
    c_outN = deepcopy(c_outN).reshape([len(G), 1])

    # we are going to run multiple different approaches to compute this
    # and make sure that they all agree
    simOldApp = np.concatenate([p_outN, c_inN, c_outN, p_inN], axis=1)
    return simOldApp

def __getVectApp__(G,p_in,p_out,c_in,c_out):
    p_in1 = deepcopy(p_in).reshape([len(G), 1])
    p_out1 = deepcopy(p_out).reshape([len(G), 1])
    c_in1 = deepcopy(c_in).reshape([len(G), 1])
    c_out1 = deepcopy(c_out).reshape([len(G), 1])
    t1 = np.concatenate([p_out1, c_in1, c_out1, p_in1], axis=1)
    return __runOneVectCycle__(G, t1)


def __simple__(G,p_in,p_out,c_in,c_out):
    p_in1 = deepcopy(p_in).reshape([len(G), 1])
    p_out1 = deepcopy(p_out).reshape([len(G), 1])
    c_in1 = deepcopy(c_in).reshape([len(G), 1])
    c_out1 = deepcopy(c_out).reshape([len(G), 1])
    t4 = np.concatenate([p_out1, c_in1, c_out1, p_in1], axis=1)
    return hitsMk2SimpleVectorised(G, t4)

def __reallySimple__(G,p_in,p_out,c_in,c_out):
    p_in1 = deepcopy(p_in).reshape([len(G), 1])
    p_out1 = deepcopy(p_out).reshape([len(G), 1])
    c_in1 = deepcopy(c_in).reshape([len(G), 1])
    c_out1 = deepcopy(c_out).reshape([len(G), 1])
    reallySimple = np.array(hitsReallySimple(G, p_in1, p_out1, c_in1, c_out1)).T
    return reallySimple


def __actualSlowApp__(G,p_in,p_out,c_in,c_out):
    A, mA, At, mAt = hmNV.__makeMatrices__(G)
    q1 = G.number_of_edges()
    advCyc = hmNV.__runOneAdvancedHitsCycle__
    pIn, pOut, cIn, cOut = advCyc(p_in, p_out, c_in, c_out, A, mA, At, mAt, q1)
    p_inNa = deepcopy(pIn).reshape([len(G), 1])
    p_outNa = deepcopy(pOut).reshape([len(G), 1])
    c_inNa = deepcopy(cIn).reshape([len(G), 1])
    c_outNa = deepcopy(cOut).reshape([len(G), 1])
    actualSlowApp = np.concatenate([p_outNa, c_inNa, c_outNa, p_inNa], axis=1)
    return actualSlowApp


def __compareVectToNonVector__(G, p_in, p_out, c_in, c_out):
    p_in1 = np.array(deepcopy(p_in)).reshape([len(G), 1])
    p_out1 = np.array(deepcopy(p_out)).reshape([len(G), 1])
    c_in1 = np.array(deepcopy(c_in)).reshape([len(G), 1])
    c_out1 = np.array(deepcopy(c_out)).reshape([len(G), 1])

    simOldApp = __getSimOldApp__(G, p_in, p_out, c_in, c_out)

    vectApp = __getVectApp__(G, p_in1, p_out1, c_in1, c_out1)

    simple = __simple__(G, p_in1, p_out1, c_in1, c_out1)

    reallySimple = __reallySimple__(G, p_in, p_out, c_in, c_out)

    actualSlowApp = __actualSlowApp__(G,p_in,p_out,c_in,c_out)

    print(abs(simOldApp-reallySimple).max() < 10**(-8))
    print(abs(reallySimple-simple).max() < 10**(-8))
    print(abs(vectApp-reallySimple).max() < 10**(-8))
    print(abs(simOldApp-vectApp).max() < 10**(-8))
    print(abs(simOldApp-simple).max() < 10**(-8))
    print(abs(vectApp-simple).max() < 10**(-8))
    print(abs(actualSlowApp-reallySimple).max() < 10**(-8))
    print(abs(actualSlowApp-vectApp).max() < 10**(-8))
    print(abs(actualSlowApp-simple).max() < 10**(-8))


    assert(abs(simOldApp-reallySimple).max() < 10**(-8))
    assert(abs(reallySimple-simple).max() < 10**(-8))
    assert(abs(vectApp-reallySimple).max() < 10**(-8))
    assert(abs(simOldApp-vectApp).max() < 10**(-8))
    assert(abs(simOldApp-simple).max() < 10**(-8))
    assert(abs(vectApp-simple).max() < 10**(-8))

    assert(abs(actualSlowApp-reallySimple).max() < 10**(-8))
    assert(abs(actualSlowApp-vectApp).max() < 10**(-8))
    assert(abs(actualSlowApp-simple).max() < 10**(-8))


def test_EmptyGraphOnlyOnePinVect():
        G = nx.DiGraph()
        G.add_nodes_from(range(20))
        p_in = np.array([1, ]+[0 for x in range(19)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_EmptyGraphOnlyOneCinVect():
        G = nx.DiGraph()
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([1, ]+[0 for x in range(19)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_EmptyGraphOnlyOnePoutVect():
        G = nx.DiGraph()
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([1, ]+[0 for x in range(19)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_EmptyGraphOnlyOneCoutVect():
        G = nx.DiGraph()
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([1, ]+[0 for x in range(19)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_EmptyGraphOnlyPinVect():
        G = nx.DiGraph()
        G.add_nodes_from(range(20))
        p_in = np.array([1, ]+[1 for x in range(19)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_EmptyGraphOnlyCinVect():
        G = nx.DiGraph()
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([1, ]+[1 for x in range(19)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_EmptyGraphOnlyPoutVect():
        G = nx.DiGraph()
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([1, ]+[1 for x in range(19)])
        c_out = np.array([0 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_EmptyGraphOnlyCoutVect():
        G = nx.DiGraph()
        G.add_nodes_from(range(20))
        p_in = np.array([0 for x in range(20)])
        c_in = np.array([0 for x in range(20)])
        p_out = np.array([0 for x in range(20)])
        c_out = np.array([1, ]+[1 for x in range(19)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_EmptyGraphConstVect():
        G = nx.DiGraph()
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([1 for x in range(20)])
        p_out = np.array([1 for x in range(20)])
        c_out = np.array([1 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_EmptyGraphVariable():
        G = nx.DiGraph()
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([x for x in range(20)])
        p_out = np.array([x**2 for x in range(20)])
        c_out = np.array([(x-10)**3 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_EmptyGraphVariable1():
        G = nx.DiGraph()
        G.add_nodes_from(range(20))
        p_in = np.array([1 for x in range(20)])
        c_in = np.array([abs(x-10) for x in range(20)])
        p_out = np.array([x**2-x for x in range(20)])
        c_out = np.array([(x-10)**3 for x in range(20)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOnePinVectSelfLoop():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        p_in = np.array([1, ]+[0 for x in range(1)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOneCinVectSelfLoop():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([1, ]+[0 for x in range(1)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnlyOnePoutVectSelfLoop():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([1, ]+[0 for x in range(1)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)

###################################




def test_TwoNodeOneEdgeOnePinVectSelfLoop():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        p_in = np.array([1, ]+[1 for x in range(1)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOneCinVectSelfLoop():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([1, ]+[1 for x in range(1)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnePoutVectSelfLoop():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([1, ]+[1 for x in range(1)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOneCoutVectSelfLoop():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([1, ]+[1 for x in range(1)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


####################################


###################################


def test_TwoNodeOneEdgeOnePinVectSelfLoop2():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        G.add_edge(1, 1)
        p_in = np.array([1, ]+[1 for x in range(1)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOneCinVectSelfLoop2():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        G.add_edge(1, 1)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([1, ]+[1 for x in range(1)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOnePoutVectSelfLoop2():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        G.add_edge(1, 1)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([1, ]+[1 for x in range(1)])
        c_out = np.array([0 for x in range(2)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdgeOneCoutVectSelfLoop2():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        G.add_edge(1, 1)
        p_in = np.array([0 for x in range(2)])
        c_in = np.array([0 for x in range(2)])
        p_out = np.array([0 for x in range(2)])
        c_out = np.array([1, ]+[1 for x in range(1)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


####################################


###################################


def test_TwoNodeOneEdge_halfs_SelfLoop2():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        G.add_edge(1, 1)
        p_in = np.array([0.5,0.5 ])
        c_in = np.array([-0.5,0.5 ])
        p_out = np.array([0.5,-0.5 ])
        c_out = np.array([-0.5,-0.5 ])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdge_half_SelfLoop2_t1():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        G.add_edge(1, 1)
        p_in = np.array([-0.5,0.5 ])
        c_in = np.array([0.5,0.5 ])
        p_out = np.array([-0.5,-0.5 ])
        c_out = np.array([0.5,0.5 ])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)


def test_TwoNodeOneEdge_half_SelfLoop2():
        G = nx.DiGraph()
        G.add_edge(0, 1)
        G.add_edge(0, 0)
        G.add_edge(1, 1)
        p_in = np.array([-0.5, -0.5])
        c_in = np.array([0.5, -0.5])
        p_out = np.array([0.5, 0.5])
        c_out = np.array([-0.5, 0.5])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)



####################################


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
        G.add_edge(3,2)
        p_in = np.array([1 for x in range(5)])
        c_in = np.array([1 for x in range(5)])
        p_out = np.array([1 for x in range(5)])
        c_out = np.array([1 for x in range(5)])
        __compareVectToNonVector__(G, p_in, p_out, c_in, c_out)

def test_BadCase():
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
