# We fill test this code by constructing graphs with a known low rank
# approximation and then testing if we recover it

# TODO add the non symmetric case


import pytest
import numpy as np
import networkx as nx

# this will need to be fixed before this is a package
import sys
sys.path.append('../')

import lowRankApproximation as lra

# Test without the graph


def __makeIndeHelper__(v1, vs):
        v1 /= sqrt((v1.T@v1))
        for i in range(len(vs)):
                v = vs[i]
                v1 = v1-float(v1.T@v)*v
                v1 /= sqrt((v1.T@v1))
        return v1


def __makeRandomVector__(n):
        v1 = np.random.random([n, 1])
        return v1


def test_basicNoGraphTest():
    np.random.seed(0)
    for n in [5, 15, 50, 100,400]:
        v1 = __makeRandomVector__(n)
        v1 = __makeIndeHelper__(v1, [])

        v2 = __makeRandomVector__(n)
        v2 = __makeIndeHelper__(v2, [v1])

        v3 = __makeRandomVector__(n)
        v3 = __makeIndeHelper__(v3, [v1, v2])

        v4 = __makeRandomVector__(n)
        v4 = __makeIndeHelper__(v4, [v1, v2, v3])

        u1 = __makeRandomVector__(n)
        u1 = __makeIndeHelper__(u1, [])

        u2 = __makeRandomVector__(n)
        u2 = __makeIndeHelper__(u2, [u1])

        u3 = __makeRandomVector__(n)
        u3 = __makeIndeHelper__(u3, [u1, u2])

        u4 = __makeRandomVector__(n)
        u4 = __makeIndeHelper__(u4, [u1, u2, u3])

        for d1 in [1, 5, 10, 34.5]:
            for d2 in [1, 5, 10, 45.5]:
                for d3 in [1, 5, 10, 12.4]:
                    for d4 in [1, 5, 10, 100.9]:
                        A = 0
                        ordering = []
                        ordering.append([d1, [u1, v1]])
                        ordering.append([d2, [u2, v2]])
                        ordering.append([d3, [u3, v3]])
                        ordering.append([d4, [u4, v4]])
                        ordering.sort(
                            key=lambda x: -x[0])
                        A += d1*np.outer(u1, v1)
                        A += d2*np.outer(u2, v2)
                        A += d3*np.outer(u3, v3)
                        A += d4*np.outer(u4, v4)
                        for i in range(1, 4):
                            A2 = 0
                            singularValues = []
                            for j in range(i):
                                    o1 = ordering[j]
#                                A2+=o1[0]*np.outer(o1[1][0],o1[1][1])
                                    A2 += o1[0]*o1[1][0]@o1[1][1].T
                            for j in range(i+1):
                                    o1 = ordering[j]
                                    singularValues.append(o1[0])
                            if (len(set(singularValues)) != i+1):
                                with pytest.raises(lra.EqualSingularValues):
                                        A1 = lra.__lowRankHelper__(
                                                A, i, strict=True)
                            else:
                                A1 = lra.__lowRankHelper__(
                                        A, i, strict=True)
                                assert(np.abs(A1-A2).max() < 10**(-10))


# Test with the graph

# Helper function
def __symmetricSimpleHelper__(s1, s2):
        l1 = [1.0, ]*s1+[0.0, ]*s2
        l2 = [0.0, ]*s1+[1.0, ]*s2
        l1 = np.array(l1)
        l2 = np.array(l2)
        m1 = np.outer(l1, l1)
        m2 = np.outer(l2, l2)
        m = m1+m2
        i1, i2 = np.nonzero(m)
        G = nx.Graph()
        G.add_edges_from(zip(i1, i2))
        result = lra.lowRankApproximation(G, 2)
        assert((np.abs(result-m) < 10**(-12)).all())


# Important the analytical form was guessed from numerics need to go back
# to prove it.
from math import sqrt


def __symmetricHelper__(s1, s2, s3):
    real = np.zeros([s1+s2+s3, s1+s2+s3])
    real[:s1+s2, :s1+s2] = 1
    real[s1:s1+s2, s1:s1+s2] = 0
    real[s1+s2:, s1+s2:] = 1
    eigs1 = np.linalg.eigh(real)
    assert(abs(eigs1[0][0]) > 0.01)
    assert(abs(eigs1[0][-2]) > 0.01)
    assert(abs(eigs1[0][-1]) > 0.01)
    assert(max(abs(eigs1[0][2:-2]) < 0.0001))
    l1 = eigs1[1][:, 0]
    l2 = eigs1[1][:, -1]
    l3 = eigs1[1][:, -2]
    v1 = eigs1[0][0]
    v2 = eigs1[0][-1]
    v3 = eigs1[0][-2]
    m1 = np.outer(l1, l1)
    m2 = np.outer(l2, l2)
    m3 = np.outer(l3, l3)
    m = v1*m1+v2*m2+v3*m3
    m[np.abs(m) < 10**(-10)] = 0
    i1, i2 = np.nonzero(m)
    G = nx.Graph()
    G.add_edges_from(zip(i1, i2))
    result = lra.lowRankApproximation(G, 2)
    av1 = abs(v1)
    av2 = abs(v2)
    av3 = abs(v3)
    if av3 < min(av1, av2):
        print('s3 smallest')
        actual = v1*m1+v2*m2
    if av2 < min(av1, av3):
        print('s2 smallest')
        actual = v1*m1+v3*m3
    if av1 < min(av2, av3):
        print('s1 smallest')
        actual = v2*m2+v3*m3
    assert((np.abs(result-actual) < 10**(-12)).all())


def test_symmetric():
    __symmetricHelper__(2, 2, 4)


def test_symmetricEqual():
    __symmetricHelper__(2, 2, 2)


def test_symmetricS1Big():
    __symmetricHelper__(200, 10, 10)


def test_symmetricS2Big():
    __symmetricHelper__(10, 200, 10)


def test_symmetricS3Big():
    __symmetricHelper__(10, 10, 200)


def test_symmetricDiffSizes():
    for a in [2, 10, 50]:
        for b in [2, 10, 50]:
            for c in [2, 10, 50]:
                __symmetricHelper__(a, b, c)


def test_symmetricSimple():
    __symmetricSimpleHelper__(100, 100)

# maybe make bigger?

def test_symmetricSimpleLarge():
        __symmetricSimpleHelper__(500, 500)


def test_symmetricSimpleUneven():
        __symmetricSimpleHelper__(10, 500)
