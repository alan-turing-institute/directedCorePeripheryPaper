import numpy as np
import random as rd
import networkx as nx
import pytest
# this will need to be fixed before this is a package
import sys
sys.path.append('../')
import scoring as sc
import graphGeneration as gg

#### Helper functions ####

def __penalityHelperSlow__(G, Cin_score, Cout_score):
        n = len(G)
        Pin_score = np.zeros(Cin_score.shape)
        Pout_score = np.zeros(Cin_score.shape)
        for i in range(n):
                for j in range(n):
                        if i != j:
                                Pin_score[i] -= G.has_edge(i, j)*Cin_score[j]
                                Pin_score[i] -= G.has_edge(i, j)*Cout_score[j]
                                Pin_score[i] -= G.has_edge(j, i)*Cin_score[j]
                                Pin_score[i] += G.has_edge(j, i)*Cout_score[j]
                                Pout_score[i] += G.has_edge(i, j)*Cin_score[j]
                                Pout_score[i] -= G.has_edge(i, j)*Cout_score[j]
                                Pout_score[i] -= G.has_edge(j, i)*Cin_score[j]
                                Pout_score[i] -= G.has_edge(j, i)*Cout_score[j]
        t1 = np.concatenate(
            [Cin_score, Cout_score, Pin_score, Pout_score],
            axis=1)
        return t1


def __penalityHelperSlowFullGraph__(G):
        Cin_score = np.array([G.in_degree(x) for x in range(len(G))])
        Cout_score = np.array([G.out_degree(x) for x in range(len(G))])
        Cin_score = Cin_score.reshape([len(G), 1])
        Cout_score = Cout_score.reshape([len(G), 1])
        return __penalityHelperSlow__(G, Cin_score, Cout_score)


def __reverseHelperSlow__(G, Cin_score, Cout_score):
        CinMax = 0
        for i in range(max(Cin_score.shape)):
                if CinMax < Cin_score[i]:
                        CinMax = Cin_score[i]
        CoutMax = 0
        for i in range(max(Cout_score.shape)):
                if CoutMax < Cout_score[i]:
                        CoutMax = Cout_score[i]
        Pout_score = CoutMax-Cout_score
        Pin_score = CinMax-Cin_score
        t1 = np.concatenate(
            [Cin_score, Cout_score, Pin_score, Pout_score],
            axis=1)
        return t1


def __reverseHelperSlowFullGraph__(G):
        Cin_score = np.array([G.in_degree(x) for x in range(len(G))])
        Cout_score = np.array([G.out_degree(x) for x in range(len(G))])
        Cin_score = Cin_score.reshape([len(G), 1])
        Cout_score = Cout_score.reshape([len(G), 1])
        return __reverseHelperSlow__(G, Cin_score, Cout_score)

# Test Penality Helper


def test_penalityHelper_EmptyGraph():
        A = np.zeros([100, 100])
        Cin_score = np.ones([100, 1])
        Cout_score = np.ones([100, 1])*2
        t1 = sc.__penalityHelper__(A, Cin_score, Cout_score)
        assert(t1.shape[0] == 100)
        assert(t1.shape[1] == 4)
        for i in range(100):
                assert(t1[i, 0] == Cin_score[i])
                assert(t1[i, 1] == Cout_score[i])
                assert(t1[i, 2] == 0)
                assert(t1[i, 3] == 0)


def __testHelper__(G, A, Cin_score, Cout_score):
        t1 = sc.__penalityHelper__(A, Cin_score, Cout_score)
        assert(t1.shape[0] == 100)
        assert(t1.shape[1] == 4)
        t2 = __penalityHelperSlow__(G, Cin_score, Cout_score)
        for i in range(100):
                for j in range(4):
                        assert(abs(t1[i, j] - t2[i, j]) < 10**(-8))


def test_penalityHelper_OneEdge():
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        A = nx.to_numpy_array(G)
        Cin_score = np.ones([100, 1])
        Cout_score = np.ones([100, 1])*2
        __testHelper__(G, A, Cin_score, Cout_score)


def test_penalityHelper_TwoEdge():
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        A = nx.to_numpy_array(G)
        Cin_score = np.ones([100, 1])
        Cout_score = np.ones([100, 1])*2
        __testHelper__(G, A, Cin_score, Cout_score)


def test_penalityHelper_RandomGraph():
        rd.seed(51)
        np.random.seed(52)
        G = nx.erdos_renyi_graph(100, 0.1, directed=True)
        A = nx.to_numpy_array(G)
        Cin_score = np.ones([100, 1])
        Cout_score = np.ones([100, 1])*2
        __testHelper__(G, A, Cin_score, Cout_score)


def test_penalityHelper_RandomGraphSelfloop():
        rd.seed(53)
        np.random.seed(54)
        G = nx.erdos_renyi_graph(100, 0.1, directed=True)
        for i in range(100):
                if rd.random() < 0.1:
                        G.add_edge(i, i)
        A = nx.to_numpy_array(G)
        Cin_score = np.ones([100, 1])
        Cout_score = np.ones([100, 1])*2
        __testHelper__(G, A, Cin_score, Cout_score)


def test_penalityHelper_OneEdgeRandom():
        rd.seed(55)
        np.random.seed(56)
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        A = nx.to_numpy_array(G)
        Cin_score = np.random.random([100, 1])
        Cout_score = np.random.random([100, 1])
        __testHelper__(G, A, Cin_score, Cout_score)


def test_penalityHelper_TwoEdgeRandom():
        rd.seed(57)
        np.random.seed(58)
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        A = nx.to_numpy_array(G)
        Cin_score = np.random.random([100, 1])
        Cout_score = np.random.random([100, 1])
        __testHelper__(G, A, Cin_score, Cout_score)


def test_penalityHelper_RandomGraphEdgeRandom():
        rd.seed(59)
        np.random.seed(60)
        G = nx.erdos_renyi_graph(100, 0.1, directed=True)
        A = nx.to_numpy_array(G)
        Cin_score = np.random.random([100, 1])
        Cout_score = np.random.random([100, 1])
        __testHelper__(G, A, Cin_score, Cout_score)


def test_penalityHelper_RandomGraphEdgeRandomSelfLoop():
        rd.seed(71)
        np.random.seed(72)
        G = nx.erdos_renyi_graph(100, 0.1, directed=True)
        for i in range(100):
                if rd.random() < 0.1:
                        G.add_edge(i, i)
        A = nx.to_numpy_array(G)
        Cin_score = np.random.random([100, 1])
        Cout_score = np.random.random([100, 1])
        __testHelper__(G, A, Cin_score, Cout_score)

#  PenalityHelperWrap test


def test_penalityHelperWrap_EmptyGraph():
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        A = nx.to_numpy_array(G)
        t1 = sc.__penalityHelperWrap__(A)
        assert(t1.shape[0] == 100)
        assert(t1.shape[1] == 4)
        for i in range(100):
                assert(t1[i, 0] == 0)
                assert(t1[i, 1] == 0)
                assert(t1[i, 2] == 0)
                assert(t1[i, 3] == 0)


def __testHelperWrap__(G):
        A = nx.to_numpy_array(G)
        t1 = sc.__penalityHelperWrap__(A)
        assert(t1.shape[0] == 100)
        assert(t1.shape[1] == 4)
        t2 = __penalityHelperSlowFullGraph__(G)
        for i in range(100):
                for j in range(4):
                        assert(abs(t1[i, j] - t2[i, j]) < 10**(-8))


def test_penalityWrapHelper_OneEdge():
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        __testHelperWrap__(G)


def test_penalityWrapHelper_TwoEdge():
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        __testHelperWrap__(G)


def test_penalityWrapHelper_RandomGraph():
        rd.seed(73)
        np.random.seed(74)
        for rep in range(10):
                G = nx.erdos_renyi_graph(100, 0.1, directed=True)
                __testHelperWrap__(G)


def test_penalityWrapHelper_RandomGraphSelfloop():
        rd.seed(75)
        np.random.seed(76)
        for rep in range(10):
                G = nx.erdos_renyi_graph(100, 0.1, directed=True)
                for i in range(100):
                        if rd.random() < 0.1:
                                G.add_edge(i, i)
                __testHelperWrap__(G)


# Reverse Helper Tests


def test_reverseHelper_EmptyGraph():
        A = np.zeros([100, 100])
        Cin_score = np.ones([100, 1])
        Cout_score = np.ones([100, 1])*2
        __testHelperReverse__([], A, Cin_score, Cout_score)


def __testHelperReverse__(G, A, Cin_score, Cout_score):
        t1 = sc.__reverseFormulation__(Cin_score, Cout_score)
        assert(t1.shape[0] == 100)
        assert(t1.shape[1] == 4)
        t2 = __reverseHelperSlow__(G, Cin_score, Cout_score)
        for i in range(100):
                for j in range(4):
                        assert(abs(t1[i, j] - t2[i, j]) < 10**(-8))


def test_reverseHelper_OneEdge():
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        A = nx.to_numpy_array(G)
        Cin_score = np.ones([100, 1])
        Cout_score = np.ones([100, 1])*2
        __testHelperReverse__(G, A, Cin_score, Cout_score)


def test_reverseHelper_TwoEdge():
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        A = nx.to_numpy_array(G)
        Cin_score = np.ones([100, 1])
        Cout_score = np.ones([100, 1])*2
        __testHelperReverse__(G, A, Cin_score, Cout_score)


def test_reverseHelper_RandomGraph():
        rd.seed(77)
        np.random.seed(78)
        G = nx.erdos_renyi_graph(100, 0.1, directed=True)
        A = nx.to_numpy_array(G)
        Cin_score = np.ones([100, 1])
        Cout_score = np.ones([100, 1])*2
        __testHelperReverse__(G, A, Cin_score, Cout_score)


def test_reverseHelper_RandomGraphSelfLoop():
        rd.seed(79)
        np.random.seed(80)
        G = nx.erdos_renyi_graph(100, 0.1, directed=True)
        A = nx.to_numpy_array(G)
        for i in range(100):
                if rd.random() < 0.1:
                        G.add_edge(i, i)
        A = nx.to_numpy_array(G)
        Cin_score = np.ones([100, 1])
        Cout_score = np.ones([100, 1])*2
        __testHelperReverse__(G, A, Cin_score, Cout_score)


def test_reverseHelper_OneEdgeRandom():
        rd.seed(81)
        np.random.seed(82)
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        A = nx.to_numpy_array(G)
        Cin_score = np.random.random([100, 1])
        Cout_score = np.random.random([100, 1])
        __testHelperReverse__(G, A, Cin_score, Cout_score)


def test_reverseHelper_TwoEdgeRandom():
        rd.seed(83)
        np.random.seed(84)
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        A = nx.to_numpy_array(G)
        Cin_score = np.random.random([100, 1])
        Cout_score = np.random.random([100, 1])
        __testHelperReverse__(G, A, Cin_score, Cout_score)


def test_reverseHelper_RandomGraphEdgeRandom():
        rd.seed(85)
        np.random.seed(86)
        G = nx.erdos_renyi_graph(100, 0.1, directed=True)
        A = nx.to_numpy_array(G)
        Cin_score = np.random.random([100, 1])
        Cout_score = np.random.random([100, 1])
        __testHelperReverse__(G, A, Cin_score, Cout_score)


def test_reverseHelper_RandomGraphEdgeRandomSelfLoop():
        rd.seed(87)
        np.random.seed(88)
        G = nx.erdos_renyi_graph(100, 0.1, directed=True)
        for i in range(100):
                if rd.random() < 0.1:
                        G.add_edge(i, i)
        A = nx.to_numpy_array(G)
        Cin_score = np.random.random([100, 1])
        Cout_score = np.random.random([100, 1])
        __testHelperReverse__(G, A, Cin_score, Cout_score)


def test__convertScoresToClassesAllUniform():
        rd.seed(187)
        np.random.seed(188)
        G = nx.erdos_renyi_graph(100,0.5)
        result = [[0, ]*4 for x in range(100)]
        t1 = sc.__convertScoresToClasses__(result,G)
        assert(len(t1) == 100)
        for i in range(100):
                assert(t1[i] == 0)
        G = nx.erdos_renyi_graph(200,0.5)
        result = [[0, ]*4, ]*200
        t1 = sc.__convertScoresToClasses__(result,G)
        assert(len(t1) == 200)
        for i in range(200):
                assert(t1[i] == 0)
        result = [[1, ]*4 for x in range(100)]
        G = nx.erdos_renyi_graph(100,0.5)
        t1 = sc.__convertScoresToClasses__(result,G)
        assert(len(t1) == 100)
        for i in range(100):
                assert(t1[i] == 0)
        G = nx.erdos_renyi_graph(200,0.5)
        result = [[2, ]*4 for x in range(200)]
        t1 = sc.__convertScoresToClasses__(result,G)
        assert(len(t1) == 200)
        for i in range(200):
                assert(t1[i] == 0)


def test_convertScoresToClassesAllOne():
        rd.seed(287)
        np.random.seed(288)
        result = [[0, ]*4 for x in range(100)]
        for i in range(100):
                result[i][i % 4] = 1
        G = nx.erdos_renyi_graph(100,0.5)
        t1 = sc.__convertScoresToClasses__(result,G)
        assert(len(t1) == 100)
        groups = [[], [], [], []]
        for i in range(100):
                groups[t1[i]].append(i)
        for j in range(4):
                temp1 = []
                temp3 = []
                for k in range(4):
                        temp2 = ([i*4+k for i in range(25)] == groups[j])
                        temp1.append(temp2)
                        temp2 = (groups[k] == groups[j])
                        temp3.append(temp2)
                assert(sum(temp1) == 1)
                assert(sum(temp3) == 1)
        result = [[0, ]*4 for x in range(200)]
        for i in range(200):
                result[i][i % 4] = 1
        G = nx.erdos_renyi_graph(200,0.5)
        t1 = sc.__convertScoresToClasses__(result,G)
        assert(len(t1) == 200)
        groups = [[], [], [], []]
        for i in range(100):
                groups[t1[i]].append(i)
        for j in range(4):
                temp1 = []
                temp3 = []
                for k in range(4):
                        temp2 = ([i*4+k for i in range(25)] == groups[j])
                        temp1.append(temp2)
                        temp2 = (groups[k] == groups[j])
                        temp3.append(temp2)
                assert(sum(temp1) == 1)
                assert(sum(temp3) == 1)


def test_convertScoresToClassesDiffAmount():
        result = [[0, ]*4 for x in range(100)]
        for i in range(100):
                result[i][i % 4] = 0.9
                result[i][(i+1) % 4] = 0.1

        G = nx.erdos_renyi_graph(100, 0.5, directed=True)
        t1 = sc.__convertScoresToClasses__(result,G)
        assert(len(t1) == 100)
        groups = [[], [], [], []]
        for i in range(100):
                groups[t1[i]].append(i)
        for j in range(4):
                temp1 = []
                temp3 = []
                for k in range(4):
                        temp2 = ([i*4+k for i in range(25)] == groups[j])
                        temp1.append(temp2)
                        temp2 = (groups[k] == groups[j])
                        temp3.append(temp2)
                assert(sum(temp1) == 1)
                assert(sum(temp3) == 1)
        result = [[0, ]*4 for x in range(200)]
        for i in range(100):
                result[i][i % 4] = 0.9
                result[i][(i+1) % 4] = 0.1
        G = nx.erdos_renyi_graph(200, 0.5, directed=True)
        t1 = sc.__convertScoresToClasses__(result,G)
        assert(len(t1) == 200)
        groups = [[], [], [], []]
        for i in range(100):
                groups[t1[i]].append(i)
        for j in range(4):
                temp1 = []
                temp3 = []
                for k in range(4):
                        temp2 = ([i*4+k for i in range(25)] == groups[j])
                        temp1.append(temp2)
                        temp2 = (groups[k] == groups[j])
                        temp3.append(temp2)
                assert(sum(temp1) == 1)
                assert(sum(temp3) == 1)


def test_convertScoresToClassesMixture():
        rd.seed(89)
        np.random.seed(90)
        result = [[0, ]*4 for x in range(100)]
        for i in range(100):
                if i % 4 == 0:
                        result[i] = [0.25, 0.25, 0.25, 0.25]
                if i % 4 == 1:
                        result[i] = [-0.25, 0.25, 0.25, 0.25]
                if i % 4 == 2:
                        result[i] = [0.25, -0.25, 0.25, 0.25]
                if i % 4 == 3:
                        result[i] = [0.25, 0.25, -0.25, 0.25]
        for i in range(100):
                for j in range(4):
                        result[i][j] += (0.5-rd.random())/8.0
        G = nx.erdos_renyi_graph(100, 0.5, directed=True)
        t1 = sc.__convertScoresToClasses__(result,G)
        assert(len(t1) == 100)
        groups = [[], [], [], []]
        for i in range(100):
                groups[t1[i]].append(i)
        for j in range(4):
                temp1 = []
                temp3 = []
                for k in range(4):
                        temp2 = ([i*4+k for i in range(25)] == groups[j])
                        temp1.append(temp2)
                        temp2 = (groups[k] == groups[j])
                        temp3.append(temp2)
                assert(sum(temp1) == 1)
                assert(sum(temp3) == 1)


def test_normHelperTestArray():
        rd.seed(91)
        np.random.seed(92)
        t1 = np.random.random([100, 4])
        t2 = sc.__normHelper__(t1)
        assert(isinstance(t2, np.ndarray))


def test_normHelperTestArraySquareSum():
        rd.seed(93)
        np.random.seed(94)
        for rep in range(10):
                t1 = np.random.random([100, 4])
                t2 = sc.__normHelper__(t1)
                for i in range(100):
                        temp1 = 0
                        for j in range(4):
                                temp1 += t2[i, j]*t2[i, j]
                        assert(abs(temp1-1) < 10**(-8))


def test_normHelperTestArrayRatio():
        rd.seed(95)
        np.random.seed(96)
        for rep in range(10):
                t1 = np.random.random([100, 4])
                t2 = sc.__normHelper__(t1)
                for i in range(100):
                        temp1 = 0
                        for j in range(4):
                                for k in range(4):
                                        temp1 = t1[i, j]/t1[i, k]
                                        temp2 = t2[i, j]/t2[i, k]
                                        assert(abs(temp1-temp2) < 10**(-10))


# Reverse Helper Wrapper Tests


def test_reverseHelperWrap_EmptyGraph():
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        __testHelperWrapReverse__(G)


def __testHelperWrapReverse__(G):
        A = nx.to_numpy_array(G)
        t1 = sc.__reverseFormulationWrap__(A)
        assert(t1.shape[0] == 100)
        assert(t1.shape[1] == 4)
        t2 = __reverseHelperSlowFullGraph__(G)
        for i in range(100):
                for j in range(4):
                        assert(abs(t1[i, j] - t2[i, j]) < 10**(-8))


def test_reverseHelperWrap_OneEdge():
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        __testHelperWrapReverse__(G)


def test_reverseHelperWrap_TwoEdge():
        G = nx.DiGraph()
        G.add_nodes_from(range(100))
        G.add_edge(0, 1)
        G.add_edge(1, 2)
        __testHelperWrapReverse__(G)


def test_reverseHelperWrap_RandomGraph():
        rd.seed(97)
        np.random.seed(98)
        G = nx.erdos_renyi_graph(100, 0.1, directed=True)
        __testHelperWrapReverse__(G)


def test_reverseHelperWrap_RandomGraphEdgeRandom():
        rd.seed(99)
        np.random.seed(100)
        G = nx.erdos_renyi_graph(100, 0.1, directed=True)
        __testHelperWrapReverse__(G)


def test_reverseHelperWrap_RandomGraphEdgeRandomSelfLoop():
        rd.seed(101)
        np.random.seed(102)
        G = nx.erdos_renyi_graph(100, 0.1, directed=True)
        for i in range(100):
                if rd.random() < 0.1:
                        G.add_edge(i, i)
        __testHelperWrapReverse__(G)


## test reorder
import itertools as it
def test_reorder1():
    rd.seed(103)
    np.random.seed(104)
    G = gg.syntheticModel1(100,0)
    result = sorted([x%4 for x in range(100)])
    result = list(reversed(result))
    arr0=[0,]*25
    arr1=[1,]*25
    arr2=[2,]*25
    arr3=[3,]*25
    for item in it.permutations([arr0,arr1,arr2,arr3]):
        temp1 = sum(item,[])
        result1 = sc.reorderConvertGroups(G,temp1)
        for i in range(25):
            assert(result1[i]==0)
        for i in range(25,50):
            assert(result1[i]==1)
        for i in range(50,75):
            assert(result1[i]==2)
        for i in range(75,100):
            assert(result1[i]==3)

def test_reorder2():
    rd.seed(105)
    np.random.seed(106)
    G = gg.syntheticModel1(1000,0.01)
    result = sorted([x%4 for x in range(1000)])
    result = list(reversed(result))
    arr0=[0,]*250
    arr1=[1,]*250
    arr2=[2,]*250
    arr3=[3,]*250
    for item in it.permutations([arr0,arr1,arr2,arr3]):
        temp1 = sum(item,[])
        result1 = sc.reorderConvertGroups(G,temp1)
        for i in range(250):
            assert(result1[i]==0)
        for i in range(250,500):
            assert(result1[i]==1)
        for i in range(500,750):
            assert(result1[i]==2)
        for i in range(750,1000):
            assert(result1[i]==3)

def test_reorder3():
    # confirm that it stays in the same group if all equal
    rd.seed(107)
    np.random.seed(108)
    G = nx.erdos_renyi_graph(12,1)
    arr0=[0,]*3
    arr1=[1,]*3
    arr2=[2,]*3
    arr3=[3,]*3
    temp1 = sum([arr0,arr1,arr2,arr3],[])
    result1 = sc.reorderConvertGroups(G,temp1)
    for i in range(3):
        assert(result1[i]==0)
    for i in range(3,6):
        assert(result1[i]==1)
    for i in range(6,9):
        assert(result1[i]==2)
    for i in range(9,12):
        assert(result1[i]==3)


def test_reorder4():
    # confirm that it stays in the same group if all equal
    rd.seed(107)
    np.random.seed(108)
    G = nx.DiGraph()
    G.add_nodes_from(range(120))
    for i in range(60,90):
        for j in range(30,120):
            G.add_edge(i,j)
    arr0=[0,]*30
    arr1=[1,]*30
    arr2=[2,]*30
    arr3=[3,]*30
    for item in it.permutations([arr0,arr1,arr2,arr3]):
        print('hi i am starting')
        temp1 = sum(item,[])
        result1 = sc.reorderConvertGroups(G,temp1)
        for i in range(30):
            assert(result1[i]==0)
        temp1=[]
        for i in range(30,60):
            temp1.append(result1[i])
        assert(len(set(temp1))==1)
        assert(temp1[0] in [1,3])
        for i in range(60,90):
            assert(result1[i]==2)
        temp1=[]
        for i in range(90,120):
            temp1.append(result1[i])
        assert(len(set(temp1))==1)
        assert(temp1[0] in [1,3])



def test_reorder5():
    # confirm that it stays in the same group if all equal
    rd.seed(109)
    np.random.seed(110)
    n=200
    for run in range(10):
        G = nx.erdos_renyi_graph(n,0.1)
        group = [rd.randint(0,3) for i in range(n)]
        result1 = sc.reorderConvertGroups(G,group)
        assert(len(group)==len(result1))
        for i in range(len(group)):
            for j in range(len(group)):
                if group[i]==group[j]:
                   assert(result1[i]==result1[j])
                else:
                   assert(result1[i]!=result1[j])




def test_reordering_checks_sizes():
    G = gg.syntheticModel1(100,0)
    with pytest.raises(Exception):
        result1 = sc.reorderConvertGroups(G,[0,]*200)



